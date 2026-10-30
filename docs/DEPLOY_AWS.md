# Deploying ResearchMind on AWS

ResearchMind is one container image with three roles:

| Role | Command | Needs |
|---|---|---|
| API | default (`uvicorn app.main:app`) | port 8000, the database |
| Worker | `python -m app.workers.research_worker` | the database, outbound internet (Gemini, search, pages) |
| Migrations | `alembic upgrade head` | the database; run once per release, before the others |

The only shared state is PostgreSQL. Runs are queued in it, workers claim them
under a lease, and every step is checkpointed, so API tasks and workers can be
added, replaced, or killed at any time without losing runs (see *Durability* in
the README).

Two ways to run it:

* **A. One EC2 instance with Docker Compose** — simplest, cheapest; good for a
  demo or a small team.
* **B. ECS Fargate** — the API and the workers as separate services behind an
  Application Load Balancer; rolling deploys and scaling per role.

Both use a managed PostgreSQL: **Amazon RDS** or **Supabase**.

---

## 1. Database

### Amazon RDS for PostgreSQL

1. Create an RDS PostgreSQL **16** instance (`db.t4g.micro` is enough to start)
   in private subnets, not publicly accessible. pgvector is available on RDS;
   the first migration runs `CREATE EXTENSION IF NOT EXISTS vector`, which the
   master user may do.
2. Security group: allow 5432 only from the application's security group.
3. Connection settings for the app:
   ```
   DATABASE_URL=postgresql+asyncpg://<user>:<password>@<endpoint>:5432/<db>
   ```
   `DATABASE_LISTEN_URL` can stay empty: a direct connection supports
   `LISTEN/NOTIFY`. Percent-encode special characters in the password.

### Supabase

Use the pooler URLs from *Project Settings → Database*:
```
# Transaction pooler: the app's queries
DATABASE_URL=postgresql+asyncpg://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres
# Session pooler (same host, port 5432): LISTEN/NOTIFY between processes
DATABASE_LISTEN_URL=postgresql+asyncpg://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
```
Without `DATABASE_LISTEN_URL`, everything still works, but event streams and
cancels between processes fall back to polling (a second or two of delay).

---

## 2. Secrets and settings

Keep secrets out of the image (`.dockerignore` already excludes every `.env`).
Store these in **AWS Secrets Manager** or **SSM Parameter Store** (SecureString):

| Name | Value |
|---|---|
| `GEMINI_API_KEY` | from Google AI Studio |
| `API_KEY` | a long random string; clients send it as `X-API-Key` |
| `DATABASE_URL` | from step 1 |
| `DATABASE_LISTEN_URL` | Supabase only |

Production settings (plain environment variables; `.env.example` documents all
of them):

```
RUN_WORKER_IN_API=false        # workers run as their own service
CORS_ALLOW_ORIGINS=https://your-frontend.example
DEBUG=false
LOG_FORMAT=json
MAX_CONCURRENT_RUNS=2          # per worker
MAX_PENDING_RUNS=20            # across all workers
LLM_PRICE_PER_1M_INPUT=...     # on a paid Gemini tier, so MAX_COST_PER_RUN_USD applies
LLM_PRICE_PER_1M_OUTPUT=...
```

---

## 3. Build and push the image (both options)

```bash
AWS_REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/researchmind

aws ecr create-repository --repository-name researchmind --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${REPO%/*}

# From the repository root. Fargate runs linux/amd64 unless you pick ARM.
docker build --platform linux/amd64 -f backend/Dockerfile -t $REPO:$(git rev-parse --short HEAD) .
docker push $REPO:$(git rev-parse --short HEAD)
```

---

## 4A. One EC2 instance with Docker Compose

1. Launch an instance: Amazon Linux 2023, `t3.small` or larger, 20 GB disk, in
   a public subnet. Security group: 80 and 443 from anywhere, 22 from your IP.
   Give it an instance role allowed to read your secrets.
2. Install Docker and the Compose plugin:
   ```bash
   sudo dnf install -y docker git
   sudo systemctl enable --now docker
   sudo usermod -aG docker ec2-user   # log out and in again
   sudo mkdir -p /usr/local/lib/docker/cli-plugins
   sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
     -o /usr/local/lib/docker/cli-plugins/docker-compose
   sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
   ```
3. Get the code and write `backend/.env` (from your secrets, never committed):
   ```bash
   git clone <repository> researchmind && cd researchmind
   cp .env.example backend/.env && chmod 600 backend/.env
   # fill in GEMINI_API_KEY, API_KEY, DATABASE_URL (+ DATABASE_LISTEN_URL), CORS_ALLOW_ORIGINS
   ```
4. Start it (builds the image on the instance; or set `image:` in the compose
   file to your ECR image and `docker compose pull`):
   ```bash
   docker compose -f docker-compose.prod.yml up -d --build
   docker compose -f docker-compose.prod.yml ps
   curl http://localhost:8080/ready
   ```
   `DATABASE_URL` comes from `backend/.env` (do not add
   `docker-compose.localdb.yml` here: it replaces it with a local container).
   The `migrate` service applies migrations first; the API and workers start
   once it succeeds. `restart: unless-stopped` and the enabled Docker service
   bring everything back after a reboot. Nginx listens on `HTTP_PORT` (8080 by
   default); set `HTTP_PORT=80` in the shell, or put an ALB in front.
5. **TLS.** Either an Application Load Balancer with an ACM certificate
   forwarding to the instance's port (simplest; set the ALB idle timeout to 120
   s), or certbot on the instance with the commented HTTPS block in
   `deploy/nginx/nginx.conf`. Behind an ALB, enable `set_real_ip_from` /
   `real_ip_header` in that file so rate limits see client addresses.
6. **Scale** on the same machine: `API_PROCESSES=4 WORKER_REPLICAS=3 docker
   compose -f docker-compose.prod.yml up -d`.
7. **Logs:** `docker compose -f docker-compose.prod.yml logs -f worker`, or add
   the `awslogs` logging driver to ship them to CloudWatch.
8. **Updates:** `git pull && docker compose -f docker-compose.prod.yml up -d
   --build`. Workers stopped by the update hand their runs back (SIGTERM, 30 s
   grace) and the new workers continue them from their checkpoints.

---

## 4B. ECS Fargate

### Network

* VPC with public subnets (ALB) and private subnets (tasks, RDS), and a NAT
  gateway: workers need outbound internet for Gemini, search, and page fetches.
* Security groups: ALB ← 443 from anywhere; API tasks ← 8000 from the ALB;
  RDS ← 5432 from the tasks.

### Task definitions

One image, three task definitions. The API task (abridged):

```json
{
  "family": "researchmind-api",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<account>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "<account>.dkr.ecr.<region>.amazonaws.com/researchmind:<tag>",
      "portMappings": [{ "containerPort": 8000 }],
      "environment": [
        { "name": "RUN_WORKER_IN_API", "value": "false" },
        { "name": "WEB_CONCURRENCY", "value": "2" },
        { "name": "FORWARDED_ALLOW_IPS", "value": "*" },
        { "name": "CORS_ALLOW_ORIGINS", "value": "https://your-frontend.example" }
      ],
      "secrets": [
        { "name": "GEMINI_API_KEY", "valueFrom": "arn:aws:secretsmanager:...:GEMINI_API_KEY" },
        { "name": "API_KEY", "valueFrom": "arn:aws:secretsmanager:...:API_KEY" },
        { "name": "DATABASE_URL", "valueFrom": "arn:aws:secretsmanager:...:DATABASE_URL" }
      ],
      "healthCheck": {
        "command": ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"],
        "interval": 15, "timeout": 5, "retries": 3, "startPeriod": 20
      },
      "stopTimeout": 30,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/researchmind/api",
          "awslogs-region": "<region>",
          "awslogs-stream-prefix": "api"
        }
      }
    }
  ]
}
```

* **Worker** (`researchmind-worker`): the same, with
  `"command": ["python", "-m", "app.workers.research_worker"]`, no port mapping,
  no health check, `"stopTimeout": 30`, and its own log group. Size it for the
  runs it executes at once (`MAX_CONCURRENT_RUNS`, 1 vCPU / 2 GB for 2 runs).
* **Migrations** (`researchmind-migrate`): the same image with
  `"command": ["alembic", "upgrade", "head"]`.

`FORWARDED_ALLOW_IPS="*"` is safe only because the API tasks accept
traffic from the ALB alone (security group).

### Services

```bash
# 1. Migrations, before every release that has new ones
aws ecs run-task --cluster researchmind --launch-type FARGATE \
  --task-definition researchmind-migrate \
  --network-configuration "awsvpcConfiguration={subnets=[<private>],securityGroups=[<tasks>]}"

# 2. API behind the ALB, 2 tasks
aws ecs create-service --cluster researchmind --service-name api \
  --task-definition researchmind-api --desired-count 2 --launch-type FARGATE \
  --load-balancers "targetGroupArn=<tg>,containerName=api,containerPort=8000" \
  --network-configuration "awsvpcConfiguration={subnets=[<private>],securityGroups=[<tasks>]}"

# 3. Workers, 2 tasks
aws ecs create-service --cluster researchmind --service-name worker \
  --task-definition researchmind-worker --desired-count 2 --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<private>],securityGroups=[<tasks>]}"
```

**Load balancer.** HTTPS listener with an ACM certificate, target group on
port 8000 with health check `GET /health`. Set the ALB **idle timeout to 120 s**:
event streams send a keep-alive comment every 15 s, and an ALB does not
buffer them. Deregistration delay 30 s.

**Deploys.** ECS stops old tasks with SIGTERM and waits `stopTimeout` (30 s).
A worker interrupts its runs and hands them back; a new worker resumes them
from their checkpoints at once. A task that is killed instead (out of memory,
host failure) stops renewing its lease, and its runs are resumed after
`RUN_LEASE_SECONDS`.

**Scaling.** API tasks hold no run state, so scale them on CPU or request
count. Scale workers on the number of queued runs (`GET /ready` reports
`runs.queued`), or simply on CPU. Rate limits are per API process; for a strict
global limit put one in front (AWS WAF rate-based rules).

---

## 5. Check the deployment

From your machine:

```bash
python backend/scripts/smoke_test.py --base-url https://research.example.com --api-key <API_KEY> api
```

It checks health, readiness, auth and error envelopes, one real quick-mode run
streamed to completion, every read endpoint, cancel, resume, and delete. To
see a run survive its worker, kill the workers mid-run and start new ones:

```bash
# EC2 / Compose: a hard kill (nothing is handed back)
python backend/scripts/smoke_test.py --base-url https://research.example.com --api-key <API_KEY> restart \
  --kill "ssh ec2-user@<host> docker compose -f researchmind/docker-compose.prod.yml kill -s SIGKILL worker" \
  --start "ssh ec2-user@<host> docker compose -f researchmind/docker-compose.prod.yml up -d worker"

# ECS: stopping tasks is a clean stop (SIGTERM): runs are handed back at once
python backend/scripts/smoke_test.py --base-url https://research.example.com --api-key <API_KEY> restart \
  --kill "for t in \$(aws ecs list-tasks --cluster researchmind --service-name worker --query 'taskArns[]' --output text); do aws ecs stop-task --cluster researchmind --task \$t; done"
```

Find one run's whole story in CloudWatch Logs Insights:

```
fields @timestamp, worker_id, logger, message
| filter run_id = "<run id>"
| sort @timestamp asc
```

---

## 6. Checklist

- [ ] `API_KEY` set; `DEBUG=false`; `CORS_ALLOW_ORIGINS` lists your frontend only
- [ ] Secrets in Secrets Manager / SSM, not in the image or the repository
- [ ] Database not publicly reachable; TLS on the public endpoint
- [ ] Migrations run before the new release starts
- [ ] Workers can reach the internet (NAT); the API needs no outbound access
- [ ] On a paid Gemini tier, `LLM_PRICE_PER_1M_*` set so the per-run cost cap works
- [ ] Smoke test passes against the public URL
