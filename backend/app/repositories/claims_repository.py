"""
Repositories for Claim, ClaimEvidence, and Contradiction.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.claims import (
    Claim,
    ClaimEvidence,
    Contradiction,
    ContradictionClaim,
)
from app.repositories.base import BaseRepository


class ClaimRepository(BaseRepository[Claim]):
    """Repository handling atomic factual claims."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Claim, session)

    async def list_by_run(
        self,
        run_id: uuid.UUID | str,
        include_evidence: bool = True,
    ) -> Sequence[Claim]:
        """List all claims for a run with optional evidence passages pre-loaded."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        stmt = select(Claim).where(Claim.run_id == uid).order_by(Claim.created_at)
        if include_evidence:
            stmt = stmt.options(
                selectinload(Claim.evidence_links).selectinload(ClaimEvidence.passage),
                selectinload(Claim.contradiction_links),
            )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_verdict(
        self,
        claim_id: uuid.UUID | str,
        verdict: str,
        confidence: float,
    ) -> Claim | None:
        """Update fact-checker verdict and confidence score for a claim."""
        uid = uuid.UUID(str(claim_id)) if not isinstance(claim_id, uuid.UUID) else claim_id
        return await self.update(uid, verdict=verdict, confidence=confidence)


class ClaimEvidenceRepository(BaseRepository[ClaimEvidence]):
    """Repository handling many-to-many claim-to-passage evidence links."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(ClaimEvidence, session)

    async def link_evidence(
        self,
        claim_id: uuid.UUID | str,
        passage_id: uuid.UUID | str,
        stance: str = "supports",
    ) -> ClaimEvidence:
        """Link a claim to an evidence passage with stance (supports/contradicts/neutral)."""
        c_id = uuid.UUID(str(claim_id)) if not isinstance(claim_id, uuid.UUID) else claim_id
        p_id = uuid.UUID(str(passage_id)) if not isinstance(passage_id, uuid.UUID) else passage_id

        link = ClaimEvidence(claim_id=c_id, passage_id=p_id, stance=stance)
        self.session.add(link)
        await self.session.flush()
        return link

    async def get_for_claim(self, claim_id: uuid.UUID | str) -> Sequence[ClaimEvidence]:
        """Fetch all evidence links for a claim with passages preloaded."""
        c_id = uuid.UUID(str(claim_id)) if not isinstance(claim_id, uuid.UUID) else claim_id
        stmt = (
            select(ClaimEvidence)
            .where(ClaimEvidence.claim_id == c_id)
            .options(selectinload(ClaimEvidence.passage))
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()


class ContradictionRepository(BaseRepository[Contradiction]):
    """Repository handling contradictions between claims."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Contradiction, session)

    async def create_with_claims(
        self,
        run_id: uuid.UUID | str,
        explanation: str,
        likely_cause: str,
        claim_ids: list[uuid.UUID | str],
    ) -> Contradiction:
        """Record a contradiction and link it to the conflicting claims."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        contradiction = Contradiction(
            run_id=uid,
            explanation=explanation,
            likely_cause=likely_cause,
        )
        self.session.add(contradiction)
        await self.session.flush()

        # Link conflicting claims
        for cid in claim_ids:
            c_uid = uuid.UUID(str(cid)) if not isinstance(cid, uuid.UUID) else cid
            link = ContradictionClaim(
                contradiction_id=contradiction.id,
                claim_id=c_uid,
            )
            self.session.add(link)

        await self.session.flush()
        return contradiction

    async def list_by_run(self, run_id: uuid.UUID | str) -> Sequence[Contradiction]:
        """Fetch all contradictions for a run with linked claims preloaded."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        stmt = (
            select(Contradiction)
            .where(Contradiction.run_id == uid)
            .options(
                selectinload(Contradiction.claim_links).selectinload(ContradictionClaim.claim)
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
