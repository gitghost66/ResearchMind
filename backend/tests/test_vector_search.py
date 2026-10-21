"""
Tests for vector similarity queries on passages.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.evidence_repository import (
    PassageRepository,
    SourceRepository,
    _cosine_distance,
)
from app.repositories.research_repository import (
    ResearchRunRepository,
    SubQuestionRepository,
)


def test_cosine_distance_calculation():
    """Verify mathematical properties of the cosine distance helper."""
    # Identical vectors -> distance 0
    v1 = [1.0, 0.0, 0.0]
    assert _cosine_distance(v1, v1) == 0.0

    # Orthogonal vectors -> distance 1
    v2 = [0.0, 1.0, 0.0]
    assert _cosine_distance(v1, v2) == 1.0

    # Opposite vectors -> distance 2
    v3 = [-1.0, 0.0, 0.0]
    assert _cosine_distance(v1, v3) == 2.0


@pytest.mark.asyncio
async def test_vector_similarity_search(db_session: AsyncSession):
    """Test passage similarity ranking and top-k retrieval."""
    run_repo = ResearchRunRepository(db_session)
    src_repo = SourceRepository(db_session)
    pass_repo = PassageRepository(db_session)

    run = await run_repo.create(question="Vector search ranking test")
    source = await src_repo.create(run_id=run.id, url="https://example.com/ai")

    # 3 passages with 768-dimensional embeddings
    # Target query will be aligned with target_vec
    target_vec = [1.0] + [0.0] * 767
    similar_vec = [0.9] + [0.1] * 767
    distant_vec = [0.0] + [1.0] + [0.0] * 766

    p_target = await pass_repo.create(
        source_id=source.id,
        text="Exact match topic passage",
        embedding=target_vec,
        chunk_index=0,
    )
    p_similar = await pass_repo.create(
        source_id=source.id,
        text="Close semantically related passage",
        embedding=similar_vec,
        chunk_index=1,
    )
    p_distant = await pass_repo.create(
        source_id=source.id,
        text="Completely unrelated topic passage",
        embedding=distant_vec,
        chunk_index=2,
    )

    # Query with target_vec
    results = await pass_repo.search_similar(
        query_embedding=target_vec,
        run_id=run.id,
        limit=3,
    )

    assert len(results) == 3
    # First result should be p_target with lowest distance
    first_passage, first_dist = results[0]
    assert first_passage.id == p_target.id
    assert first_dist < 0.01

    # Second result should be p_similar
    second_passage, second_dist = results[1]
    assert second_passage.id == p_similar.id
    assert second_dist > first_dist

    # Third result should be p_distant
    third_passage, third_dist = results[2]
    assert third_passage.id == p_distant.id
    assert third_dist > second_dist


@pytest.mark.asyncio
async def test_vector_similarity_search_scoped_by_sub_question(db_session: AsyncSession):
    """Test that similarity search respects sub_question_id filtering."""
    run_repo = ResearchRunRepository(db_session)
    sq_repo = SubQuestionRepository(db_session)
    src_repo = SourceRepository(db_session)
    pass_repo = PassageRepository(db_session)

    run = await run_repo.create(question="Scoped vector search")
    sq1 = await sq_repo.create(run_id=run.id, text="Sub question 1")
    sq2 = await sq_repo.create(run_id=run.id, text="Sub question 2")
    src = await src_repo.create(run_id=run.id, url="https://example.com")

    vec = [0.5] * 768

    await pass_repo.create(
        source_id=src.id,
        sub_question_id=sq1.id,
        text="Passage under SQ1",
        embedding=vec,
        chunk_index=0,
    )
    await pass_repo.create(
        source_id=src.id,
        sub_question_id=sq2.id,
        text="Passage under SQ2",
        embedding=vec,
        chunk_index=1,
    )

    # Query restricted to SQ1
    results = await pass_repo.search_similar(
        query_embedding=vec,
        sub_question_id=sq1.id,
        limit=5,
    )
    assert len(results) == 1
    assert results[0][0].sub_question_id == sq1.id
    assert results[0][0].text == "Passage under SQ1"
