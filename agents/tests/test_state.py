"""Tests for the agents package — state model validation."""

from researchmind_agents.state import ResearchState


def test_research_state_defaults():
    """ResearchState should initialize with sensible defaults."""
    state = ResearchState()
    assert state.mode == "quick"
    assert state.iteration == 0
    assert state.max_iterations == 3
    assert state.sources == []
    assert state.claims == []


def test_research_state_custom_values():
    """ResearchState should accept custom values."""
    state = ResearchState(
        run_id="test-123",
        question="Is RAG better than fine-tuning?",
        mode="deep",
        max_iterations=5,
    )
    assert state.run_id == "test-123"
    assert state.question == "Is RAG better than fine-tuning?"
    assert state.mode == "deep"
    assert state.max_iterations == 5
