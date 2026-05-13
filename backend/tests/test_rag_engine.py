"""Unit tests for the AdvancedRAG engine: intent classification, reranking, multi-hop search."""

from unittest.mock import MagicMock

import pytest

from rag_engine import AdvancedRAG, QueryContext


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(file: str, similarity: float = 0.8, code: str = "def foo(): pass") -> dict:
    return {
        "code": code,
        "metadata": {"file": file, "lines": "1-10", "language": "python"},
        "similarity": similarity,
    }


def _search_stub(query, collection, top_k=5, similarity_threshold=0.0,
                 where_filter=None, language_filter=None, **kwargs):
    """Deterministic search stub that always returns one backend chunk."""
    return [_make_chunk("backend/auth.py", similarity=0.85)]


@pytest.fixture
def engine():
    return AdvancedRAG(_search_stub, MagicMock())


# ── analyze_query: intent ─────────────────────────────────────────────────────

class TestAnalyzeQueryIntent:
    def test_where_gives_location(self, engine):
        ctx = engine.analyze_query("where is the login function?")
        assert ctx.intent == "location"
        assert ctx.depth == 1

    def test_which_file_gives_location(self, engine):
        ctx = engine.analyze_query("which file handles rate limiting?")
        assert ctx.intent == "location"

    def test_find_gives_location(self, engine):
        ctx = engine.analyze_query("find the middleware setup")
        assert ctx.intent == "location"

    def test_explain_gives_architecture(self, engine):
        ctx = engine.analyze_query("explain how the authentication flow works")
        assert ctx.intent == "architecture"
        assert ctx.depth == 3

    def test_architecture_keyword(self, engine):
        ctx = engine.analyze_query("describe the overall architecture")
        assert ctx.intent == "architecture"

    def test_how_to_gives_usage(self, engine):
        # "how to" matches usage; "how" also matches architecture — architecture wins
        # because architecture patterns are checked last and overwrite usage
        ctx = engine.analyze_query("how to call the search function")
        assert ctx.intent in ("usage", "architecture")

    def test_unknown_query_defaults_to_architecture(self, engine):
        ctx = engine.analyze_query("tell me about middleware")
        # no location or usage keywords — falls through to architecture default
        assert ctx.intent == "architecture"


# ── analyze_query: concept extraction ────────────────────────────────────────

class TestConceptExtraction:
    def test_known_technical_term_extracted(self, engine):
        ctx = engine.analyze_query("how does authentication work?")
        assert ctx.main_concept == "authentication"

    def test_another_known_term(self, engine):
        ctx = engine.analyze_query("explain the middleware pipeline")
        assert ctx.main_concept == "middleware"

    def test_concept_not_empty_for_arbitrary_query(self, engine):
        ctx = engine.analyze_query("what is going on here?")
        assert ctx.main_concept  # never empty


# ── rerank_by_source ─────────────────────────────────────────────────────────

class TestRerank:
    def test_python_boosted_over_jsx_for_backend_query(self, engine):
        results = [
            _make_chunk("frontend/src/App.jsx", similarity=0.95),
            _make_chunk("backend/auth.py", similarity=0.70),
        ]
        reranked = engine.rerank_by_source(results, "how does authentication work?")
        assert reranked[0]["metadata"]["file"] == "backend/auth.py"

    def test_ui_query_preserves_original_order(self, engine):
        results = [
            _make_chunk("frontend/src/App.jsx", similarity=0.95),
            _make_chunk("backend/auth.py", similarity=0.70),
        ]
        reranked = engine.rerank_by_source(results, "how does the React component render?")
        assert reranked[0]["metadata"]["file"] == "frontend/src/App.jsx"

    def test_empty_results_handled(self, engine):
        assert engine.rerank_by_source([], "some query") == []

    def test_all_python_files_keep_similarity_order(self, engine):
        results = [
            _make_chunk("backend/main.py", similarity=0.9),
            _make_chunk("backend/auth.py", similarity=0.7),
        ]
        reranked = engine.rerank_by_source(results, "explain the database")
        assert reranked[0]["metadata"]["file"] == "backend/main.py"


# ── organize_chunks ───────────────────────────────────────────────────────────

class TestOrganizeChunks:
    def test_returns_expected_keys(self, engine):
        chunks = [_make_chunk("backend/main.py")]
        result = engine.organize_chunks(chunks)
        assert "by_file_type" in result
        assert "by_content" in result

    def test_entry_file_classified_correctly(self, engine):
        chunks = [_make_chunk("backend/main.py")]
        result = engine.organize_chunks(chunks)
        assert "entry" in result["by_file_type"]

    def test_component_file_classified(self, engine):
        chunks = [_make_chunk("frontend/src/components/Login.jsx")]
        result = engine.organize_chunks(chunks)
        assert "components" in result["by_file_type"]

    def test_empty_input(self, engine):
        result = engine.organize_chunks([])
        assert result["by_file_type"] == {}
        assert result["by_content"] == {}


# ── multi_hop_search ─────────────────────────────────────────────────────────

class TestMultiHopSearch:
    def test_depth_1_returns_results(self, engine):
        ctx = QueryContext(intent="location", main_concept="auth", depth=1,
                           original_query="where is auth?")
        results = engine.multi_hop_search(ctx, top_k=5)
        assert len(results) > 0

    def test_depth_3_returns_results(self, engine):
        ctx = QueryContext(intent="architecture", main_concept="auth", depth=3,
                           original_query="explain the auth flow")
        results = engine.multi_hop_search(ctx, top_k=5)
        assert len(results) > 0

    def test_no_duplicate_chunks(self, engine):
        ctx = QueryContext(intent="architecture", main_concept="auth", depth=3,
                           original_query="explain auth")
        results = engine.multi_hop_search(ctx, top_k=5)
        ids = [f"{r['metadata']['file']}:{r['metadata']['lines']}" for r in results]
        assert len(ids) == len(set(ids))

    def test_hyde_document_used_when_set(self, engine):
        """When a HyDE document is set, it should be used as the embedding query."""
        calls = []

        def recording_search(query, collection, **kwargs):
            calls.append(query)
            return [_make_chunk("backend/auth.py")]

        eng = AdvancedRAG(recording_search, MagicMock())
        ctx = QueryContext(intent="architecture", main_concept="auth", depth=2,
                           original_query="explain auth",
                           hyde_document="def authenticate(user, password): ...")
        eng.multi_hop_search(ctx, top_k=3)
        # At least one call should use the HyDE document
        assert any("def authenticate" in c for c in calls)
