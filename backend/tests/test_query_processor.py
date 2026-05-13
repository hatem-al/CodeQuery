"""Unit tests for typo correction and query suggestion utilities."""

import pytest
from utils.query_processor import fix_common_typos, suggest_alternatives


class TestFixCommonTypos:
    def test_known_typo_corrected(self):
        result, corrected, _ = fix_common_typos("authentification")
        assert corrected
        assert result == "authentication"

    def test_clean_query_unchanged(self):
        result, corrected, _ = fix_common_typos("authentication")
        assert not corrected
        assert result == "authentication"

    def test_uppercase_preserved(self):
        result, corrected, _ = fix_common_typos("AUTHENTIFICATION")
        assert corrected
        assert result == "AUTHENTICATION"

    def test_capitalized_preserved(self):
        result, corrected, _ = fix_common_typos("Authentification")
        assert corrected
        assert result == "Authentication"

    def test_multiple_typos_in_one_query(self):
        result, corrected, pairs = fix_common_typos("authentification databse")
        assert corrected
        assert "authentication" in result
        assert "database" in result
        assert len(pairs) == 2

    def test_typo_mid_sentence(self):
        result, corrected, _ = fix_common_typos("how does authentification work")
        assert corrected
        assert "authentication" in result
        assert "how does" in result
        assert "work" in result

    def test_empty_string(self):
        result, corrected, pairs = fix_common_typos("")
        assert not corrected
        assert result == ""
        assert pairs == []

    def test_no_typos_in_sentence(self):
        query = "where is the authentication middleware?"
        result, corrected, _ = fix_common_typos(query)
        assert not corrected
        assert result == query

    def test_corrections_list_records_pair(self):
        _, _, pairs = fix_common_typos("databse")
        assert len(pairs) == 1
        assert pairs[0] == ("databse", "database")

    def test_endpoint_typo(self):
        result, corrected, _ = fix_common_typos("endpoit")
        assert corrected
        assert result == "endpoint"


class TestSuggestAlternatives:
    def test_returns_list(self):
        suggestions = suggest_alternatives("middleware")
        assert isinstance(suggestions, list)

    def test_middleware_yields_suggestions(self):
        suggestions = suggest_alternatives("middleware")
        assert len(suggestions) > 0
        assert all("middleware" in s.lower() for s in suggestions)

    def test_authentication_yields_suggestions(self):
        suggestions = suggest_alternatives("authentication")
        assert len(suggestions) > 0

    def test_max_four_suggestions(self):
        suggestions = suggest_alternatives("authentication middleware database endpoint")
        assert len(suggestions) <= 4

    def test_no_duplicates(self):
        suggestions = suggest_alternatives("middleware")
        lower = [s.lower() for s in suggestions]
        assert len(lower) == len(set(lower))
