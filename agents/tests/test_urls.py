"""M3 tests — URL normalisation for de-duplication."""

from __future__ import annotations

import pytest

from researchmind_agents.tools.urls import domain_of, is_http_url, normalize_url


class TestNormalizeUrl:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("HTTPS://Example.COM/Path", "https://example.com/Path"),
            ("https://example.com:443/a", "https://example.com/a"),
            ("http://example.com:80/a", "http://example.com/a"),
            ("https://example.com:8443/a", "https://example.com:8443/a"),
            ("https://example.com/a#section", "https://example.com/a"),
            ("https://example.com/a/", "https://example.com/a"),
            ("https://example.com", "https://example.com/"),
            ("https://example.com/", "https://example.com/"),
            ("  https://example.com/a  ", "https://example.com/a"),
        ],
    )
    def test_canonical_forms(self, raw: str, expected: str) -> None:
        assert normalize_url(raw) == expected

    def test_strips_tracking_params_keeps_others(self) -> None:
        url = "https://example.com/a?id=7&utm_source=x&UTM_Medium=y&gclid=z&fbclid=w&page=2"
        assert normalize_url(url) == "https://example.com/a?id=7&page=2"

    def test_variants_collapse_to_one(self) -> None:
        variants = [
            "https://Example.com/article/",
            "https://example.com:443/article#top",
            "https://example.com/article?utm_campaign=spring",
        ]
        assert len({normalize_url(v) for v in variants}) == 1

    @pytest.mark.parametrize(
        "raw",
        ["ftp://example.com/file", "mailto:someone@example.com", "not a url", "https://"],
    )
    def test_non_http_returned_as_is(self, raw: str) -> None:
        assert normalize_url(raw) == raw

    def test_unparseable_port_returned_as_is(self) -> None:
        assert normalize_url("https://example.com:notaport/a") == "https://example.com:notaport/a"


class TestDomainOf:
    def test_strips_www_and_lowercases(self) -> None:
        assert domain_of("https://WWW.Example.com/a") == "example.com"

    def test_keeps_other_subdomains(self) -> None:
        assert domain_of("https://docs.example.com/a") == "docs.example.com"

    def test_unparseable_is_empty(self) -> None:
        assert domain_of("not a url") == ""


class TestIsHttpUrl:
    @pytest.mark.parametrize("url", ["http://a.example", "https://a.example/x?y=1"])
    def test_accepts_http(self, url: str) -> None:
        assert is_http_url(url)

    @pytest.mark.parametrize(
        "url", ["ftp://a.example", "javascript:alert(1)", "", "https://", "/relative/path"]
    )
    def test_rejects_others(self, url: str) -> None:
        assert not is_http_url(url)
