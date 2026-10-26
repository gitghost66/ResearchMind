"""
URL normalisation for de-duplicating search results and sources.

Two URLs that differ only in case of the host, a default port, a fragment, a
trailing slash, or tracking parameters point at the same page and should be
fetched and stored once.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

__all__ = ["normalize_url", "domain_of", "is_http_url"]

_TRACKING_PREFIXES = ("utm_",)
_TRACKING_PARAMS = frozenset({"gclid", "fbclid", "msclkid", "mc_cid", "mc_eid", "ref_src"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def is_http_url(url: str) -> bool:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False
    return parts.scheme.lower() in _DEFAULT_PORTS and bool(parts.hostname)


def normalize_url(url: str) -> str:
    """
    Canonical form of *url* for de-duplication.

    Returns the stripped input unchanged if it cannot be parsed as http(s).
    """
    raw = url.strip()
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        return raw
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in _DEFAULT_PORTS or not host:
        return raw

    netloc = host
    if port is not None and port != _DEFAULT_PORTS[scheme]:
        netloc = f"{host}:{port}"

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKING_PREFIXES) and key.lower() not in _TRACKING_PARAMS
    ]
    return urlunsplit((scheme, netloc, path, urlencode(query_pairs), ""))


def domain_of(url: str) -> str:
    """Host of *url* without a leading ``www.``; empty if unparseable."""
    try:
        host = (urlsplit(url.strip()).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host
