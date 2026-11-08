"""look_up: short factual answers from the internet for the small model.

1. DuckDuckGo Instant Answer API (official, no key; mostly Wikipedia abstracts)
2. Wikipedia search + page summary as a fallback
Both only count if the article title actually matches the term: Wikipedia's search always returns *some*
article ("ArcFace" -> "Arc de Triomphe"), and a small model would repeat that as fact.
Results are cached in the database for LOOKUP_CACHE_DAYS.
"""

import logging
import re
import time
import urllib.parse

import httpx

import config

log = logging.getLogger("lookup")

# Wikimedia's user-agent policy asks for contact information (email or URL). Without LOOKUP_CONTACT the
# Wikipedia fallback stays off and only DuckDuckGo is used.
USER_AGENT = "laptopai-housemate/0.1 (private home assistant" + (f"; {config.LOOKUP_CONTACT}" if config.LOOKUP_CONTACT else "") + ")"
GENERIC = set("""a an the of in on at for to and or is what who whats whos how about tell me please
game video wiki wikipedia definition meaning""".split())

TOOL = {
    "type": "function",
    "function": {
        "name": "look_up",
        "description": (
            "Look up a short factual description of a thing, concept, place, game, sport, animal, technology or "
            "public figure on the internet (Wikipedia/DuckDuckGo). Use it when you don't know something or aren't "
            "sure. Not for the people in this home, and not for live information like weather, news or prices."
        ),
        "parameters": {
            "type": "object",
            "properties": {"term": {"type": "string", "description": (
                "just the name of the thing, without the question, e.g. 'Hades video game' "
                "(not 'Hades video game genre' or 'what is Hades')")}},
            "required": ["term"],
        },
    },
}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9äöüß]+", text.lower()))


def relevant(title: str, term: str) -> bool:
    """The article's core title must be part of the term, and the title must cover most of the term."""
    core = _words(re.sub(r"\(.*?\)", "", title)) - GENERIC
    term_words = _words(term) - GENERIC
    if not core or not term_words or not core <= _words(term):
        return False
    return len(_words(title) & term_words) / len(term_words) >= 0.75


def _short(text: str, max_chars: int = 420) -> str:
    """At most two sentences / max_chars, so the small model gets a compact answer."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(sentences[:2])
    return out if len(out) <= max_chars else out[: max_chars].rsplit(" ", 1)[0] + "…"


class Lookup:
    def __init__(self, db):
        self.db = db
        self._client = httpx.AsyncClient(timeout=10, headers={"User-Agent": USER_AGENT}, follow_redirects=True)

    async def look_up(self, term: str) -> dict:
        term = " ".join(term.split())[:120]
        if not term:
            return {"found": False, "term": term, "reason": "empty term"}
        cached = self.db.cached_lookup(term, time.time() - config.LOOKUP_CACHE_DAYS * 86400)
        if cached:
            return {**cached, "cached": True}
        result = None
        sources = [self._duckduckgo] + ([self._wikipedia] if config.LOOKUP_CONTACT else [])
        # Small models tend to put the question into the term ("Hades video game genre"):
        # if the full term finds nothing, try it with trailing words dropped.
        words = term.split()
        candidates = [" ".join(words[:n]) for n in range(len(words), max(0, len(words) - 3), -1)]
        for candidate in candidates:
            for source in sources:
                try:
                    result = await source(candidate)
                except (httpx.HTTPError, ValueError) as e:  # network errors, error pages instead of JSON
                    log.warning("lookup via %s failed: %s", source.__name__, e)
                if result:
                    break
            if result:
                break
        if not result:  # misses aren't cached, so a later attempt can still succeed
            return {"found": False, "term": term, "reason": "nothing reliable found"}
        result["term"] = term
        self.db.store_lookup(term, result)
        return result

    async def _duckduckgo(self, term: str) -> dict | None:
        resp = await self._client.get("https://api.duckduckgo.com/", params={
            "q": term, "format": "json", "no_html": 1, "skip_disambig": 1, "no_redirect": 1,
        })
        resp.raise_for_status()
        d = resp.json()
        if d.get("AbstractText") and relevant(d.get("Heading", ""), term):
            return {"found": True, "text": _short(d["AbstractText"]), "title": d.get("Heading"),
                    "source": d.get("AbstractSource") or "DuckDuckGo", "url": d.get("AbstractURL")}
        if d.get("Definition") and relevant(d.get("Heading", "") or term, term):
            return {"found": True, "text": _short(d["Definition"]), "title": d.get("Heading"),
                    "source": d.get("DefinitionSource") or "DuckDuckGo", "url": d.get("DefinitionURL")}
        return None

    async def _wikipedia(self, term: str) -> dict | None:
        resp = await self._client.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": term, "srlimit": 3, "format": "json",
        })
        resp.raise_for_status()
        for hit in resp.json().get("query", {}).get("search", []):
            if not relevant(hit["title"], term):
                continue
            path = urllib.parse.quote(hit["title"].replace(" ", "_"))
            page = await self._client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{path}")
            page.raise_for_status()
            summary = page.json()
            if summary.get("extract") and summary.get("type") != "disambiguation":
                return {"found": True, "text": _short(summary["extract"]), "title": hit["title"], "source": "Wikipedia",
                        "url": summary.get("content_urls", {}).get("desktop", {}).get("page")}
        return None
