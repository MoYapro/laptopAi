"""Ollama client: streaming chat for conversation, non-streaming (optionally JSON-constrained) calls."""

import json
from collections.abc import AsyncIterator

import httpx


class Ollama:
    def __init__(self, url: str, model: str, think: bool | None, num_ctx: int | None = None):
        self.url, self.model, self.think, self.num_ctx = url.rstrip("/"), model, think, num_ctx
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=900))

    def _body(self, messages, model, think, stream, options, keep_alive) -> dict:
        body: dict = {"model": model or self.model, "messages": messages, "stream": stream}
        think = self.think if think is None else think
        if think is not None:
            body["think"] = think
        opts = {"num_ctx": self.num_ctx} if self.num_ctx else {}
        opts.update({k: v for k, v in (options or {}).items() if v is not None})
        if opts:
            body["options"] = opts
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        return body

    async def chat(self, messages: list[dict], temperature: float | None = None) -> AsyncIterator[dict]:
        """Yields Ollama's NDJSON chunks; the last one has done=True and timing stats."""
        body = self._body(messages, None, None, True, {"temperature": temperature}, None)
        async with self._client.stream("POST", f"{self.url}/api/chat", json=body) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode(errors="replace")
                raise RuntimeError(f"Ollama {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if line.strip():
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise RuntimeError(f"Ollama: {chunk['error']}")
                    yield chunk

    async def complete(
        self, messages: list[dict], schema: dict | None = None,
        max_tokens: int | None = None, temperature: float | None = None,
        model: str | None = None, think: bool | None = None,
        options: dict | None = None, keep_alive: str | None = None,
    ) -> str:
        """Non-streaming chat call; with a JSON schema, Ollama constrains the output to it."""
        opts = {"num_predict": max_tokens, "temperature": temperature, **(options or {})}
        body = self._body(messages, model, think, False, opts, keep_alive)
        if schema:
            body["format"] = schema
        resp = await self._client.post(f"{self.url}/api/chat", json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama {resp.status_code}: {resp.text}")
        return resp.json()["message"]["content"]

    async def load(self) -> None:
        """Ask Ollama to load the model into memory (a generate call without a prompt)."""
        body = {"model": self.model}
        if self.num_ctx:
            body["options"] = {"num_ctx": self.num_ctx}
        resp = await self._client.post(f"{self.url}/api/generate", json=body)
        resp.raise_for_status()

    async def has_model(self, model: str) -> bool:
        resp = await self._client.post(f"{self.url}/api/show", json={"model": model})
        return resp.status_code == 200

    async def ping(self) -> bool:
        try:
            return (await self._client.get(f"{self.url}/api/version")).status_code == 200
        except httpx.HTTPError:
            return False
