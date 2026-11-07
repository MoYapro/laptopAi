"""Streaming chat client for Ollama."""

import json
from collections.abc import AsyncIterator

import httpx


class Ollama:
    def __init__(self, url: str, model: str, think: bool | None):
        self.url, self.model, self.think = url.rstrip("/"), model, think
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=300))

    async def chat(self, messages: list[dict], temperature: float | None = None) -> AsyncIterator[dict]:
        """Yields Ollama's NDJSON chunks; the last one has done=True and timing stats."""
        body: dict = {"model": self.model, "messages": messages, "stream": True}
        if self.think is not None:
            body["think"] = self.think
        if temperature is not None:
            body["options"] = {"temperature": temperature}
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
    ) -> str:
        """Non-streaming chat call; with a JSON schema, Ollama constrains the output to it."""
        body: dict = {"model": self.model, "messages": messages, "stream": False}
        if self.think is not None:
            body["think"] = self.think
        if schema:
            body["format"] = schema
        options = {}
        if max_tokens:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if options:
            body["options"] = options
        resp = await self._client.post(f"{self.url}/api/chat", json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama {resp.status_code}: {resp.text}")
        return resp.json()["message"]["content"]

    async def load(self) -> None:
        """Ask Ollama to load the model into memory (a generate call without a prompt)."""
        resp = await self._client.post(f"{self.url}/api/generate", json={"model": self.model})
        resp.raise_for_status()

    async def ping(self) -> bool:
        try:
            return (await self._client.get(f"{self.url}/api/version")).status_code == 200
        except httpx.HTTPError:
            return False
