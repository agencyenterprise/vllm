# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A chat whose whole context is one live KV cache, editable between turns.

The conversation is a single resumable streaming request (``AsyncLLM.generate``
fed by an input stream): each user turn appends a chunk and the model replies
until it stops, then the request pauses with its KV cache intact. Between
turns the cache can be inspected and edited with KV surgery; the next turn
appends after whatever the cache holds by then.

Chunks are rendered with the model's chat template: each user turn sends the
part of the template rendering that the new message adds. The model's final
sampled token of a reply (normally the end-of-turn token) never reaches the
cache, so the next chunk re-adds the template's end-of-turn text first. The
transcript (``messages``) records what was said; the cache records what the
model has, and only the cache is changed by edits.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass

from transformers import AutoTokenizer

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.kv_surgery.ops import DropOp, KVEditResult, KVViewInfo

_SENTINEL = "\x00reply\x00"


def make_engine(
    model: str,
    max_model_len: int = 16384,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
) -> AsyncLLM:
    """An engine KV surgery accepts (prefix caching off)."""
    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=model,
            enable_prefix_caching=False,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    )


@dataclass
class Token:
    slot: int
    position: float
    token_id: int
    text: str


class ChatSession:
    def __init__(
        self,
        engine: AsyncLLM,
        tokenizer,
        request_id: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.request_id = request_id
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages: list[dict[str, str]] = []
        self._inputs: asyncio.Queue[StreamingInput | None] = asyncio.Queue()
        self._outputs: AsyncGenerator | None = None
        self._lock = asyncio.Lock()
        self._closing = self._end_of_turn_text()
        self._sampling_defaults = engine.model_config.get_diff_sampling_param()

    @classmethod
    def for_engine(cls, engine: AsyncLLM, request_id: str, **kwargs) -> "ChatSession":
        tokenizer = AutoTokenizer.from_pretrained(engine.model_config.tokenizer)
        return cls(engine, tokenizer, request_id, **kwargs)

    def _render(self, messages: list[dict[str, str]], add_generation_prompt: bool):
        if not messages:
            return ""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _end_of_turn_text(self) -> str:
        """What the template puts after an assistant reply's text."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": _SENTINEL},
        ]
        return self._render(messages, False).split(_SENTINEL, 1)[1]

    def _params(self, max_tokens: int) -> SamplingParams:
        return SamplingParams(
            **{**self._sampling_defaults, "max_tokens": max_tokens},
            output_kind=RequestOutputKind.DELTA,
        )

    async def _chunks(self):
        while (chunk := await self._inputs.get()) is not None:
            yield chunk

    async def _run_chunk(
        self, text: str, max_tokens: int, sink: asyncio.Queue[str | None] | None
    ) -> str:
        """Send one chunk and collect the reply until the request pauses.
        Runs as its own task, so a reader that goes away (a closed browser
        tab) cannot cancel the request mid-reply."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk = StreamingInput(
            {"prompt_token_ids": token_ids}, self._params(max_tokens)
        )
        if self._outputs is None:
            self._outputs = self.engine.generate(
                self._chunks(), self._params(max_tokens), self.request_id
            )
        self._inputs.put_nowait(chunk)
        reply = []
        while True:
            out = await anext(self._outputs)
            delta = out.outputs[0].text
            reply.append(delta)
            if sink is not None and delta:
                sink.put_nowait(delta)
            if out.outputs[0].finish_reason is not None:
                return "".join(reply)

    async def start(self) -> None:
        """Prefill the system prompt, if any (its one sampled token is
        discarded when the first user turn arrives)."""
        if self.system_prompt is None:
            return
        async with self._lock:
            system = [{"role": "system", "content": self.system_prompt}]
            await self._run_chunk(self._render(system, False), 1, None)
            self.messages = system

    async def send(self, text: str) -> AsyncIterator[str]:
        """Append a user message and stream the reply's text."""
        sink: asyncio.Queue[str | None] = asyncio.Queue()
        task = asyncio.create_task(self._turn(text, sink))
        while (delta := await sink.get()) is not None:
            yield delta
        await task

    async def _turn(self, text: str, sink: asyncio.Queue[str | None]) -> None:
        try:
            async with self._lock:
                before = self._render(self.messages, False)
                messages = self.messages + [{"role": "user", "content": text}]
                after = self._render(messages, True)
                if not after.startswith(before):
                    raise ValueError("the chat template's rendering is not append-only")
                after_reply = bool(self.messages) and (
                    self.messages[-1]["role"] == "assistant"
                )
                chunk = (self._closing if after_reply else "") + after[len(before) :]
                reply = await self._run_chunk(chunk, self.max_tokens, sink)
                self.messages = messages + [{"role": "assistant", "content": reply}]
        finally:
            sink.put_nowait(None)

    @property
    def started(self) -> bool:
        """Whether the request exists (the first chunk has been sent)."""
        return self._outputs is not None

    def decode(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    async def context(self) -> KVViewInfo:
        return await self.engine.inspect_kv(
            self.request_id, positions=True, token_ids=True
        )

    async def tokens(self) -> list[Token]:
        info = await self.context()
        assert info.positions is not None and info.token_ids is not None
        return [
            Token(slot, position, token_id, self.decode(token_id))
            for slot, (position, token_id) in enumerate(
                zip(info.positions, info.token_ids)
            )
        ]

    async def drop(self, start: int, end: int, close_gap: bool = True) -> KVEditResult:
        """Remove cache slots ``[start, end)``. With ``close_gap`` the later
        slots are re-rotated to sit right after slot ``start - 1``; without it
        they keep their positions and a gap is left."""
        async with self._lock:
            return await self.engine.edit_kv(
                self.request_id, DropOp(start, end, close_gap=close_gap)
            )

    async def close(self) -> None:
        """End the request and free its cache."""
        async with self._lock:
            self._inputs.put_nowait(None)
            if self._outputs is not None:
                # Closing the output stream aborts the request.
                await self._outputs.aclose()
                self._outputs = None
