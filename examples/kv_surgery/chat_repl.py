# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Terminal chat over one live, editable KV cache.

Type a message to chat. Commands:

    /count              tokens in the KV cache right now
    /tokens [START END] list cache slots (all, or [START, END))
    /drop START END     remove slots [START, END); later slots shift down
    /dropgap START END  remove slots [START, END) and leave a position gap
    /quit

Example (on a GPU node)::

    python examples/kv_surgery/chat_repl.py \\
        --model meta-llama/Llama-3.1-8B-Instruct --system "You are terse."
"""

import argparse
import asyncio

from chat_session import ChatSession, make_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--system", default=None, help="system prompt")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


async def show_tokens(session: ChatSession, args: list[str]) -> None:
    tokens = await session.tokens()
    start, end = (int(args[0]), int(args[1])) if args else (0, len(tokens))
    for t in tokens[start:end]:
        gap = "" if t.position == t.slot else f"  (position {t.position:g})"
        print(f"{t.slot:6d}  {t.text!r}{gap}")


async def main() -> None:
    args = parse_args()
    engine = make_engine(
        args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    session = ChatSession.for_engine(
        engine, "chat", system_prompt=args.system, max_tokens=args.max_tokens
    )
    try:
        await session.start()
        if args.system is not None:
            print(f"[system prompt: {(await session.context()).num_slots} tokens]")
        while True:
            try:
                line = (await asyncio.to_thread(input, "> ")).strip()
            except EOFError:
                break
            if not line:
                continue
            command, *rest = line.split()
            try:
                if command == "/quit":
                    break
                elif command.startswith("/") and not session.started:
                    print("[the KV cache is empty: nothing has been sent yet]")
                elif command == "/count":
                    info = await session.context()
                    print(f"[{info.num_slots} tokens in the KV cache]")
                elif command == "/tokens":
                    await show_tokens(session, rest)
                elif command in ("/drop", "/dropgap"):
                    start, end = int(rest[0]), int(rest[1])
                    result = await session.drop(
                        start, end, close_gap=command == "/drop"
                    )
                    print(
                        f"[dropped {end - start} tokens; {result.num_slots} left, "
                        f"next position {result.next_position}]"
                    )
                elif command.startswith("/"):
                    print(__doc__.split("\n\n")[1])
                else:
                    async for delta in session.send(line):
                        print(delta, end="", flush=True)
                    info = await session.context()
                    print(f"\n[{info.num_slots} tokens in the KV cache]")
            except Exception as e:  # a bad span, from the engine core
                print(f"[error: {e}]")
    finally:
        await session.close()
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
