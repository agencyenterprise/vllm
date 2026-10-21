# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Web chat over one live, editable KV cache.

The left pane is the chat; the right pane is the model's KV cache, token by
token, with the token count and a control to drop a span of slots. Click a
token to set the start of the span, shift-click to set its (inclusive) end.

Example (on a GPU node; then open http://localhost:8000 through an SSH
tunnel, e.g. ``ssh -L 8000:localhost:8000 <node>``)::

    python examples/kv_surgery/chat_web.py \\
        --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import asyncio
import itertools
from contextlib import asynccontextmanager

import uvicorn
from chat_session import ChatSession, make_engine
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from vllm.v1.engine.async_llm import AsyncLLM


class SendBody(BaseModel):
    text: str


class DropBody(BaseModel):
    start: int
    end: int
    close_gap: bool = True


class ResetBody(BaseModel):
    system: str | None = None


class App:
    def __init__(self, engine: AsyncLLM, max_tokens: int):
        self.engine = engine
        self.max_tokens = max_tokens
        self.session_ids = (f"chat-{i}" for i in itertools.count())
        self.session: ChatSession | None = None
        self.log: list[str] = []

    async def reset(self, system: str | None) -> None:
        if self.session is not None:
            await self.session.close()
        self.session = ChatSession.for_engine(
            self.engine,
            next(self.session_ids),
            system_prompt=system,
            max_tokens=self.max_tokens,
        )
        self.log = []
        await self.session.start()
        if system is not None:
            await self.note("system prompt")

    async def note(self, event: str) -> None:
        assert self.session is not None
        count = (await self.session.context()).num_slots
        self.log.append(f"{event}: {count} tokens")

    async def state(self) -> dict:
        session = self.session
        assert session is not None
        tokens = []
        count, status, next_position = 0, "empty", 0
        if session.started:
            info = await session.context()
            assert info.token_ids is not None and info.positions is not None
            tokens = [
                [session.decode(t), p] for t, p in zip(info.token_ids, info.positions)
            ]
            count, status, next_position = (
                info.num_slots,
                info.status,
                info.next_position,
            )
        return {
            "count": count,
            "status": status,
            "next_position": next_position,
            "messages": session.messages,
            "tokens": tokens,
            "log": self.log,
        }


def build(engine: AsyncLLM, args: argparse.Namespace) -> FastAPI:
    app_state = App(engine, args.max_tokens)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await app_state.reset(args.system)
        yield
        if app_state.session is not None:
            await app_state.session.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return PAGE

    @app.get("/api/state")
    async def state() -> dict:
        return await app_state.state()

    @app.post("/api/send")
    async def send(body: SendBody) -> StreamingResponse:
        session = app_state.session
        assert session is not None
        turn = sum(m["role"] == "user" for m in session.messages) + 1

        async def stream():
            async for delta in session.send(body.text):
                yield delta
            await app_state.note(f"after turn {turn}")

        return StreamingResponse(stream(), media_type="text/plain")

    @app.post("/api/drop")
    async def drop(body: DropBody) -> dict:
        assert app_state.session is not None
        try:
            await app_state.session.drop(body.start, body.end, body.close_gap)
        except Exception as e:
            # The engine core reports a bad span as a plain Exception.
            raise HTTPException(400, str(e)) from e
        gap = "" if body.close_gap else ", gap kept"
        await app_state.note(f"drop [{body.start}, {body.end}){gap}")
        return await app_state.state()

    @app.post("/api/reset")
    async def reset(body: ResetBody) -> dict:
        await app_state.reset(body.system or None)
        return await app_state.state()

    return app


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>KV surgery chat</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#fafafa; --fg:#1d1d1f; --muted:#6e6e73; --line:#d9d9de;
  --panel:#fff; --tok:#eef1f6; --tok2:#e4e8f0; --special:#e6dcf5;
  --sel:#ffd6d6; --gap:#d0342c; --user:#e8f0fe; }
@media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#ececf0;
  --muted:#9a9aa2; --line:#33333a; --panel:#1f1f23; --tok:#2a2d35;
  --tok2:#31353f; --special:#3d3150; --sel:#5c2626; --user:#23324d; } }
* { box-sizing: border-box; }
body { margin:0; font:14px/1.45 system-ui, sans-serif; background:var(--bg);
  color:var(--fg); }
main { display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1.3fr);
  gap:12px; padding:12px; height:100vh; }
@media (max-width: 800px) { main { grid-template-columns: 1fr; height:auto; } }
section { background:var(--panel); border:1px solid var(--line);
  border-radius:8px; display:flex; flex-direction:column; min-height:0; }
h2 { font-size:13px; margin:0; padding:10px 12px; border-bottom:1px solid
  var(--line); display:flex; gap:8px; align-items:baseline; }
h2 .big { font-size:20px; } .muted { color:var(--muted); font-weight:normal; }
#chat, #tokens { flex:1; overflow:auto; padding:12px; }
.msg { white-space:pre-wrap; margin:0 0 10px; padding:8px 10px;
  border-radius:6px; }
.msg.user { background:var(--user); } .msg.system { color:var(--muted);
  font-style:italic; }
form, .controls { display:flex; gap:6px; padding:10px 12px;
  border-top:1px solid var(--line); flex-wrap:wrap; align-items:center; }
textarea { flex:1; min-width:0; resize:vertical; font:inherit; padding:6px;
  background:var(--bg); color:var(--fg); border:1px solid var(--line);
  border-radius:6px; }
input[type=number] { width:80px; font:inherit; padding:4px; background:var(--bg);
  color:var(--fg); border:1px solid var(--line); border-radius:6px; }
button { font:inherit; padding:5px 12px; border-radius:6px; cursor:pointer;
  border:1px solid var(--line); background:var(--tok); color:var(--fg); }
button:disabled { opacity:.5; cursor:default; }
#tokens { font:12.5px/1.9 ui-monospace, monospace; white-space:pre-wrap;
  word-break:break-all; }
.t { background:var(--tok); border-radius:3px; cursor:pointer; }
.t:nth-child(even) { background:var(--tok2); }
.t.special { background:var(--special); }
.t.sel { background:var(--sel); text-decoration:line-through; }
.gapmark { color:var(--gap); font-weight:bold; }
#log { padding:6px 12px; border-top:1px solid var(--line); color:var(--muted);
  font-size:12px; max-height:110px; overflow:auto; }
#err { color:var(--gap); }
</style></head><body><main>
<section>
  <h2>Chat <span class="muted">transcript: what was said; edits do not
    change it</span></h2>
  <div id="chat"></div>
  <form id="send"><textarea id="text" rows="2"
    placeholder="Message (Enter to send, Shift+Enter for a newline)"></textarea>
    <button id="sendbtn">Send</button></form>
  <form id="reset"><textarea id="system" rows="1"
    placeholder="System prompt (optional)"></textarea>
    <button>New chat</button></form>
</section>
<section>
  <h2>KV cache <span class="big" id="count">0</span> tokens
    <span class="muted" id="status"></span></h2>
  <div id="tokens"></div>
  <div class="controls">
    drop slots from <input type="number" id="start" min="0">
    to <input type="number" id="end" min="0"> (exclusive)
    <label><input type="checkbox" id="close" checked> shift later tokens
      down to fill the gap</label>
    <button id="dropbtn">Drop</button> <span id="err"></span>
  </div>
  <div id="log"></div>
</section>
</main><script>
const $ = (id) => document.getElementById(id);
let state = null, busy = false, anchor = null;

function setBusy(b) { busy = b; $("sendbtn").disabled = b;
  $("dropbtn").disabled = b; }

function render() {
  const chat = $("chat"); chat.innerHTML = "";
  for (const m of state.messages) {
    const d = document.createElement("div");
    d.className = "msg " + m.role; d.textContent = m.content;
    chat.appendChild(d);
  }
  chat.scrollTop = chat.scrollHeight;
  $("count").textContent = state.count;
  $("status").textContent = state.status.toLowerCase() +
    " · next position " + state.next_position;
  const box = $("tokens"); box.innerHTML = "";
  const lo = +$("start").value, hi = +$("end").value;
  const hasSel = $("start").value !== "" && $("end").value !== "";
  state.tokens.forEach(([text, pos], slot) => {
    const prev = slot ? state.tokens[slot - 1][1] : -1;
    if (pos !== prev + 1) {
      const g = document.createElement("span"); g.className = "gapmark";
      g.textContent = "⌇"; g.title = "position jumps " + prev + " → " + pos;
      box.appendChild(g);
    }
    const s = document.createElement("span");
    s.className = "t" + (/^<\\|.*\\|>$|^<[a-z_]+>$/.test(text) ? " special" : "") +
      (hasSel && slot >= lo && slot < hi ? " sel" : "");
    s.textContent = text; s.title = "slot " + slot + " · position " + pos;
    s.dataset.slot = slot; box.appendChild(s);
  });
  $("log").innerHTML = state.log.map((l) => "<div></div>").join("");
  [...$("log").children].forEach((d, i) => d.textContent = state.log[i]);
}

async function refresh() { state = await (await fetch("/api/state")).json();
  render(); }

$("tokens").addEventListener("click", (e) => {
  const slot = e.target.dataset && e.target.dataset.slot;
  if (slot === undefined) return;
  const i = +slot;
  if (e.shiftKey && anchor !== null) {
    $("start").value = Math.min(anchor, i); $("end").value = Math.max(anchor, i) + 1;
  } else { anchor = i; $("start").value = i; $("end").value = i + 1; }
  render();
});
for (const id of ["start", "end"]) $(id).addEventListener("input", render);

$("dropbtn").addEventListener("click", async () => {
  $("err").textContent = "";
  setBusy(true);
  const r = await fetch("/api/drop", { method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ start: +$("start").value, end: +$("end").value,
      close_gap: $("close").checked }) });
  setBusy(false);
  if (!r.ok) { $("err").textContent = (await r.json()).detail; return; }
  $("start").value = ""; $("end").value = ""; anchor = null;
  state = await r.json(); render();
});

$("send").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("text").value; if (!text.trim() || busy) return;
  $("text").value = ""; setBusy(true);
  state.messages.push({ role: "user", content: text },
    { role: "assistant", content: "" });
  render();
  const reply = $("chat").lastChild;
  const r = await fetch("/api/send", { method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }) });
  const reader = r.body.getReader(), dec = new TextDecoder();
  for (;;) { const { done, value } = await reader.read(); if (done) break;
    reply.textContent += dec.decode(value, { stream: true });
    $("chat").scrollTop = $("chat").scrollHeight; }
  setBusy(false); await refresh();
});
$("text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault();
    $("send").requestSubmit(); } });

$("reset").addEventListener("submit", async (e) => {
  e.preventDefault(); setBusy(true);
  const r = await fetch("/api/reset", { method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ system: $("system").value }) });
  setBusy(false); state = await r.json(); render();
});

refresh();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--system", default=None, help="initial system prompt")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    async def serve() -> None:
        engine = make_engine(
            args.model,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        try:
            config = uvicorn.Config(build(engine, args), host=args.host, port=args.port)
            await uvicorn.Server(config).serve()
        finally:
            engine.shutdown()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
