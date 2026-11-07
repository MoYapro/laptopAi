"""Send one message to Housemate over its WebSocket and print the reply.

Runs inside the app container (see project-scripts/send.sh). Messages are always sent with
source "developer" and a sender name, so Housemate knows a tool is talking, not a person in the room.
"""

import argparse
import asyncio
import json
import sys

import websockets


async def main(args: argparse.Namespace) -> int:
    async with websockets.connect("ws://localhost:8080/ws", max_size=2**24) as ws:
        await ws.send(json.dumps({
            "type": "turn", "text": args.message, "source": "developer", "sender": args.sender,
            "camera": args.camera, "speak": args.speak,
        }))
        reply, timings, started = "", None, False
        async for raw in ws:
            m = json.loads(raw)
            kind = m.get("type")
            if kind == "assistant_start":
                started = True
                print("Housemate: ", end="", flush=True)
            elif kind == "token":
                reply += m["text"]
                print(m["text"], end="", flush=True)
            elif kind == "done":
                timings = m.get("timings")
            elif kind in ("error", "notice"):
                print(f"\n[{kind}] {m['text']}", file=sys.stderr)
                if kind == "error":
                    return 1
            elif kind == "state" and m.get("state") == "idle" and started:
                break
        print()
        if args.timings and timings:
            print(f"[timings] {json.dumps(timings)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("message")
    p.add_argument("--from", dest="sender", default="a developer tool", help="who is sending (e.g. 'Claude Code')")
    p.add_argument("--camera", action="store_true", help="attach the current camera frame")
    p.add_argument("--speak", action="store_true", help="also synthesize the reply as speech on open pages")
    p.add_argument("--timings", action="store_true", help="print timings to stderr")
    sys.exit(asyncio.run(main(p.parse_args())))
