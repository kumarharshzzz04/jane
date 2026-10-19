"""Capture the README's screenshots from the app that actually ships.

A committed screenshot with no way to regenerate it goes stale in silence, so
the images in `docs/assets/` come from this script rather than from a one-off
paste. It drives headless Edge — present on every Windows box, so nothing gets
downloaded — over the Chrome DevTools Protocol, which is why the only import
beyond the stdlib is `websockets` (already a transitive dep in the venv).

    # bridge first (:8100), then:
    .venv\\Scripts\\python.exe scripts/capture_readme_shots.py

Writes docs/assets/landing.png, curtain.png and dashboard.png.

The dashboard is the only screen here that needs no model call: it is
deterministic pandas, so the shot is fast and comes out the same every time.
An answer shot is available behind `--with-answer`, but it is off by default —
the local 7B currently answers most questions with the verifier's factual
fallback line ("Executed result — ..."), which is honest and a poor advert.
Turn it on once that reads like an analyst.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "src" / "jane" / "sample_data" / "retail_sales.csv"

DEFAULT_URL = "http://127.0.0.1:8100/"
DEFAULT_QUESTION = "Break down revenue by region — who leads, and by how much?"

LANDING_FRAME = (1440, 900)
# The dashboard stacks a stat rail, insight cards, a chart grid and a column
# ledger. Rather than guess a height and crop a panel in half, the frame is
# measured from the page once it renders — see the dashboard step in capture().
DASHBOARD_WIDTH = 1440
DASHBOARD_MAX_HEIGHT = 1700

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for path in BROWSERS:
        if Path(path).exists():
            return path
    for name in ("msedge", "chrome", "chromium"):
        if found := shutil.which(name):
            return found
    sys.exit("no Edge/Chrome found — install one, or capture the shots by hand")


class CDP:
    """A minimal DevTools Protocol client: send a command, await its reply."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self._id = 0

    async def call(self, method: str, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expression: str):
        res = await self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        if res.get("exceptionDetails"):
            raise RuntimeError(res["exceptionDetails"].get("text", "page error"))
        return res.get("result", {}).get("value")

    async def viewport(self, width: int, height: int) -> None:
        await self.call(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=1,
            mobile=False,
        )

    async def shot(self, path: Path) -> None:
        data = (await self.call("Page.captureScreenshot", format="png"))["data"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(data))
        print(f"  -> {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")

    async def click(self, text: str, tag: str = "button") -> bool:
        return bool(
            await self.js(
                f"""(() => {{
                  const el = [...document.querySelectorAll({tag!r})].find(
                    n => n.textContent.trim().toLowerCase().includes({text.lower()!r}));
                  if (!el) return false;
                  el.click();
                  return true;
                }})()"""
            )
        )

    async def until(self, expression: str, what: str, timeout: float = 240.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.js(expression):
                return
            await asyncio.sleep(1)
        raise TimeoutError(f"gave up waiting for {what}")


async def load_sample(cdp: CDP) -> None:
    """Enter the chat and hand the app a real file through its own file input."""
    await cdp.js("window.scrollTo(0, 0)")
    await asyncio.sleep(0.5)
    if not await cdp.click("start asking"):
        sys.exit("could not find the landing's Start asking button")
    await cdp.until("!!document.querySelector('input[type=file]')", "the chat view")

    print(f"uploading {SAMPLE.name}")
    doc = await cdp.call("DOM.getDocument")
    node = await cdp.call(
        "DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="input[type=file]"
    )
    await cdp.call("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(SAMPLE)])
    # the composer only stops asking for a dataset once the bridge holds a table
    await cdp.until(
        "!!document.querySelector('form input[placeholder]')?.placeholder.includes('Ask about')",
        "the table to load",
        timeout=120,
    )
    await asyncio.sleep(2)


async def ask(cdp: CDP, question: str) -> None:
    print(f"asking: {question}")
    typed = await cdp.js(
        f"""(() => {{
          const input = document.querySelector('form input[placeholder]');
          if (!input) return false;
          // React tracks its own value, so go through the native setter and
          // dispatch the event it listens for — assigning .value does nothing.
          const setter = Object.getOwnPropertyDescriptor(
            HTMLInputElement.prototype, 'value').set;
          setter.call(input, {question!r});
          input.dispatchEvent(new Event('input', {{ bubbles: true }}));
          return true;
        }})()"""
    )
    if not typed or not await cdp.click("ask"):
        sys.exit("could not drive the composer")
    # A turn ends when the button stops saying Thinking…; it may land a receipt
    # or an error block, and either one means the screenshot is ready to take.
    await cdp.until(
        """(() => {
             const btn = document.querySelector('form button[type=submit]');
             const thinking = !!btn && /Thinking/.test(btn.textContent);
             return !thinking && !!document.querySelector('.receipt, .r-verified, p.text-ember');
           })()""",
        "the answer",
    )
    await asyncio.sleep(4)  # chart images finish decoding


async def capture(cdp: CDP, url: str, assets: Path, question: str, with_answer: bool) -> None:
    await cdp.call("Page.enable")
    await cdp.call("Runtime.enable")
    await cdp.call("DOM.enable")
    await cdp.viewport(*LANDING_FRAME)

    await cdp.call("Page.navigate", url=url)
    await asyncio.sleep(7)  # webfonts, then the WebGL factory's first frames

    print("landing")
    await cdp.shot(assets / "landing.png")

    print("curtain")
    await cdp.js("window.scrollTo(0, document.documentElement.scrollHeight)")
    await asyncio.sleep(3)  # the scrub needs a frame to settle on its last value
    await cdp.shot(assets / "curtain.png")

    await load_sample(cdp)

    print("dashboard")
    await cdp.viewport(DASHBOARD_WIDTH, 1100)
    if not await cdp.click("dashboard"):
        sys.exit("could not find the dashboard tab")
    await cdp.until(
        "document.querySelectorAll('figure img').length >= 4", "the chart panels", timeout=120
    )
    await asyncio.sleep(3)

    # Grow the frame to the content, so the shot ends where the dashboard does
    # instead of cutting the second row of panels in half.
    height = await cdp.js(
        "Math.min(1700, Math.ceil("
        "(document.querySelector('main')?.scrollHeight || 1100) + 24))"
    )
    await cdp.viewport(DASHBOARD_WIDTH, int(height))
    await asyncio.sleep(2)
    await cdp.js("document.querySelector('main')?.scrollTo(0, 0)")
    await asyncio.sleep(1)
    await cdp.shot(assets / "dashboard.png")

    if with_answer:
        await cdp.viewport(*LANDING_FRAME)
        await cdp.js(
            """[...document.querySelectorAll('button')]
                 .find(b => b.textContent.trim().toLowerCase() === 'chat')?.click()"""
        )
        await asyncio.sleep(1)
        await ask(cdp, question)
        print("answer")
        # Put the whole turn at the top of the scroller so one frame holds it.
        await cdp.js(
            """(() => {
                 const receipt = [...document.querySelectorAll('.receipt')].pop();
                 const turn = receipt?.closest('div.space-y-3') ?? receipt;
                 const scroller = turn?.closest('main');
                 if (!turn || !scroller) return false;
                 scroller.scrollTop += turn.getBoundingClientRect().top
                                     - scroller.getBoundingClientRect().top - 24;
                 return true;
               })()"""
        )
        await asyncio.sleep(1.5)
        await cdp.shot(assets / "answer.png")


async def main() -> None:
    ap = argparse.ArgumentParser(description="Capture the README screenshots.")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets"))
    ap.add_argument("--port", type=int, default=9223)
    ap.add_argument(
        "--with-answer",
        action="store_true",
        help="also ask a question and capture the answer (a real model run)",
    )
    ap.add_argument("--question", default=DEFAULT_QUESTION, help="with --with-answer")
    ap.add_argument("--keep-open", action="store_true", help="leave the browser up to poke at")
    args = ap.parse_args()
    assets = Path(args.out)

    try:
        urllib.request.urlopen(args.url, timeout=5).read(64)
    except Exception as e:  # noqa: BLE001 - any failure means "is it even up?"
        sys.exit(f"{args.url} is not answering ({e}) — start the bridge first")
    print(f"capturing from {args.url}\n")

    profile = tempfile.mkdtemp(prefix="jane-shots-")
    proc = subprocess.Popen(
        [
            find_browser(),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={args.port}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        ws_url = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{args.port}/json/list", timeout=2
                ) as r:
                    for target in json.load(r):
                        if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                            ws_url = target["webSocketDebuggerUrl"]
                            break
            except Exception:  # noqa: BLE001 - the browser is still starting
                pass
            if ws_url:
                break
            time.sleep(0.5)
        if not ws_url:
            sys.exit("the browser never opened a debuggable page")

        async with connect(ws_url, max_size=None) as ws:
            await capture(CDP(ws), args.url, assets, args.question, args.with_answer)
        print(f"\ndone — wrote into {assets}")
    finally:
        if not args.keep_open:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    # The question and the paths in the log can carry characters a cp1252
    # console cannot encode (cli.py reconfigures for the same reason).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
