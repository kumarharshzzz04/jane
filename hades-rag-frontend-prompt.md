# HadeS RAG Frontend — Rebuild Prompt

> Everything needed to recreate the HadeS React frontend in a fresh project.
> Stack: Vite + React 19 + TypeScript + Tailwind v4 + shadcn v4 (base-nova) + FastAPI bridge.
> Theme: "obsidian + chrome" — near-black ink surfaces, metallic silver gradient text, emerald strictly for live states.
> Last verified working: 2026-10-01.

---

## 1. Scaffold

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend && npm install
npm install tailwindcss @tailwindcss/vite
# shadcn v4 needs path aliases configured BEFORE init:
#   tsconfig.json + tsconfig.app.json → "paths": { "@/*": ["./src/*"] }
#   vite.config.ts → resolve.alias { "@": path.resolve(__dirname, "./src") }
npx shadcn@latest init -y -d          # style: base-nova, css vars in src/index.css
npx shadcn@latest add button card badge input separator -y

npm install three @react-three/drei @react-three/fiber framer-motion lucide-react thinking-orbs gsap
npm install react-markdown remark-gfm rehype-highlight highlight.js
npm install @fontsource-variable/inter @fontsource-variable/jetbrains-mono @fontsource-variable/plus-jakarta-sans
```

- Tailwind v4 = CSS-first: no tailwind.config; theme goes in `src/index.css` via `@theme inline`.
- Vite proxy for the backend: `server.proxy { "/api": { target: "http://localhost:8000", changeOrigin: true } }`.
- **Apply `class="dark"` on `<html>` in index.html** — the #1 pitfall: shadcn defaults to light and the whole UI renders as white-on-white until this is set.

## 2. Theme tokens (paste into `src/index.css` under `.dark`)

Core idea: **obsidian + chrome**. Backgrounds are near-black slate (`oklch(0.145 0.004 260)` ≈ #0a0c10), text is chrome white. ONE accent family: silver gradients. Emerald (`oklch(0.76 0.13 150)`) ONLY for "live/ready" states. No purple/violet anywhere (reads as AI slop).

```css
.dark {
  --background: oklch(0.145 0.004 260);   /* #0a0c10 */
  --foreground: oklch(0.93 0.004 240);    /* #e9eaec */
  --card: oklch(0.175 0.004 260);
  --primary: oklch(0.75 0.006 240);       /* chrome silver */
  --primary-foreground: oklch(0.14 0.004 260);
  --border: oklch(1 0 0 / 9%);
  --ring: oklch(0.75 0.006 240);
  --hades-live: oklch(0.76 0.13 150);     /* emerald — readiness only */
}
```

Signature CSS utilities (in index.css):
- `.hades-chrome-text` — metallic headline: `linear-gradient(180deg,#fff,#c9ccd3 55%,#8b8f98)` + bg-clip text.
- `.hades-grad-cta` — brushed metal button: `linear-gradient(180deg,#f2f3f5,#c6cad2)` + ink text + subtle border `rgba(255,255,255,.4)`.
- `.hades-live-dot` — 7px emerald dot, glow shadow, 2s pulse.
- `.hades-time` — timestamps: JetBrains Mono + `font-variant-numeric: tabular-nums` (timestamps are the product).
- Body backdrop: `linear-gradient(180deg,#08090c,#0a0c10 45%,#0d0f14)` + two faint silver radial glows, `background-attachment: fixed`.
- `.prose-hades` — markdown answer styles: open layout (no card), 0.925rem/1.7, code chips with `oklch(1 0 0 / .07)` bg, code blocks as **own near-black box** `#07080b` + border + radius 12 + JetBrains Mono, `.is-streaming > :last-child::after` blinking chrome caret `▍`.
- Import highlight.js theme in main.tsx: `import 'highlight.js/styles/github-dark.css'`.
- Fonts: `@fontsource-variable/...` imports (NEVER Google Fonts CDN — offline requirement).

## 3. Layout & components

- **Landing** (`components/hades/Landing.tsx`): full-viewport `DotGlobeHero` (wireframe 3D sphere, silver `#aab0bb`, slow rotation) with chrome headline "Ask your **Lectures**" (two-weight split, underline sweep animation), live-badge pill `LOCAL AI · ZERO CLOUD`, CTAs "Start asking" + "Try 'What are functions?'", example chips, scroll cue. Below the fold: **CinematicFooter** (GSAP ScrollTrigger reveal — giant outlined background wordmark, rotated infinite marquee `LOCAL AI ✦ EXACT TIMESTAMPS ✦ ZERO CLOUD…`, magnetic glass pills, 3-zone grid bottom bar: © left / "Made with ♥ by" center / back-to-top right).
- **Chat** (`App.tsx`): sidebar (17.5rem, obsidian rail) + main. Header: mark + "HadeS" + dataset name + Ready pill. Messages: open-layout answer with avatar mark column, user question as right-aligned quiet `bg-secondary` bubble (NOT gradient), thinking state floats ABOVE the answer (no box): small orb + stage label; answer streams in with markdown; meta row (time + ★ main lesson); sources block; quiz block. Composer: pill input + round metal send button.
- **Dataset switcher** (`Sidebar.tsx`): selectable "tape cards" (radio group, aria-checked) — NEVER a native `<select>` (unstyleable, flashes light-theme). Active card: chrome border + bevel + emerald Live dot + chunks row; each card shows duration + EXACT/ESTIMATED badge.
- **Sources** (`SourceList.tsx`): ★ Main lesson rows (highlighter-yellow border treatment was old; now chrome `border-primary/35 bg-primary/[0.07]`) with YouTube `?t=` seek links; "Also mentioned in N other spots" `<details>` expander with hint "mention the keyword but don't teach it".
- **Thinking states** (`Thinking.tsx` + `thinking-orbs`): map pipeline stages → orb states: boot→working, searching→searching, solving→solving, composing→composing, shaping (quiz)→shaping. Compact after 6s.
- **Avatar** (`HadeSMark.tsx`): custom SVG — hexagon core with speaking waveform bars inside, chrome gradient fill, ink waveform. Used: sidebar brand, message avatar, header, favicon.
- **Transition**: `AnimatePresence mode="wait"` — landing blurs/scales out, chat fades/slides up. Chat crossfades (y:10, 300ms) on dataset switch keyed by profile.

## 4. Data layer

- `lib/types.ts`: `SourceDetail {label,text,start,end,part,youtube_url,score,cluster_size}`, `CuratedResult {main,also}`, `ChatMessage`, `Stage`, `ProfileInfo`.
- `lib/api.ts`: `ask(question, history, profile, {onStage,onToken}) → {answer,curated,response_time}` via **fetch + ReadableStream SSE parsing** (events: `stage`/`token`/`final`/`error`); `generateQuiz(question, context)`; `generateSummary(questions)`; `fetchProfiles()` merges server ready/chunks over static display metadata; `VITE_USE_MOCK=1` env flag falls back to scripted `mock-api.ts` (offline demos without Ollama).
- `lib/mock-answers.ts`: canned OG-voice answers (definition → code block → bulleted mechanics → instructor's framing) for mock mode.

## 5. Backend bridge (one file, reuses existing RAG)

`server.py` at repo root — FastAPI importing the EXISTING `rag_chat.py` untouched:
- `GET /api/health` (ollama ping), `GET /api/profiles` (list_profiles), `POST /api/profiles/{name}` (set_profile).
- `POST /api/ask` → SSE: stage `searching` → run `search_chunks_curated` in thread (emit `solving`) → stage `composing` → `ask_llm_stream` tokens → `final{answer, curated(cleaned, no raw clusters), response_time}`. Prompt = verbatim from ui.py with last-3-turns history injected.
- `POST /api/quiz`, `POST /api/summary` → `ask_llm` with verbatim ui.py prompts.
- CORS for localhost:5173/5174. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `KMP_DUPLICATE_LIB_OK=TRUE` BEFORE importing rag_chat. Run with the project's Anaconda python (`C:/Users/kingh/anaconda3/python.exe server.py`) — NOT system Python 3.14 (missing deps, cp1252 console).

## 6. Pitfalls learned (do not skip)

1. `class="dark"` on `<html>` or everything is white.
2. shadcn v4 init overwrites index.css — paste theme AFTER init; it also changes `-b` flag meaning (base library, not color).
3. three.js cannot parse `hsl(var(--x))` — pass explicit hex color prop to globe material.
4. Native `<select>` breaks dark theme — use custom radio cards.
5. `h-screen` → `h-svh` for mobile browser chrome.
6. thinking-orbs package only accepts preset sizes (20/64) — override canvas via wrapper `[&_canvas]:!size-7`.
7. Windows cp1252 console: avoid unicode arrows in server prints.
8. framer-motion v13 import is `framer-motion` still; oxlint warns on unused vars — keep clean.
9. Frontend speaks shapes identical to rag_chat's `search_chunks_curated` output — never invent response shapes.
10. Offline guarantee: zero CDN/font/http references in src; everything npm-bundled. Only external link = YouTube watch links (click-through only).

## 7. Verify

```bash
npx tsc -b            # clean
npm run lint          # 0 errors
curl localhost:8000/api/health && curl localhost:8000/api/profiles
curl -N -X POST localhost:8000/api/ask -H "Content-Type: application/json" \
  -d '{"question":"what are functions","profile":"hades_demo","history":[]}'
# expect: stage searching → solving → composing → tokens → final with
# main lesson "HadeS Demo — 02:06:45 – 02:10:30 (6 chunks)" (matches EVAL.md)
```
