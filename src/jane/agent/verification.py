"""Deterministic answer verification guards (no model calls here)."""
import json
import re

from ..types import GuardOutcome



def _extract_numbers(text: str) -> list[float]:
    """All numbers in a text, tolerant of commas and trailing %."""
    out: list[float] = []
    for m in re.finditer(r"[-+]?\d[\d,]*\.?\d*", text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def _result_numbers(res) -> set[float]:
    """Numbers that legitimately exist in an executed result (rounded for float noise)."""
    nums: set[float] = set()
    def _add(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            nums.add(round(float(v), 4))
            nums.add(round(float(v), 1))
            nums.add(float(int(v)) if abs(v - int(v)) < 1e-9 else round(float(v), 2))
    for t in getattr(res, "tables", []) or []:
        for row in t.get("rows", [])[:50]:
            vals = row.values() if isinstance(row, dict) else (row or [])
            for v in vals:
                _add(v)
                if isinstance(v, str):
                    for n in _extract_numbers(v):
                        _add(n)
    return nums


def _derivable(n: float, truth: list[float], allow_percent: bool = False) -> bool:
    """True when n is a difference or sum of two result numbers — or, when the
    answer presents it as a percentage, a percentage-difference of two result
    numbers. Deliberately NOT a free a/b*100 search: that accepts meaningless
    coincidences (38.5/308*100 = 12.5), and on a large raw result the
    percentage branch alone is a lottery — 200 prices always contain a pair
    whose ratio happens to equal any small number (live: a hallucinated
    r = 0.73 was blessed as "0.65% of a pair"). Hence allow_percent."""
    vals = [t for t in truth if t != 0][:24]
    for i, a in enumerate(vals):
        for j, b in enumerate(vals):
            if i == j:
                # a paired with ITSELF makes every even number "derivable"
                # (a + a = 12 from the ram=6 winner), which blessed 'No — 12 GB
                # has the highest average price at 40900.16' while the winning
                # row was ram = 6.
                continue
            cands = [(a - b), (a + b)]
            if allow_percent and b:
                cands.append((a - b) / b * 100)
            for cand in cands:
                # 0.11 tolerates truncation to one decimal place (live: the
                # repair wrote "1.4%" for south's true 1.4786% lead and the
                # guard threw the correct sentence away)
                if abs(cand) < 1e12 and abs(n - cand) <= max(0.11, abs(cand) * 0.01):
                    return True
    return False


def _find_cell(value: float, res, tol: float | None = None) -> bool:
    """True when value matches some cell in the executed result within the
    truth guard's tolerance scale."""
    tol = tol if tol is not None else lambda t: max(0.051, abs(t) * 0.002)
    return any(abs(value - t) <= tol(t) for t in sorted(_result_numbers(res)))


def _percent_claim(claim: dict) -> bool:
    """True when a claim is *about* a percentage (metric/unit/operation/group
    naming percent) — decides which failure message and rule applies."""
    blob = " ".join(str(claim.get(k, "")) for k in
                    ("metric", "unit", "operation", "group")).lower()
    return any(t in blob for t in ("percent", "pct", "%"))


def _pct_pair_match(stated_percent: float, res) -> bool:
    """True when `stated_percent` equals (a - b) / b * 100 for two numbers in
    the executed result. A pct_change stated without the nested `comparison`
    dict (a 7B wrote {"operation": "pct_change", "value": 1.48}) is verified
    against the result's own group pairs, not a raw cell."""
    vals: list[float] = []
    for t in getattr(res, "tables", []) or []:
        for row in t.get("rows", [])[:24]:
            cells = row.values() if isinstance(row, dict) else (row or [])
            for v in cells:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append(float(v))
    vals = vals[:24]
    for a in vals:
        for b in vals:
            if b == 0 or a == b:
                continue
            cand = (a - b) / b * 100.0
            if abs(stated_percent - cand) <= max(0.11, abs(cand) * 0.002):
                return True
    return False


def _claim_unit_value(claim: dict) -> float | None:
    """Normalize a claim's value by unit: percent -> /100, fraction -> as-is.
    Returns None for non-numeric values (the prose guards own text)."""
    v = claim.get("value")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    unit = str(claim.get("unit", "number")).lower()
    if unit in ("percent", "%", "pct"):
        return v / 100.0
    if unit == "fraction":
        return v
    return v          # "number" / unknown unit: verify as-is


def _claims_fail(claim: dict, res, q_val: float | None, failures: list[str]) -> None:
    """Check one claim against the result; append a named failure or nothing.
    Numbers compare in NORMALIZED space (fraction scale) with the truth
    guard's relative tolerance; percent-space display values are matched with
    the same tolerance scaled ×100 (display form of the same check)."""
    label = str(claim.get("metric") or "claim")
    group = claim.get("group")
    if group:
        label = f"{label} / {group}"
    nv = _claim_unit_value(claim)
    if nv is None:
        return                      # non-numeric: skipped, prose guards own it
    # percent-unit display value (38.5) must exist in display space when the
    # result stores display numbers, OR fraction space when it stores 0.385:
    display = nv * 100.0
    if not (_find_cell(nv, res) or _find_cell(display, res)):
        # a pct_change may arrive WITHOUT the nested comparator dict — accept
        # it when it equals SOME group-pair percentage in the result (the
        # comparator's identity is prose's job, not the guard's).
        raw = claim.get("value")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) \
                and _percent_claim(claim):
            unit = str(claim.get("unit", "number")).lower()
            stated_pct = float(raw) * 100.0 if unit == "fraction" else float(raw)
            if -1000.0 <= stated_pct <= 1000.0 and _pct_pair_match(stated_pct, res):
                return
        if _percent_claim(claim):
            failures.append(
                f"{label} claims {raw} but no percentage in the result "
                "matches — a percentage change must equal (a - b) / b * 100 "
                "for two values in the result")
        else:
            failures.append(
                f"{label} claims {claim.get('value')} but that number does "
                "not appear (in any unit) in the executed result")
        return
    comp = claim.get("comparison")
    if isinstance(comp, dict):
        cval = comp.get("value")
        if isinstance(cval, bool) or not isinstance(cval, (int, float)):
            failures.append(f"{label}: comparison value is not numeric")
            return
        unit = str(claim.get("unit", "number")).lower()
        c_norm = float(cval) / 100.0 if unit in ("percent", "%", "pct") else float(cval)
        if not (_find_cell(c_norm, res) or _find_cell(c_norm * 100.0, res)):
            failures.append(
                f"{label}: comparison group {comp.get('group')!r} claims "
                f"{cval} but that number does not appear in the result")
            return
        op = str(comp.get("operation") or "difference").lower()
        a, b = nv, c_norm
        if op == "difference":
            holds = True                                    # a - b is real arithmetic
        elif op == "pct_change":
            holds = b != 0 and abs((a - b) / b * 100 - (claim.get("value") if False else 0)) < 1e18
            holds = True                                    # checked below via derived claim value
        elif op == "ratio":
            holds = b != 0
        else:
            holds = True
        # the CLAIM must be consistent with its own comparison: for pct_change
        # the claim's value (in percent space) should equal (a-b)/b*100
        if op == "pct_change" and b != 0:
            stated = claim.get("value")
            if isinstance(stated, (int, float)) and not isinstance(stated, bool):
                expected = (a - b) / b * 100.0
                if abs(float(stated) - expected) > max(0.11, abs(expected) * 0.002):
                    failures.append(
                        f"{label}: claims {stated} but (result_a - result_b)/result_b*100 "
                        f"is {expected:.4g}")
        if not holds:
            failures.append(f"{label}: comparison arithmetic impossible")


def verify_claims(claims, res) -> tuple[bool, list[str]]:
    """Deterministically verify the model's structured claims against the
    executed result. Returns (ok, failures) — each failure names the claim so
    the retry hint can target it. No model calls (spec §2)."""
    if not claims:
        return True, []
    failures: list[str] = []
    for claim in claims:
        if isinstance(claim, dict):
            _claims_fail(claim, res, None, failures)
    return (not failures), failures


def _answer_matches_data(answer: str, res, question: str = "") -> bool:
    """True when every load-bearing number in the answer is explained by the
    executed result: it either appears in the table, is echoed from the
    question (a threshold like 'above 8'), is a small prose integer ('top 5',
    '1 in 8'), or is a simple derivation (share/percentage/diff) of table
    values. A LONE checked number facing a single-value result must match it
    — that catches 'there are 3 movies' when the true count is 32."""
    ans_nums = _extract_numbers(answer or "")
    if not ans_nums:
        return True
    truth_set = _result_numbers(res)
    if not truth_set:
        return True
    truth = sorted(truth_set)
    q_nums = _extract_numbers(question or "")
    checked = [n for n in ans_nums
               if not any(abs(n - q) <= max(0.051, abs(q) * 0.002) for q in q_nums)]
    if not checked:
        return True                       # pure echo of the question's numbers
    single_value_result = len(truth) <= 4
    # a percentage derivation is only meaningful for a number the answer
    # actually presents as one — otherwise any small coefficient/score can be
    # "derived" from a ratio of two raw cells it has nothing to do with.
    allow_percent = ("%" in (answer or "")) or _rate_question(question)
    for n in checked:
        if any(abs(n - t) <= max(0.051, abs(t) * 0.002) for t in truth):
            continue                      # appears verbatim in the result
        if _is_prose_int(n, answer or "") and (len(checked) > 1 or not single_value_result):
            continue                      # prose numbers: 'top 5', '3 of 4', '1 in 8'
        if _derivable(n, truth, allow_percent=allow_percent):
            continue                      # e.g. '12.5% more' from 54.21 vs 48.15
        # '38.5%' narrating a 0.385 rate stored in the result — same number,
        # percent-formatted (value*100) because the user asked for a percentage
        if 0 < n <= 100 and any(abs(n / 100 - t) <= max(0.00051, abs(t) * 0.002)
                                for t in truth):
            continue
        # percent-looking number with no source in a rate-style question:
        # almost always the model narrating COUNT(*)*1.0/COUNT(*) = 1.0*100
        if 0 < n <= 100 and _rate_question(question):
            return False
        return False                      # load-bearing number with no source
    return True


def _rate_question(question: str) -> bool:
    q = (question or "").lower()
    return any(w in q for w in ("rate", "percentage", "percent", "share",
                                "%", "fraction", "proportion"))


def _is_prose_int(n: float, answer: str) -> bool:
    """True for a small integer used as a count phrase — 'top 5', '3 of 4',
    '1 in 8' — never for a bare small number that could be a group label.
    Live defect: 'No — 12 GB has the highest average price at 40900.16' passed
    the truth guard because 12 was a small integer and the winner's value was
    right, although the result's winning row was ram = 6."""
    if not float(n).is_integer() or abs(n) >= 20:
        return False
    i = int(n)
    a = answer or ""
    return bool(re.search(
        rf"\b(?:top|first|second|third|fourth|last|next|bottom|best|worst|"
        rf"only|both|of|in|out)\s*[- ]?{i}\b", a, re.I)
        or re.search(rf"\b{i}\s+(?:of|in|out\s+of)\b", a, re.I))


# an angle-bracket token from the prompt's own HOW-TO-ANSWER examples that the
# model copied instead of a real label/value. '<1H OCEAN' is safe: the token
# must start with a letter or underscore and close immediately.
_PLACEHOLDER_RE = re.compile(r"<[A-Za-z_][A-Za-z_0-9]*>")


def _has_prompt_placeholders(text: str) -> bool:
    """True when the answer shipped the prompt's example slots verbatim
    (live: 'VERDICT — <ocean_proximity> has the highest average house value
    at <avg_house_value>' reached the user and both guards passed it because
    it contains no numbers to trace)."""
    if not isinstance(text, str):
        text = str(text or "")          # a nested dict answer must not crash
    return bool(_PLACEHOLDER_RE.search(text))


_POLAR_Q = re.compile(r"^\s*(is|are|do|does|did|can|could|will|would|should|"
                      r"was|were|has|have|had|any)\b", re.I)
# the fixed answer slots, wherever they sit (the 7B writes them inline as
# often as on separate lines, so a line-anchored pattern misses half of them)
_SLOT_SPLIT = re.compile(r"(?i)\b(VERDICT|NUMBERS|THE CATCH|NEXT)\b\s*[—:-]\s*")
# an empty slot, or a CATCH that says nothing real. The CATCH must flag an
# actual limitation — a significance claim or filler is noise at best (live:
# 'THE CATCH: No significant difference' sat under a verdict whose two groups
# differ 2.1x, and 'The difference is relatively small, but significant'
# claims a test Jane never ran).
_EMPTY_SLOT_TEXT = re.compile(
    r"(?i)^\s*(none|n/?a|not applicable|null|no catch|no specific[^.]*|"
    r"no particular[^.]*|no significant[^.]*|nothing to (?:flag|report)|"
    r"nothing significant|no issues?|no limitations?)\s*\.?\s*$")


def _tidy_answer(answer: str, question: str) -> str:
    """Deterministic presentation cleanup on the shipped answer:
    - drop an empty slot ('NEXT — None.', 'THE CATCH — None.') instead of
      printing 'None' at the user;
    - drop a CATCH that claims significance (Jane runs no significance test)
      or is filler ('No specific numbers to flag.');
    - normalize slot labels to `LABEL — text`, inline or on their own line;
    - drop a leading 'No — ' / 'Yes — ' on a question that is not a yes/no
      question ('Which RAM size has the highest average price?' shipped as
      'No — 6 GB has the highest average price at 40900.2').
    Never touches the numbers."""
    if not isinstance(answer, str):
        answer = str(answer or "")
    a = answer.strip()
    if not a:
        return answer
    parts = _SLOT_SPLIT.split(a)
    if len(parts) >= 3:
        rebuilt: list[str] = [parts[0].strip()]
        for i in range(1, len(parts), 2):
            label = parts[i].strip()
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            drop = not text or bool(_EMPTY_SLOT_TEXT.match(text))
            if label.upper() == "THE CATCH" and (drop or "significan" in text.lower()):
                continue
            if label.upper() == "NEXT" and drop:
                continue
            rebuilt.append(f"{label} — {text}" if text else label)
        a = "\n\n".join(p for p in rebuilt if p).strip()
    a = re.sub(r"\n{3,}", "\n\n", a).strip()
    if not _POLAR_Q.match(question or ""):
        a = re.sub(r"^\s*(yes|no)\s*[—–-]\s*", "", a, count=1, flags=re.I)
    return a


_REFUSAL_PHRASES = (
    "cannot settle", "can't settle", "cannot be determined", "can't determine",
    "unable to determine", "unable to answer", "cannot answer", "can't answer",
    "not enough data", "no data to answer", "does not contain enough",
    "claims do not match", "claim does not match",
    "does not match the executed result", "doesn't match the executed result",
)


def _label_metric_cols(columns, rows):
    """(label_idx, metric_idx) for a grouped result — >=2 rows with one
    mostly-string column and one mostly-numeric column, else (None, None)."""
    cols = list(columns or [])
    rows = list(rows or [])
    if len(rows) < 2 or len(cols) < 2:
        return None, None

    def _cell(r, i):
        return (r[i] if isinstance(r, (list, tuple))
                else (r.get(cols[i]) if isinstance(r, dict) else None))

    label_i = metric_i = None
    for i in range(len(cols)):
        vals = [_cell(r, i) for r in rows[:25]]
        n_str = sum(isinstance(v, str) for v in vals)
        n_num = sum(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in vals)
        if n_str >= len(vals) * 0.8 and label_i is None:
            label_i = i
        elif n_num >= len(vals) * 0.8 and metric_i is None:
            metric_i = i
    return label_i, metric_i


def _small_numeric_label_metric(columns, rows, max_rows: int | None = 12):
    """(label_idx, metric_idx) for a grouped result whose group labels are
    numbers ('sims' 1/2, 'ram' 2/4/6/8): among all-numeric columns, the
    largest-magnitude one is the measure and the smaller one the label.
    max_rows caps the heuristic at aggregated sizes — a raw listing does not
    rank (pass None to allow any size, for guards that only need the extremes)."""
    cols = list(columns or [])
    rows = list(rows or [])
    if not (2 <= len(rows) and len(cols) >= 2
            and (max_rows is None or len(rows) <= max_rows)):
        return None, None

    def _cell(r, i):
        return (r[i] if isinstance(r, (list, tuple))
                else (r.get(cols[i]) if isinstance(r, dict) else None))

    nums = []
    for i in range(len(cols)):
        vals = [float(_cell(r, i)) for r in rows
                if isinstance(_cell(r, i), (int, float))
                and not isinstance(_cell(r, i), bool)]
        if len(vals) == len(rows):
            nums.append((max(abs(v) for v in vals), i))
    if len(nums) >= 2:
        nums.sort()
        return nums[0][1], nums[-1][1]
    return None, None


def _answerable_refusal(answer: str, columns, rows, question: str) -> bool:
    """True when the answer refuses/meta-comments although the executed result
    IS the grouped answer its question asked for (break-down / who-leads with
    >=2 groups): 'No — the data cannot settle it. … the specific numbers in
    the claims do not match the executed result' shipped live on a perfect
    200-row camera breakdown. The repair call shows the model the rows; a
    refusal it repeats anyway (genuinely missing column) still ships."""
    if not isinstance(answer, str):
        answer = str(answer or "")
    a = answer.lower()
    if not any(p in a for p in _REFUSAL_PHRASES):
        return False
    q = question or ""
    if not (_BREAKDOWN_Q.search(q)
            or (_SUPERLATIVE_MAX + _SUPERLATIVE_MIN)
            and re.search(r"\b(which|who|what)\b", q, re.I)):
        return False
    label_i, metric_i = _label_metric_cols(columns, rows)
    if label_i is None:
        label_i, metric_i = _small_numeric_label_metric(columns, rows)
    return label_i is not None and metric_i is not None


def _looks_like_json_answer(text) -> bool:
    """True when the 'explanation' is a serialized JSON object rather than
    prose (live through the bridge: the 7B returned
    {"verdict": ..., "numbers": [6], "next": ...} and the chat showed raw
    JSON). Such a text carries traceable numbers but reads as machine output,
    so it gets one rewrite call and a deterministic flatten as the backstop."""
    if not isinstance(text, str):
        return True
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return False
    try:
        return isinstance(json.loads(t), dict)
    except (json.JSONDecodeError, ValueError):
        return False


_SKIP_JSON_KEYS = ("code", "language", "plan", "claims", "chart", "charts",
                   "sql", "python")


def _flatten_json_answer(text) -> str:
    """Deterministic prose from a nested JSON answer: the string values, in
    key order, joined into sentences. Numbers are dropped — a bare
    "numbers": [6] rendered as text reads as a claim without context, and the
    verdict sentence already carries the real value. Returns "" when there is
    nothing to say."""
    if not isinstance(text, str):
        return ""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(obj, dict):
        return ""
    parts: list[str] = []

    def _walk(node) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).lower() in _SKIP_JSON_KEYS:
                    continue
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)
        elif isinstance(node, str):
            s = node.strip()
            if s:
                parts.append(s if s.endswith((".", "!", "?")) else s + ".")

    _walk(obj)
    out = " ".join(parts).strip()
    return out[:600]


def _missing_label_column(columns, rows, code) -> bool:
    """True when a grouped/comparison query returned ONE numeric column over
    >=2 rows — the group key never made it into the SELECT, so the rows do not
    say which value belongs to which group (live: the SIM comparison ran
    SELECT AVG(price) … WHERE sims = 2 UNION ALL SELECT AVG(price) … and the
    answer shipped as 'Executed result — avg_price=39,806.38 | avg_price=38,602.40').
    GROUP BY / UNION in the code is the tell — a plain projection that happens
    to return two values is left alone."""
    cols = list(columns or [])
    rows = list(rows or [])
    if len(cols) != 1 or len(rows) < 2:
        return False
    c = str(code or "").upper()
    if "GROUP BY" not in c and "UNION" not in c:
        return False
    vals = []
    for r in rows[:25]:
        v = (r[0] if isinstance(r, (list, tuple))
             else (r.get(cols[0]) if isinstance(r, dict) else None))
        vals.append(v)
    return all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in vals)


def _association_question(question: str) -> bool:
    q = (question or "").lower()
    return any(w in q for w in ("correlat", "relationship", "link",
                                "associated", "association"))


def _association_missing_coefficient(question: str, columns, rows) -> bool:
    """True when an association question came back as a raw listing of the
    two columns with no coefficient computed anywhere — a listing cannot tell
    you the STRENGTH of a link, and the 7B then invents one (live: a scatter
    of ram/price produced 'r = 0.73', copied from the prompt's example, on
    data whose true coefficient is -0.024). A named coefficient column (r,
    corr, correlation, coefficient, pearson) or a single-number result passes."""
    if not _association_question(question):
        return False
    cols = list(columns or [])
    rows = list(rows or [])
    if not cols or not rows:
        return False
    for c in cols:
        if re.search(r"\b(r|corr|correlation|coefficient|pearson)\b",
                     str(c).lower()):
            return False                   # a coefficient is present — no retry
    numeric = []
    for i in range(len(cols)):
        vals = [(r[i] if isinstance(r, (list, tuple))
                 else (r.get(cols[i]) if isinstance(r, dict) else None))
                for r in rows[:25]]
        if sum(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in vals) >= len(vals) * 0.8:
            numeric.append(i)
    if len(rows) == 1 and len(numeric) == 1:
        return False                       # a single coefficient as the value
    label_i, _ = _label_metric_cols(cols, rows)
    if label_i is not None:
        return False                       # grouped result with a category column
    return len(numeric) >= 2 and len(rows) >= 10   # raw listing of two measures


def _fmt_num(v) -> str:
    """Human number for the model-free answer: 38602.39801980198 -> 38,602.40,
    -0.02410143750581424 -> -0.0241, 500 -> 500 (never raw float noise)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    f = float(v)
    if f.is_integer():
        return f"{int(f):,}"
    if abs(f) >= 1:
        return f"{f:,.2f}"
    return f"{f:.4f}".rstrip("0").rstrip(".") or "0"


def _factual_answer(columns: list[str], rows: list, question: str = "") -> str:
    """Model-free answer built from the executed result itself.

    Accepts dict rows (sandbox) or positional rows (SQL). This line replaces
    prose the guards could not bless, so it must be both beyond suspicion and
    readable: for a ranked/grouped result it writes the actual finding (who
    leads, by how much, over how many groups) instead of a raw cell dump —
    live, a correct two-group price comparison shipped as 'Executed result —
    sims=1, avg_price=38602.39801980198 | sims=2, …', which reads like a
    machine and buries the answer. Falls back to the labelled cell line for
    shapes with nothing to rank.
    """
    if not rows:
        return "The query returned no rows."
    cols = list(columns or [])

    def _cell(r, i):
        return (r[i] if isinstance(r, (list, tuple))
                else (r.get(cols[i]) if isinstance(r, dict) and i < len(cols) else None))

    def _fmt_cell(v) -> str:
        return _fmt_num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    # single row, single value: a plain sentence
    if len(rows) == 1 and len(cols) == 1:
        return f"The {str(cols[0]).replace('_', ' ')} is {_fmt_cell(_cell(rows[0], 0))}."

    # ranked result: one label column + one metric column
    label_i, metric_i = _label_metric_cols(cols, rows)
    if label_i is None:
        label_i, metric_i = _small_numeric_label_metric(cols, rows)
    if label_i is not None and metric_i is not None:
        try:
            ranked = sorted(rows[:25],
                            key=lambda r: float(_cell(r, metric_i)),
                            reverse=True)
        except (TypeError, ValueError):
            ranked = []
        if len(ranked) >= 2:
            def _lbl(r) -> str:
                v = _cell(r, label_i)
                # a numeric group label reads as a bare number otherwise
                # ('2 is higher' — 2 what?)
                return (f"{cols[label_i]} {_fmt_cell(v)}" if not isinstance(v, str)
                        else str(v))
            top_lbl, top_v = _lbl(ranked[0]), float(_cell(ranked[0], metric_i))
            second_lbl = _lbl(ranked[1])
            second_v = float(_cell(ranked[1], metric_i))
            if len(ranked) == 2:
                if second_v:
                    pct = (top_v - second_v) / abs(second_v) * 100.0
                    return (f"{top_lbl} is higher: {_fmt_num(top_v)} vs "
                            f"{_fmt_num(second_v)} ({pct:+.1f}%).")
                return f"{top_lbl} is higher: {_fmt_num(top_v)} vs {_fmt_num(second_v)}."
            tail = ""
            if second_v:
                pct = (top_v - second_v) / abs(second_v) * 100.0
                tail = f"; {second_lbl} is second at {_fmt_num(second_v)} ({pct:+.1f}%)"
            return (f"{top_lbl} leads at {_fmt_num(top_v)}{tail} "
                    f"({len(ranked)} groups in the result).")

    parts: list[str] = []
    for r in rows[:4]:
        if isinstance(r, dict):
            pairs = [f"{k}={_fmt_cell(v)}" for k, v in r.items()
                     if v is not None and v != "" and v != []]
        else:
            pairs = [f"{c}={_fmt_cell(v)}" for c, v in zip(cols, list(r))
                     if v is not None and v != "" and v != []]
        if pairs:
            parts.append(", ".join(pairs)[:300])
    if not parts:
        return "The query returned only empty values."
    line = "Executed result — " + " | ".join(parts)
    if len(rows) > 4:
        line += f" | … (+{len(rows) - 4} more rows)"
    return line + "."


def _pct_retry_hint(columns, rows) -> str | None:
    """Deterministic example for the claims retry: the two largest values in
    the result and the percentage change between them, so the model can copy
    the arithmetic instead of repeating a stale number (live: a carried-over
    12.3% burned every attempt on the region question)."""
    vals: list[float] = []
    for r in rows[:10]:
        cells = r.values() if isinstance(r, dict) else (r or [])
        for v in cells:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
    if len(vals) < 2:
        return None
    top, second = sorted(vals, reverse=True)[:2]
    if second == 0:
        return None
    pct = (top - second) / second * 100.0
    return (f"Example from THIS result: ({top:,.2f} - {second:,.2f}) / "
            f"{second:,.2f} * 100 = {pct:.4g}%. A percentage change must be "
            "computed exactly like that — never reuse a number from an "
            "earlier attempt.")


_BREAKDOWN_Q = re.compile(r"\b(break\s*down|breakdown|who\s+leads)\b", re.I)
# top-N / filter / time markers: a partial result can be the CORRECT answer,
# so the coverage guard stays out then (a miss is cheaper than bouncing a
# right answer — this is a net, not a proof).
_COVERAGE_EXEMPT = re.compile(
    r"\b(top|bottom|first|last|worst|best|highest|lowest|largest|smallest|"
    r"most|least|only|among|where|excluding|including|except|within|during|"
    r"since|until|before|after|for|with)\b|\b(19|20)\d{2}\b", re.I)


def _code_tables(code: str, names) -> tuple[list[str], list[str]]:
    """(strong, weak) table names the executed code actually names.

    strong = named in SQL position, right after FROM / JOIN / USING.
    weak   = the name appears as a word anywhere. The python path injects every
             table as a variable, so `kiosks.groupby(...)` has no FROM to match.
    Both forms of a name are tried: the schema name and the identifier the
    sandbox derives from it (`bank-additional-full` -> `bank_additional_full`).

    The two ranks exist because a bare word can be ordinary prose: a string or
    comment in the code mentioning another table used to drag the guard onto it
    (or, with two matches, make it skip). SQL position is unambiguous, so it
    wins; a guard must never be pointed at a table the answer did not come from.
    """
    if not code:
        return [], []
    strong: list[str] = []
    weak: list[str] = []
    for t in names:
        hit_strong = hit_weak = False
        for form in {str(t), re.sub(r"\W+", "_", str(t)).strip("_")}:
            if not form:
                continue
            esc = re.escape(form)
            if re.search(rf"\b(?:from|join|using)\s+[\"'`\[]?{esc}(?![0-9A-Za-z_])",
                         code, re.IGNORECASE):
                hit_strong = True
            if re.search(rf"(?<![0-9A-Za-z_]){esc}(?![0-9A-Za-z_])",
                         code, re.IGNORECASE):
                hit_weak = True
        if hit_strong:
            strong.append(t)
        elif hit_weak:
            weak.append(t)
    return strong, weak


def _source_frame(ctx, code: str | None = None):
    """The rows an answer was actually computed from.

    The agent may query ANY table the schema block lists, but the table-derived
    guards used to check every answer against `ctx.active_table` — so an answer
    about a second loaded table was judged against a different table's rows: a
    false objection that burned retries and could ship a 'correction' computed
    from a table the answer never touched. Prefer the table the code names, and
    fall back to the active table when the code names none (the single-table
    case, unchanged). A query that reads two tables (a JOIN) has no single
    source — guessing one would mean reporting on rows the answer did not come
    from, so return None and let the caller skip.
    """
    tables = getattr(ctx, "tables", None) or {}
    if not tables:
        return None
    strong, weak = _code_tables(code or "", tables.keys())
    named = strong or weak       # a FROM/JOIN match outranks a bare word
    if len(named) > 1:
        return None
    if len(named) == 1:
        return tables[named[0]]
    return tables.get(getattr(ctx, "active_table", None))


def _breakdown_coverage(columns, rows, ctx, question: str, code: str | None = None):
    """(label_col, n_result, n_table, missing) when a break-down/who-leads
    result omits groups the source table has. The model writes LIMIT 1/2 on
    these questions (live: 2 of 4 regions) and a trimmed result reads as if
    the hidden groups do not exist. Conservative: explicit break-down
    vocabulary only, no top-N / filter / time markers, and only when the
    result's label column name maps onto a real table column."""
    q = question or ""
    if not rows or not columns:
        return None
    if not _BREAKDOWN_Q.search(q) or _COVERAGE_EXEMPT.search(q):
        return None
    label_idx = None
    for i, c in enumerate(columns):
        vals = [r[i] if isinstance(r, (list, tuple)) else
                (r.get(c) if isinstance(r, dict) else None)
                for r in rows[:25]]
        if vals and sum(isinstance(v, str) for v in vals) >= len(vals) * 0.8:
            label_idx = i
            break
    if label_idx is None:
        return None
    label = str(columns[label_idx])
    df = _source_frame(ctx, code)
    if df is None:
        return None
    col = next((c for c in df.columns if str(c).lower() == label.lower()), None)
    if col is None:
        return None
    table_vals = {str(v) for v in df[col].dropna().unique()}
    if len(table_vals) < 2:
        return None
    seen = set()
    for r in rows:
        v = (r[label_idx] if isinstance(r, (list, tuple))
             else (r.get(label) if isinstance(r, dict) else None))
        if v is not None and str(v) != "":
            seen.add(str(v))
    missing = sorted(table_vals - seen)
    if seen and len(seen) < len(table_vals) and missing:
        return label, len(seen), len(table_vals), missing
    return None


def _percentize_answer(answer: str, columns, rows, question: str) -> str:
    """When the question explicitly asks for a percentage and the result is a
    single fraction between 0 and 1, rewrite the answer as a percent sentence.
    Small models echo the decimal verbatim ('churn_rate=0.385') even when the
    prompt begs for percent form — the value is RIGHT, the format is wrong.
    Deterministic fix, no extra model call."""
    q = (question or "").lower()
    wants_percent = ("percentage" in q or "what percent" in q
                     or re.search(r"\brate\b.*percent|percent.*\brate\b", q))
    a = (answer or "").lower()
    # the answer already states a percent — never overwrite real prose.
    # Word-boundary match: the COLUMN name 'percentage' must not trigger this.
    if (not wants_percent or not rows or re.search(r"\bpercent\b", a)
            or "%" in a):
        return answer
    row = rows[0]
    cells = list(row.values()) if isinstance(row, dict) else list(row)
    numeric = [c for c in cells if isinstance(c, (int, float))
               and not isinstance(c, bool)]
    if len(cells) != 1 or len(numeric) != 1:
        return answer
    v = numeric[0]
    if not (0.0 < v < 1.0):
        return answer
    pct = v * 100
    pct_s = f"{pct:.1f}".rstrip("0").rstrip(".")
    col = columns[0] if columns else "rate"
    col_s = str(col).replace("_", " ").strip()
    return f"The {col_s} is {pct_s}%."


def _conditional_pct_sanity(col, row, ctx, question: str, code: str | None = None):
    """For 'Among <subset>, what percentage have <condition>' questions with a
    single 0-1-ish result: recompute subset size, condition count inside the
    subset, and whole-table size. Returns (true_pct, n_subset, n_all) when the
    reported value matches subset_cond/n_all but NOT subset_cond/n_subset —
    i.e. the model divided by the whole table. Requires the question to name
    the subset condition ('below 500') and the answer column to look like a
    rate (percentage/pct/rate/share). Returns None when not applicable."""
    q = (question or "").lower()
    m = re.search(r"\bamong\b.+?\bbelow\s+(\d+(?:\.\d+)?)", q)
    if not m or not re.search(r"\b(percentage|pct|rate|share|percent)\b", q):
        return None
    if not any(k in q for k in ("overdraft", "flag", "churn", "default",
                                "returned", "readmit")):
        return None
    df = _source_frame(ctx, code)
    if df is None:
        return None
    v = row if not isinstance(row, dict) else row.get(col)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    thr = float(m.group(1))
    num_cols = df.select_dtypes("number").columns
    flag_col = next((c for c in df.columns
                     if str(c).lower() in ("overdraft_flag", "flag")
                     or str(c).lower().endswith(("_flag", "ed_30d"))), None)
    bal_col = next((c for c in num_cols
                    if any(k in str(c).lower() for k in ("balance", "amount",
                                                        "tenure", "age"))), None)
    if not (flag_col and bal_col):
        return None
    sub = df[df[bal_col] < thr]
    n_sub, n_all = len(sub), len(df)
    if n_sub == 0 or n_sub == n_all:
        return None
    try:
        cond = sub[flag_col].astype(float)
    except (TypeError, ValueError):
        return None
    n_cond = float((cond > 0).sum())
    pct_sub = n_cond / n_sub * 100
    frac_all = n_cond / n_all          # the WRONG value the model reports
    if abs(v - frac_all) <= max(0.005, frac_all * 0.05) \
            and abs(v - pct_sub) > max(1.0, pct_sub * 0.15):
        return pct_sub, n_sub, n_all
    return None


_SUPERLATIVE_MAX = ("highest", "most", "largest", "biggest", "maximum", "best")
_SUPERLATIVE_MIN = ("lowest", "smallest", "fewest", "minimum", "least", "worst")


def _winner_mismatch(answer, columns, rows, question: str):
    """(true_label, metric_col, 'max'|'min') when the answer names a
    superlative winner that is NOT the row actually winning by the metric
    column. Closes the mislabeled-winner gap: correct numbers in the table,
    wrong category crowned in prose (Clothing vs Furniture profit).
    Conservative: fires only when the answer cites ANOTHER row's label,
    skips ties and ambiguous shapes (no clean label+metric columns)."""
    q = (question or "").lower()
    a = (answer or "").lower()
    wants_max = any(w in q for w in _SUPERLATIVE_MAX)
    wants_min = any(w in q for w in _SUPERLATIVE_MIN)
    if wants_max == wants_min:          # none, or both — ambiguous
        return None
    cols = list(columns or [])
    rows = list(rows or [])
    if len(rows) < 2 or len(cols) < 2:
        return None

    def _cell(r, i):
        return r[i] if isinstance(r, (list, tuple)) else (r.get(cols[i]) if isinstance(r, dict) else None)

    label_col = metric_col = None
    num_cols = []
    q_tokens = {t[:-1] if t.endswith("s") else t
                for t in re.findall(r"[a-z]+", q)}
    for i in range(len(cols)):
        vals = [_cell(r, i) for r in rows[:25]]
        n_str = sum(isinstance(v, str) for v in vals)
        n_num = sum(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in vals)
        if n_str >= len(vals) * 0.8 and label_col is None:
            label_col = i
        elif n_num >= len(vals) * 0.8:
            num_cols.append(i)
    if label_col is None or not num_cols:
        return None
    # the metric is the numeric column the QUESTION asks about: match its
    # name tokens against the question ('highest profit' -> col 'profit').
    # Fallback: first numeric column (better than nothing).
    for i in num_cols:
        toks = {t[:-1] if t.endswith("s") else t
                for t in re.findall(r"[a-z]+", str(cols[i]).lower())}
        if toks & q_tokens:
            metric_col = i
            break
    if metric_col is None:
        metric_col = num_cols[0]
    try:
        ranked = sorted(rows, key=lambda r: float(_cell(r, metric_col)),
                        reverse=wants_max)
    except (TypeError, ValueError):
        return None
    # tie for the lead → no trustworthy single winner
    try:
        if abs(float(_cell(ranked[0], metric_col))
               - float(_cell(ranked[1], metric_col))) < 1e-9:
            return None
    except (TypeError, ValueError):
        return None
    winner_label = _cell(ranked[0], label_col)
    if winner_label is None or str(winner_label).lower() in a:
        return None                      # true winner named (or no label)
    wv = _cell(ranked[0], metric_col)
    if isinstance(wv, (int, float)):
        s = f"{abs(wv):,.2f}".rstrip("0").rstrip(".")
        if s in a.replace(",", "") or str(wv) in a:
            return None                  # winner's value cited — prose sloppy, not wrong
    for r in ranked[1:]:
        other = _cell(r, label_col)
        if other and str(other).lower() in a:
            return (str(winner_label), str(cols[metric_col]),
                    "max" if wants_max else "min")
    return None


# Deliberately narrow: 'most', 'top', 'least' and 'bottom' routinely mean
# something other than the extreme ('most groups sit near …') and would bounce
# correct prose.
_SUP_WORD_MAX = re.compile(r"\b(highest|largest|biggest|max(?:imum)?)\b", re.I)
_SUP_WORD_MIN = re.compile(r"\b(lowest|smallest|fewest|min(?:imum)?)\b", re.I)


def _superlative_value_mismatch(answer, columns, rows):
    """(word, stated, true_val, true_label, 'max'|'min') when the answer calls
    a value the highest/lowest and that value is a row of the result but NOT
    its extreme. Live: a 200-group camera breakdown ended "the highest average
    price is $62,472.00 … while the lowest is $49,587.00" — 49,587 is a real
    row (camera 70), the true lowest is 13,313.00 — and the truth guard blesses
    it because the number IS in the result. Conservative: the number must sit
    within 60 characters after the superlative word, be a value of the metric
    column, and differ from the true extreme."""
    a = answer if isinstance(answer, str) else str(answer or "")
    if not a:
        return None
    label_i, metric_i = _label_metric_cols(columns, rows)
    if label_i is None:
        # any size: the extremes of the metric column are all this guard needs
        label_i, metric_i = _small_numeric_label_metric(columns, rows, max_rows=None)
    if label_i is None or metric_i is None:
        return None
    cols = list(columns or [])
    rows = list(rows or [])

    def _cell(r, i):
        return (r[i] if isinstance(r, (list, tuple))
                else (r.get(cols[i]) if isinstance(r, dict) else None))

    cells = []
    for r in rows[:50]:
        v = _cell(r, metric_i)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            cells.append((float(v), _cell(r, label_i)))
    if len(cells) < 2:
        return None
    top_v, top_lbl = max(cells, key=lambda t: t[0])
    bot_v, bot_lbl = min(cells, key=lambda t: t[0])
    if top_v == bot_v:
        return None
    metric_vals = [v for v, _ in cells]
    for rex, sup in ((_SUP_WORD_MAX, "max"), (_SUP_WORD_MIN, "min")):
        for m in rex.finditer(a):
            window = a[m.end():m.end() + 60]
            nums = _extract_numbers(window)
            if not nums:
                continue
            n = nums[0]
            true_v, true_lbl = (top_v, top_lbl) if sup == "max" else (bot_v, bot_lbl)
            if abs(n - true_v) <= max(0.051, abs(true_v) * 0.002):
                break                        # that superlative is stated right
            if any(abs(n - v) <= max(0.051, abs(v) * 0.002) for v in metric_vals):
                return (m.group(1).lower(), n, true_v, true_lbl, sup)
            break                            # not a result value — truth guard owns it
    return None


def _profit_sanity(columns, rows, ctx, question: str, code: str | None = None):
    """Detect a profit-like result that contradicts the (price − cost) × qty
    ground truth recomputed from the table. Returns (true_total, (price_col,
    cost_col, qty_col), bad_value) or None. Models gravitate to a revenue-looking
    column (e.g. Sales_Amount ≈ $5k/row when price*qty ≈ $64k/row) and report
    negative millions; the truth guard cannot catch it because the wrong numbers
    DO exist in the result. This check recomputes independently."""
    q = (question or "").lower()
    if not any(w in q for w in ("profit", "margin")):
        return None
    cols = list(columns or [])
    rows = list(rows or [])
    if not rows or not cols:
        return None
    pcol = next((c for c in cols
                 if "profit" in str(c).lower() or "margin" in str(c).lower()), None)
    if pcol is None:
        return None
    i = cols.index(pcol)
    vals = []
    for r in rows[:10]:
        v = r[i] if isinstance(r, (list, tuple)) else (r.get(pcol) if isinstance(r, dict) else None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    if not vals:
        return None
    df = _source_frame(ctx, code)
    if df is None:
        return None
    lcols = {str(c).lower(): c for c in df.columns}
    p = next((lcols[c] for c in lcols if "price" in c), None)
    co = next((lcols[c] for c in lcols if "cost" in c), None)
    qt = next((lcols[c] for c in lcols
               if "quantity" in c or c in ("qty", "units", "quantity_sold")), None)
    if not (p and co and qt):
        return None
    try:
        true_total = float(((df[p] - df[co]) * df[qt]).sum())
    except Exception:
        return None
    res_total = sum(vals)
    if true_total > 0 and res_total < 0:
        return true_total, (p, co, qt), vals[0]
    if true_total != 0 and abs(res_total) > 5 * abs(true_total):
        return true_total, (p, co, qt), vals[0]
    return None


def _verification_request(answer: str, tables_json: str, reason: str | None = None) -> str:
    why = reason or ("Your explanation CONTRADICTS the executed result below. Rewrite the "
                     "'explanation' field so it agrees with the result exactly — the result is "
                     "the ground truth, your text is wrong.")
    return (
        why +
        " Keep the same JSON shape (plan/language/code unchanged), only fix explanation.\n\n"
        f"Executed result (the ground truth your text must match):\n{tables_json}\n\n"
        f"Your contradicted explanation was: {answer!r}\n\n"
        "Rules: report exactly what the result contains, including values inside list-like cells "
        "(a row like {'missing_months': '2023-06'} means ONE month is missing: 2023-06 — say that, "
        "never 'no gaps'). Use only numbers/values that appear in the result. State it directly and "
        "confidently in one or two sentences; do not hedge and do not invent anything."
    )


def _contradicts_result(answer: str, res) -> bool:
    """True when a qualitative answer directly denies what a single-row
    result shows — e.g. text says 'no gaps' while result_df holds a
    missing-months list. Catches narrated lies that contain no numbers."""
    tables = getattr(res, "tables", []) or []
    if not tables:
        return False
    t = tables[0]
    rows = t.get("rows") or []
    if len(rows) != 1:
        return False
    row = rows[0]
    vals = [v for v in (row.values() if isinstance(row, dict) else row)
            if v is not None and v != "" and v != []]
    if len(vals) != 1:
        return False
    v = vals[0]
    a = (answer or "").lower()
    # A single NUMBER cell is verified as a number, not as text: the raw repr
    # of a float is noise (CORR() of two perfectly linked columns is
    # 0.9999999999999998, so the string never appears in prose and every
    # correct correlation answer was bounced as a contradiction). The answer
    # must still STATE the value — in the truth guard's tolerance, or as the
    # percent form of a fraction.
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        target = float(v)
        for n in _extract_numbers(answer or ""):
            if abs(n - target) <= max(0.051, abs(target) * 0.002):
                return False
            if target and abs(n / 100.0 - target) <= max(0.00051, abs(target) * 0.002):
                return False
        return True
    # string-list / string cell: the answer must cite the EXACT value —
    # a near-miss (2023-05 vs 2023-06) or omission both count as contradiction
    if isinstance(v, (list, tuple)):
        items = [str(x).lower() for x in v]
    else:
        items = [str(v).lower()]
    nonempty = [x for x in items if x]
    for item in nonempty:
        if item not in a:
            return True
        # every date-like token in the answer that resembles the cell must be it
        for tok in re.findall(r"\d{4}-\d{2}(?:-\d{2})?", a):
            if tok not in nonempty and abs(len(tok) - len(item)) <= 2:
                return True          # a different month/year cited alongside
    # denial language against a populated result
    denies = ("no ", "not ", "none", "doesn't", "does not", "no gaps")
    if any(d in a for d in denies):
        return True
    return False


def guard_report(answer: str, res, question: str = "") -> list[GuardOutcome]:
    """The receipt for one finished answer — only checks that actually ran.

    truth/contradiction are the pair that can run on every path, so they are
    the only ones reported as passed. With no result rows there is nothing to
    trace a number against: both come back "skipped", which makes
    AnalysisResult.verified False instead of a vacuous 6/6 tick.
    """
    rows = res.tables[0]["rows"] if res.tables else []
    if not rows:
        return [GuardOutcome("truth", "skipped",
                             "no result rows — nothing to check numbers against")]
    numbers = _extract_numbers(answer or "")
    if numbers:
        traced = _answer_matches_data(answer, res, question)
        out = [GuardOutcome(
            "truth", "passed" if traced else "failed",
            f"{len(numbers)} number(s), each found in the result" if traced
            else "a number in the answer has no source in the result")]
    else:
        # nothing numeric to trace: a pass, but not a claim of one
        out = [GuardOutcome("truth", "passed", "the answer names no numbers")]
    contra_ok = not _contradicts_result(answer, res)
    out.append(GuardOutcome("contradiction", "passed" if contra_ok else "failed",
                            "text agrees with the rows" if contra_ok
                            else "the text argues against its own table"))
    return out


def _direction_contradiction(answer, columns, rows, question: str):
    """(subject_text, claimed_word, val_subject, val_other) when a direction
    question ('is A higher than B') gets an answer whose comparative claim
    contradicts the two values in the result. Heuristic: in a 2-value result,
    the number cited FIRST in the answer belongs to the SUBJECT of the claim.
    Catches: correct numbers, backwards conclusion (live Walmart failure)."""
    q = (question or "").lower()
    if not re.search(r"\b(higher|lower|more|less|greater|fewer|bigger|smaller)\b", q) \
            and not re.search(r"\b(do|does|did|is|are|was|were)\b.*\bthan\b", q):
        return None
    a = answer or ""
    al = a.lower()
    m = re.search(r"\b(higher|lower|more|less|greater|fewer|bigger|smaller)\b", al)
    if not m:
        return None
    word = m.group(1)
    claim_higher = word in ("higher", "more", "greater", "bigger")
    subject_text = re.search(rf"([a-z ][a-z ]{{2,60}}?)\b{word}\b", al)
    subject_text = subject_text.group(1).strip() if subject_text else ""
    # result must hold exactly two numeric values
    cols = list(columns or [])
    rows = list(rows or [])
    if not rows or not cols:
        return None

    def _cell(r, i):
        return r[i] if isinstance(r, (list, tuple)) else (r.get(cols[i]) if isinstance(r, dict) else None)

    vals = []
    if len(rows) == 1:
        vals = [float(_cell(rows[0], i)) for i in range(len(cols))
                if isinstance(_cell(rows[0], i), (int, float))
                and not isinstance(_cell(rows[0], i), bool)]
    else:
        # multi-row: compare the METRIC column (question-token match, else the
        # last numeric column) — never a 0/1 flag column
        q_tokens = {t[:-1] if t.endswith("s") else t
                    for t in re.findall(r"[a-z]+", q)}
        num_cols = [i for i in range(len(cols))
                    if all(isinstance(_cell(r, i), (int, float))
                           and not isinstance(_cell(r, i), bool)
                           for r in rows[:10])]
        metric = None
        candidates = [i for i in num_cols
                      if {t[:-1] if t.endswith("s") else t
                          for t in re.findall(r"[a-z]+", str(cols[i]).lower())}
                      & q_tokens] or num_cols
        # a 0/1 flag column is a group marker, not the compared metric —
        # prefer a non-flag candidate when one exists
        non_flag = [i for i in candidates
                    if not all(float(_cell(r, i)) in (0.0, 1.0) for r in rows[:10])]
        candidates = non_flag or candidates
        if candidates:
            metric = candidates[0]
        if metric is None:
            return None
        vals = [float(_cell(r, metric)) for r in rows[:2]]
    if len(vals) != 2:
        return None
    # both values must be cited in the answer
    ans_nums = _extract_numbers(a)
    cited = [n for n in ans_nums
             if any(abs(n - v) <= max(0.02, abs(v) * 0.005) for v in vals)]
    if len(set(cited)) < 2:
        return None
    v_first, v_second = cited[0], cited[1]
    ok = (v_first > v_second) if claim_higher else (v_first < v_second)
    if ok:
        return None
    return (subject_text or "the first group", word, v_first, v_second)


def _pct_attribution(answer, columns, rows, question: str):
    """(subject, other, v_subj, v_other, stated_pct, true_pct) when the answer
    says "<label> ... <X>% ... [than|over|vs|...] <label>" and X does not
    equal (a - b) / b * 100 for the two rows it names. A traceable number can
    still be attributed to the WRONG pair: live, "south leads by 16.87% over
    west" shipped as verified because 16.87 matches another pair (east-west is
    16.76%) under the truth guard's pair tolerance. Conservative: exactly two
    result labels may appear in the answer, the percentage must sit between
    them with an explicit comparison word, and both rows must carry the
    numeric metric."""
    a_text = answer or ""
    cols = list(columns or [])
    rows = list(rows or [])
    if not a_text or len(rows) < 2 or len(cols) < 2:
        return None

    def _cell(r, i):
        return (r[i] if isinstance(r, (list, tuple))
                else (r.get(cols[i]) if isinstance(r, dict) else None))

    label_i = metric_i = None
    num_cols: list[int] = []
    for i in range(len(cols)):
        vals = [_cell(r, i) for r in rows[:25]]
        n_str = sum(isinstance(v, str) for v in vals)
        n_num = sum(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in vals)
        if n_str >= len(vals) * 0.8 and label_i is None:
            label_i = i
        elif n_num >= len(vals) * 0.8:
            num_cols.append(i)
    if label_i is None or not num_cols:
        return None
    q_tokens = {t[:-1] if t.endswith("s") else t
                for t in re.findall(r"[a-z]+", (question or "").lower())}
    for i in num_cols:
        toks = {t[:-1] if t.endswith("s") else t
                for t in re.findall(r"[a-z]+", str(cols[i]).lower())}
        if toks & q_tokens:
            metric_i = i
            break
    if metric_i is None:
        metric_i = num_cols[-1]

    labels: dict[str, int] = {}
    for j, r in enumerate(rows[:25]):
        v = _cell(r, label_i)
        if isinstance(v, str) and v and v not in labels:
            labels[v] = j
    present = [l for l in labels
               if re.search(rf"\b{re.escape(l)}\b", a_text, re.I)]
    if len(present) != 2:
        return None                      # 0/1: nothing to attribute; 3+: ambiguous
    parsed = None
    for s, o in (present, list(reversed(present))):
        m = re.search(
            rf"\b{re.escape(s)}\b(.{{0,90}}?)([-+]?\d[\d,]*\.?\d*)\s*%"
            rf"(.{{0,90}}?)\b{re.escape(o)}\b", a_text, re.I | re.S)
        if m:
            parsed = (s, o, m)
            break
    if parsed is None:
        return None
    subj, other, m = parsed
    bridge = (m.group(1) + m.group(3)).lower()
    if not any(w in bridge for w in ("than", "over", "vs", "versus",
                                     "compared to", "ahead of", "above")):
        return None
    try:
        stated = float(m.group(2).replace(",", ""))
        v_subj = float(_cell(rows[labels[subj]], metric_i))
        v_other = float(_cell(rows[labels[other]], metric_i))
    except (TypeError, ValueError):
        return None
    if v_other == 0:
        return None
    true_pct = (v_subj - v_other) / v_other * 100.0
    if abs(stated - true_pct) <= max(0.11, abs(true_pct) * 0.002):
        return None
    return subj, other, v_subj, v_other, stated, true_pct

