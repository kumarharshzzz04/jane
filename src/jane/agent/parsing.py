"""Model reply parsing — defensive JSON extraction."""
import json
import re



def _parse_model_json(raw: str) -> dict:
    """Ollama returns content constrained to JSON, but be defensive."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _parse_claims(raw) -> list[dict] | None:
    """Extract the OPTIONAL claims array from a model reply.

    Fallback contract (spec §1): ANY weirdness -> None -> the orchestrator
    runs the legacy no-claims path. Never raises."""
    try:
        if isinstance(raw, str):
            raw = _parse_model_json(raw)
        claims = raw.get("claims") if isinstance(raw, dict) else None
        if not isinstance(claims, list):
            return None
        clean: list[dict] = []
        for c in claims:
            if not isinstance(c, dict):
                return None
            if "value" not in c:          # a claim without a value is not a claim
                return None
            comp = c.get("comparison")
            if comp is not None and not isinstance(comp, dict):
                return None
            clean.append(c)
        return clean or None
    except Exception:
        return None

