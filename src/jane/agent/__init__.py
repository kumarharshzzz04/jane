"""Jane analysis agent package — re-exports the historical module surface."""
from .orchestrator import AgentContext, AnalystAgent  # noqa: F401
from .prompting import SYSTEM_PROMPT  # noqa: F401
from .parsing import _parse_model_json  # noqa: F401
from .charting import _bind_chart_data  # noqa: F401
from .verification import (  # noqa: F401
    _answerable_refusal, _answer_matches_data, _association_missing_coefficient,
    _association_question, _conditional_pct_sanity, _contradicts_result,
    _direction_contradiction, _extract_numbers, _factual_answer,
    _has_prompt_placeholders, _missing_label_column, _percentize_answer,
    _profit_sanity, _superlative_value_mismatch,
    _rate_question, _tidy_answer, _verification_request, _winner_mismatch)
from .error_hints import _error_hint  # noqa: F401
