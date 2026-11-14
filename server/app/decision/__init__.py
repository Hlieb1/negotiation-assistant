import logging
from pathlib import Path

from .base import DecisionModule
from .fuzzy import FuzzyDecision
from .hybrid import HybridDecision
from .llm import LLMDecision, LLMError

logger = logging.getLogger("negotiation")
PACKAGE_DIR = Path(__file__).resolve().parent


def build_decisions(settings, warm_up_llm: bool = True) -> dict[str, DecisionModule]:
    from ..config import resolve_path

    decision = settings.decision
    fuzzy = FuzzyDecision(resolve_path(decision.fuzzy_rules_path), resolve_path(decision.replies_path))
    llm_settings = decision.llm
    llm = LLMDecision(llm_settings.base_url, llm_settings.model, resolve_path(llm_settings.prompt_path), fuzzy,
                      llm_settings.timeout_s, llm_settings.temperature, llm_settings.max_tokens,
                      llm_settings.keep_alive)
    if warm_up_llm and llm_settings.enabled:
        try:
            llm.warm_up()
        except LLMError as exc:
            logger.warning("LLM is not available (%s); llm and hybrid modes will fall back to fuzzy", exc)
    return {"fuzzy": fuzzy, "llm": llm, "hybrid": HybridDecision(fuzzy, llm)}
