import json

import httpx
import pytest

from app.config import SERVER_DIR, Settings
from app.decision.base import STRATEGIES, Decision, StateVector
from app.decision.fuzzy import FuzzyDecision, build_variables, parse_rules
from app.decision.hybrid import HybridDecision
from app.decision.llm import LLMDecision, describe_voice
from app.hint import HintPolicy

SETTINGS = Settings()
RULES = SERVER_DIR / SETTINGS.decision.fuzzy_rules_path
REPLIES = SERVER_DIR / SETTINGS.decision.replies_path
PROMPT = SERVER_DIR / SETTINGS.decision.llm.prompt_path
STATES = ("злість", "напруга", "сумнів", "згода", "нейтральний")


def state(main: str, degree: float = 0.6, confidence: float = 0.5) -> StateVector:
    rest = (1 - degree) / (len(STATES) - 1)
    return StateVector(main, {s: degree if s == main else rest for s in STATES}, confidence)


@pytest.fixture(scope="module")
def fuzzy() -> FuzzyDecision:
    return FuzzyDecision(RULES, REPLIES, seed=0)


def test_rule_base_size_and_vocabulary(fuzzy):
    assert 15 <= len(fuzzy.rules) <= 30
    inputs, outputs = build_variables()
    with pytest.raises(ValueError, match="bad clause"):
        parse_rules("IF anger IS furious THEN calm IS strong", inputs, outputs)
    with pytest.raises(ValueError, match="mixing"):
        parse_rules("IF anger IS high AND doubt IS low OR tension IS high THEN calm IS strong", inputs, outputs)


@pytest.mark.parametrize("main, strategy, tone", [
    ("злість", "calm", "soft"),
    ("напруга", "calm", "soft"),
    ("сумнів", "clarify", "neutral"),
    ("згода", "close", "firm"),
    ("нейтральний", "argue", "neutral"),
])
def test_fuzzy_strategy_for_each_state(fuzzy, main, strategy, tone):
    decision = fuzzy.decide(state(main), {}, "", [])
    assert decision.strategy == strategy and decision.details["tone_level"] == tone
    assert set(decision.details["activations"]) == set(STRATEGIES)


def test_fuzzy_never_closes_on_anger_even_with_low_confidence(fuzzy):
    for confidence in (0.05, 0.15, 0.5, 0.9):
        assert fuzzy.decide(state("злість", 0.45, confidence), {}, "", []).strategy == "calm"


def test_replies_do_not_repeat_in_a_row(fuzzy):
    history: list[Decision] = []
    for _ in range(12):
        decision = fuzzy.decide(state("згода"), {}, "", history)
        assert decision.reply not in {d.reply for d in history[-3:]}
        history.append(decision)


def test_replies_bank_has_three_to_five_per_strategy():
    replies = json.loads(REPLIES.read_text("utf-8"))["replies"]
    assert set(replies) == set(STRATEGIES)
    assert all(3 <= len(options) <= 5 for options in replies.values())


def llm_with(handler, fuzzy, timeout_s=1.0) -> LLMDecision:
    module = LLMDecision("http://llm", "test", PROMPT, fuzzy, timeout_s=timeout_s)
    module._client = httpx.Client(base_url="http://llm", transport=httpx.MockTransport(handler))
    return module


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


def test_llm_valid_json_reply(fuzzy):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return chat_response('{"reply": "Що саме вас турбує?", "tone": "спокійний"}')

    decision = llm_with(handler, fuzzy).decide(state("сумнів"), {"pause_ratio": 1.5}, "Не знаю...", [])
    assert decision.source == "llm" and decision.reply == "Що саме вас турбує?"
    assert seen["format"]["required"] == ["reply", "tone"] and seen["think"] is False
    assert "вагається" in seen["messages"][-1]["content"] and "Не знаю" in seen["messages"][-1]["content"]


@pytest.mark.parametrize("content", ['{"reply": "так"}', "не json", '{"reply": "", "tone": "x"}'])
def test_llm_invalid_output_falls_back_to_fuzzy(fuzzy, content):
    decision = llm_with(lambda r: chat_response(content), fuzzy).decide(state("згода"), {}, "", [])
    assert decision.source == "llm_fallback" and decision.strategy == "close"


def test_llm_timeout_falls_back(fuzzy):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    decision = llm_with(handler, fuzzy).decide(state("злість"), {}, "", [])
    assert decision.source == "llm_fallback" and decision.details["error"] == "timeout"


def test_hybrid_keeps_fuzzy_strategy_and_tone(fuzzy):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return chat_response('{"reply": "Розумію вас, давайте спокійно розберемося.", "tone": "різкий"}')

    decision = HybridDecision(fuzzy, llm_with(handler, fuzzy)).decide(state("злість"), {}, "", [])
    assert decision.source == "hybrid" and decision.strategy == "calm"
    assert decision.tone == "м'який, спокійний, повільніше"
    assert "Обов'язкова стратегія: заспокоїти" in seen["messages"][-1]["content"]


def test_describe_voice():
    assert describe_voice({}) == "голос звичний для цього співрозмовника"
    text = describe_voice({"loudness_db": 2.0, "speech_rate": -1.5, "pitch_st": 0.3})
    assert "голосніше" in text and "повільніше" in text and "тон" not in text


def test_hint_policy_rate_stability_and_change():
    policy = HintPolicy(min_interval_s=4, stable_s=1, min_confidence=0.1)
    angry = state("злість")
    assert not policy.should_hint(angry, 0)
    assert policy.should_hint(angry, 1000)
    assert not policy.should_hint(angry, 9000)
    calm = state("нейтральний")
    assert not policy.should_hint(calm, 3000)
    assert not policy.should_hint(calm, 4500)
    assert policy.should_hint(calm, 5000)
    assert not policy.should_hint(state("згода", confidence=0.05), 20000)
