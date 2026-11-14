import json
import re
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from .base import STRATEGIES_UK, Decision, DecisionModule, StateVector

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"reply": {"type": "string"}, "tone": {"type": "string"}},
    "required": ["reply", "tone"],
}
VOICE_PHRASES = {
    "loudness_db": ("говорить голосніше, ніж зазвичай", "говорить тихіше, ніж зазвичай"),
    "pitch_st": ("тон голосу вищий за звичний", "тон голосу нижчий за звичний"),
    "f0_std_st": ("інтонація жвава, тон сильно змінюється", "інтонація монотонна"),
    "speech_rate": ("говорить швидше, ніж зазвичай", "говорить повільніше, ніж зазвичай"),
    "pause_ratio": ("робить багато пауз, вагається", "говорить майже без пауз"),
}


DIGITS = re.compile(r"(\d[\d ]*\d|\d)(?:[.,]\d+)?\s*(тис)?", re.IGNORECASE)
WORDS = re.compile(r"\b(одн\w*|дв\w*|три|чотири|п['’]ять|шість|сім|вісім|дев['’]ять|десять)\s+тис",
                   re.IGNORECASE)
WORD_VALUES = {"одн": 1, "дв": 2, "три": 3, "чот": 4, "п'я": 5, "п’я": 5, "шіс": 6, "сім": 7, "віс": 8,
               "дев": 9, "дес": 10}
MIN_CHECKED_NUMBER = 100


def mentioned_numbers(text: str) -> set[int]:
    values = set()
    for digits, thousands in DIGITS.findall(text):
        value = int(digits.replace(" ", ""))
        values.add(value * 1000 if thousands else value)
    for word in WORDS.findall(text):
        values.add(WORD_VALUES.get(word[:3].lower(), 0) * 1000)
    return {v for v in values if v >= MIN_CHECKED_NUMBER}


def unsupported_numbers(reply: str, sources: str) -> set[int]:
    return mentioned_numbers(reply) - mentioned_numbers(sources)


class LLMReply(BaseModel):
    reply: str = Field(min_length=3, max_length=240)
    tone: str = Field(min_length=2, max_length=80)

    @field_validator("reply", "tone")
    @classmethod
    def strip(cls, value: str) -> str:
        return value.strip().strip("«»\"")


class LLMError(Exception):
    pass


def describe_voice(prosody_z: dict[str, float], threshold: float = 1.0) -> str:
    phrases = [VOICE_PHRASES[k][0 if v > 0 else 1] for k, v in prosody_z.items()
               if k in VOICE_PHRASES and abs(v) >= threshold]
    return "; ".join(phrases) or "голос звичний для цього співрозмовника"


def last_counterpart_phrase(dialogue: str) -> str:
    for line in reversed(dialogue.splitlines()):
        speaker, _, text = line.partition(": ")
        if speaker != "Я" and text:
            return text
    return ""


def describe_situation(state: StateVector, voice_features: dict, dialogue: str, history: list[Decision],
                       strategy: str | None, tone: str | None, context: str = "") -> str:
    others = ", ".join(f"{name} {degree:.2f}" for name, degree in
                       sorted(state.degrees.items(), key=lambda kv: kv[1], reverse=True) if name != state.name)
    lines = [f"Контекст переговорів від користувача: {context}" if context else "Контекст переговорів не задано."]
    lines.append(f"Розмова (останні репліки):\n{dialogue}" if dialogue else "Слова розмови поки не розпізнано.")
    last = last_counterpart_phrase(dialogue)
    if last:
        lines.append(f"Дай відповідь саме на останню репліку співрозмовника: «{last}»")
    lines += [
        f"Стан співрозмовника: {state.name} (ступінь {state.degrees.get(state.name, 0):.2f}, "
        f"впевненість розпізнавання {state.confidence:.2f}). Інші стани: {others}.",
        f"Голос співрозмовника: {describe_voice(voice_features)}.",
    ]
    if history:
        lines.append("Попередні підказки: " + " | ".join(f"«{d.reply}»" for d in history[-3:]) + ". Не повторюй їх.")
    if strategy:
        lines.append(f"Обов'язкова стратегія: {STRATEGIES_UK[strategy]}. Тон: {tone}.")
    return "\n".join(lines)


class LLMDecision(DecisionModule):
    name = "llm"

    def __init__(self, base_url: str, model: str, prompt_path: Path, fallback: DecisionModule,
                 timeout_s: float = 4.0, temperature: float = 0.6, max_tokens: int = 120,
                 keep_alive: str = "30m"):
        self.model = model
        self.fallback = fallback
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        self.system_prompt: str = prompt["system"]
        self.examples: list[dict] = prompt["examples"]
        self._client = httpx.Client(base_url=base_url)

    def warm_up(self) -> None:
        self.generate(self.examples[0]["situation"], timeout_s=300)

    def _messages(self, situation: str) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for example in self.examples:
            messages.append({"role": "user", "content": example["situation"]})
            messages.append({"role": "assistant", "content": json.dumps(example["answer"], ensure_ascii=False)})
        messages.append({"role": "user", "content": situation})
        return messages

    def generate(self, situation: str, timeout_s: float | None = None) -> LLMReply:
        payload = {
            "model": self.model,
            "messages": self._messages(situation),
            "format": RESPONSE_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        try:
            response = self._client.post("/api/chat", json=payload, timeout=timeout_s or self.timeout_s)
            response.raise_for_status()
            return LLMReply.model_validate_json(response.json()["message"]["content"])
        except httpx.TimeoutException as exc:
            raise LLMError("timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"http: {exc}") from exc
        except (ValidationError, KeyError, ValueError) as exc:
            raise LLMError(f"invalid response: {str(exc)[:200]}") from exc

    def generate_checked(self, situation: str, sources: str) -> tuple[LLMReply, dict]:
        answer = self.generate(situation)
        invented = unsupported_numbers(answer.reply, sources)
        if not invented:
            return answer, {"retries": 0}
        note = ", ".join(str(n) for n in sorted(invented))
        answer = self.generate(f"{situation}\nУВАГА: у попередньому варіанті були числа {note}, яких немає ні в "
                               f"контексті, ні в розмові. Не називай їх і не пропонуй нових цін.")
        still = unsupported_numbers(answer.reply, sources)
        if still:
            raise LLMError(f"invented numbers: {sorted(still)}")
        return answer, {"retries": 1, "rejected_numbers": sorted(invented)}

    def decide(self, state: StateVector, voice_features: dict, transcript: str,
               history: list[Decision], context: str = "") -> Decision:
        started = time.perf_counter()
        try:
            answer, check = self.generate_checked(
                describe_situation(state, voice_features, transcript, history, None, None, context),
                f"{context}\n{transcript}")
        except LLMError as exc:
            decision = self.fallback.decide(state, voice_features, transcript, history)
            decision.source = f"{self.name}_fallback"
            decision.details.update(error=str(exc), llm_ms=round((time.perf_counter() - started) * 1000))
            return decision
        return Decision(reply=answer.reply, tone=answer.tone, confidence=state.confidence, source=self.name,
                        details={"llm_ms": round((time.perf_counter() - started) * 1000), **check})
