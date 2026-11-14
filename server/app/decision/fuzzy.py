import json
import random
import re
from pathlib import Path

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from .base import STATE_KEYS, STRATEGIES, Decision, DecisionModule, StateVector

UNIVERSE = np.linspace(0, 1, 101)
TONE_LEVELS = ("soft", "neutral", "firm")
RULE_PATTERN = re.compile(r"^IF (?P<if>.+) THEN (?P<then>.+)$")
CLAUSE_PATTERN = re.compile(r"^(\w+) IS (\w+)$")


def build_variables() -> tuple[dict, dict]:
    inputs = {}
    for name in STATE_KEYS.values():
        var = ctrl.Antecedent(UNIVERSE, name)
        var["low"] = fuzz.trapmf(UNIVERSE, [0, 0, 0.1, 0.25])
        var["medium"] = fuzz.trimf(UNIVERSE, [0.15, 0.3, 0.5])
        var["high"] = fuzz.trapmf(UNIVERSE, [0.35, 0.6, 1, 1])
        inputs[name] = var
    confidence = ctrl.Antecedent(UNIVERSE, "confidence")
    confidence["low"] = fuzz.trimf(UNIVERSE, [0, 0, 0.35])
    confidence["high"] = fuzz.trapmf(UNIVERSE, [0.2, 0.6, 1, 1])
    confidence["any"] = np.full_like(UNIVERSE, 0.2)
    inputs["confidence"] = confidence

    outputs = {}
    for name in STRATEGIES:
        var = ctrl.Consequent(UNIVERSE, name)
        var["weak"] = fuzz.trimf(UNIVERSE, [0, 0, 0.5])
        var["strong"] = fuzz.trimf(UNIVERSE, [0.5, 1, 1])
        outputs[name] = var
    tone = ctrl.Consequent(UNIVERSE, "tone")
    tone["soft"] = fuzz.trimf(UNIVERSE, [0, 0, 0.5])
    tone["neutral"] = fuzz.trimf(UNIVERSE, [0.25, 0.5, 0.75])
    tone["firm"] = fuzz.trimf(UNIVERSE, [0.5, 1, 1])
    outputs["tone"] = tone
    return inputs, outputs


def _clause(text: str, variables: dict, line: str):
    match = CLAUSE_PATTERN.match(text.strip())
    if not match or match[1] not in variables or match[2] not in variables[match[1]].terms:
        raise ValueError(f"bad clause '{text.strip()}' in rule: {line}")
    return variables[match[1]][match[2]]


def parse_rules(text: str, inputs: dict, outputs: dict) -> list[ctrl.Rule]:
    rules = []
    for line in (raw.strip() for raw in text.splitlines()):
        if not line or line.startswith("#"):
            continue
        match = RULE_PATTERN.match(line)
        if not match:
            raise ValueError(f"cannot parse rule: {line}")
        condition = match["if"]
        if " AND " in condition and " OR " in condition:
            raise ValueError(f"mixing AND and OR is not supported: {line}")
        joiner = " OR " if " OR " in condition else " AND "
        terms = [_clause(part, inputs, line) for part in condition.split(joiner)]
        antecedent = terms[0]
        for term in terms[1:]:
            antecedent = antecedent | term if joiner == " OR " else antecedent & term
        consequents = [_clause(part, outputs, line) for part in match["then"].split(",")]
        rules.append(ctrl.Rule(antecedent, consequents))
    return rules


class ReplyBank:
    def __init__(self, path: Path, seed: int | None = None):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.replies: dict[str, list[str]] = data["replies"]
        self.tones: dict[str, str] = data["tones"]
        self.strategy_tones: dict[str, str] = data.get("strategy_tones", {})
        self._random = random.Random(seed)

    def pick(self, strategy: str, history: list[Decision], avoid_last: int = 3) -> str:
        recent = {d.reply for d in history[-avoid_last:]}
        options = [r for r in self.replies[strategy] if r not in recent] or self.replies[strategy]
        return self._random.choice(options)

    def tone(self, strategy: str, level: str) -> str:
        return self.strategy_tones.get(strategy, self.tones[level])


class FuzzyDecision(DecisionModule):
    name = "fuzzy"

    def __init__(self, rules_path: Path, replies_path: Path, seed: int | None = None):
        inputs, outputs = build_variables()
        self.rules = parse_rules(rules_path.read_text(encoding="utf-8"), inputs, outputs)
        self.system = ctrl.ControlSystem(self.rules)
        self.replies = ReplyBank(replies_path, seed)

    def evaluate(self, state: StateVector) -> tuple[str, str, dict[str, float], float]:
        simulation = ctrl.ControlSystemSimulation(self.system)
        for uk_name, key in STATE_KEYS.items():
            simulation.input[key] = float(np.clip(state.degrees.get(uk_name, 0.0), 0, 1))
        simulation.input["confidence"] = float(np.clip(state.confidence, 0, 1))
        simulation.compute()
        activations = {s: round(float(simulation.output[s]), 4) for s in STRATEGIES}
        tone_value = float(simulation.output["tone"])
        strategy = max(activations, key=activations.get)
        level = "soft" if tone_value < 0.42 else "firm" if tone_value > 0.58 else "neutral"
        return strategy, level, activations, round(tone_value, 4)

    def decide(self, state: StateVector, voice_features: dict, transcript: str,
               history: list[Decision], context: str = "") -> Decision:
        strategy, level, activations, tone_value = self.evaluate(state)
        return Decision(
            reply=self.replies.pick(strategy, history),
            tone=self.replies.tone(strategy, level),
            confidence=state.confidence,
            source=self.name,
            strategy=strategy,
            details={"activations": activations, "tone_value": tone_value, "tone_level": level},
        )
