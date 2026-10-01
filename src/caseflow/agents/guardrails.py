"""Deterministic guardrails (the cheap, fast layer of defense in depth).

Layers of protection in CaseFlow, from outermost to innermost:

1. **Input screening** (this module) - size limits, card-number masking and
   prompt-injection heuristics (microseconds, catches the obvious attacks), plus an
   optional ML classifier (Llama Prompt Guard 2) for paraphrased attacks.
2. **LLM triage** - the triage model is told user text is data and flags
   ``injection_suspected``; suspicious turns get no tool-using specialists.
3. **Architecture** - tools are scoped by JWT to the signed-in customer, and
   money moves only through server-enforced policy + signed human approvals.
   This is the layer that makes a *successful* injection harmless.
4. **Output review** - deterministic leak checks (this module) + an LLM QA
   reviewer that checks the draft against the specialists' findings.

Heuristics are intentionally conservative: false positives on normal support
messages are expensive (a customer gets refused), so only high-confidence
patterns block; weaker signals are passed to triage as a flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 4000

_STRONG_INJECTION = [
    r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier|your)\s+(?:instructions|rules|prompts?|guidelines)",
    r"\bdisregard\s+(?:all\s+|any\s+)?(?:previous|prior|your|the)\s+(?:instructions|rules|guidelines)",
    r"\b(?:reveal|print|show|repeat|output)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions)",
    r"\byou\s+are\s+now\s+(?:in\s+)?(?:developer|dan|jailbreak|admin|god)\s*mode\b",
    r"\b(?:developer|jailbreak)\s+mode\s+(?:enabled|activated|on)\b",
    r"<\s*/?\s*(?:system|assistant|tool)\s*>",  # fake role tags
    r"\bapproval[_\s-]?code\s*[:=]",  # attempts to smuggle a refund approval
]
_WEAK_INJECTION = [
    r"\bact\s+as\s+(?:an?\s+)?(?:admin|administrator|supervisor|developer|another\s+customer)",
    r"\bpretend\s+(?:to\s+be|you\s+are)\b",
    r"\b(?:customer|user)[_\s-]?id\s*[:=]\s*\w+",  # trying to pick whose data to read
    r"\bsystem\s+prompt\b",
    r"\bbypass\b.{0,40}\b(?:limit|approval|policy|verification)\b",
]
_STRONG_RE = [re.compile(p, re.IGNORECASE) for p in _STRONG_INJECTION]
_WEAK_RE = [re.compile(p, re.IGNORECASE) for p in _WEAK_INJECTION]

_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_INTERNAL_TERMS = re.compile(
    r"\b(?:issue_refund|check_return_eligibility|create_return|track_shipment|list_orders|get_order|"
    # cust_NNN: internal account ids. Live red team A08 ("my customer id is cust_001") returned the
    # *right* data (the signed-in customer's) but echoed the claimed id in the reply - misleading.
    r"approval_code|system prompt|specialist_findings|MCP server|KB-\d{3}|cust_\d+)\b",
    re.IGNORECASE,
)

REFUSAL_MESSAGE = (
    "I can only help with Voltwise orders, returns, products and policies, and I can't change how I "
    "operate. If you have a question about an order or a product, I'm happy to help!"
)


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mask_card_numbers(text: str) -> str:
    """Replace Luhn-valid 13-19 digit card numbers with ``[card ending 1234]`` (KB-012 promise)."""

    def repl(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return f"[card ending {digits[-4:]}]"
        return m.group(0)

    return _CARD_CANDIDATE.sub(repl, text)


@dataclass(slots=True)
class InputVerdict:
    blocked: bool
    reason: str | None = None
    signals: list[str] = field(default_factory=list)
    message: str | None = None


def screen_input(text: str) -> InputVerdict:
    stripped = text.strip()
    if not stripped:
        return InputVerdict(
            blocked=True, reason="empty", message="It looks like your message was empty - how can I help?"
        )
    if len(stripped) > MAX_INPUT_CHARS:
        return InputVerdict(
            blocked=True,
            reason="too_long",
            message="That message is a bit long for me. Could you summarize your question in a few sentences?",
        )
    strong = [p.pattern for p in _STRONG_RE if p.search(stripped)]
    weak = [p.pattern for p in _WEAK_RE if p.search(stripped)]
    if strong or len(weak) >= 2:
        return InputVerdict(blocked=True, reason="prompt_injection", signals=strong + weak, message=REFUSAL_MESSAGE)
    return InputVerdict(blocked=False, signals=weak)


class PromptGuardClassifier:
    """ML layer: Meta's Llama Prompt Guard 2 (86M), served free on Groq.

    A small classifier fine-tuned to detect injections/jailbreaks - it catches
    paraphrased attacks that regexes miss, in ~100 ms and without spending LLM tokens.
    It **fails open** (returns 0.0) on timeouts/errors: availability of support must
    not depend on a classifier, and the architectural layer still protects data and money.
    """

    def __init__(self, model: str, api_key: str, *, timeout_s: float = 3.0) -> None:
        self.model = model
        self._api_key = api_key
        self._timeout = timeout_s

    async def score(self, text: str) -> float:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self.model, "messages": [{"role": "user", "content": text[:2000]}]},
                )
                r.raise_for_status()
                return float(r.json()["choices"][0]["message"]["content"].strip())
        except Exception:
            return 0.0


def output_issues(draft: str) -> list[str]:
    """Deterministic checks on the reply before it reaches the customer."""
    issues: list[str] = []
    if _JWT.search(draft):
        issues.append("The draft contains what looks like an access token. Remove it.")
    if mask_card_numbers(draft) != draft:
        issues.append("The draft contains a full card number. Never include card numbers.")
    if m := _INTERNAL_TERMS.search(draft):
        issues.append(
            f"The draft mentions internal system details ('{m.group(0)}'). Describe outcomes in plain language."
        )
    if re.search(r"\b(?:password|cvv|one[- ]time code)\b", draft, re.IGNORECASE) and re.search(
        r"\b(?:send|share|provide|tell|give)\b", draft, re.IGNORECASE
    ):
        issues.append("The draft appears to ask for a password/CVV/one-time code. Never request credentials.")
    return issues
