"""PII scrubbing and basic content moderation for forum text."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import re


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("ssn", re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")),
    ("api_key", re.compile(
        r"\b(?:api[_ -]?key|secret|access[_ -]?token|password)\s*[:=]\s*"
        r"[\"']?[A-Za-z0-9_./+=:-]{8,}[\"']?",
        re.I,
    )),
    ("api_key", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}\b", re.I)),
    ("api_key", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b")),
    ("ip_address", re.compile(
        r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])"
    )),
    ("phone", re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
)
_TOXIC_TERMS = frozenset({
    "asshole", "bastard", "bitch", "bullshit", "fuck", "fucked", "fucking",
    "idiot", "moron", "shit", "stupid",
})
_TOXIC_RE = re.compile(r"\b(?:" + "|".join(sorted(_TOXIC_TERMS)) + r")\b", re.I)


@dataclass(frozen=True)
class ModerationResult:
    text: str
    findings: tuple[str, ...]
    blocked: bool


def moderate_text(text: str) -> ModerationResult:
    """Redact recognized personal/secrets and block profanity-bearing text."""
    sanitized = text
    findings: list[str] = []
    for category, pattern in _PATTERNS:
        if category == "ip_address":
            def replace_ip(match: re.Match[str]) -> str:
                try:
                    ipaddress.ip_address(match.group())
                except ValueError:
                    return match.group()
                findings.append(category)
                return f"[REDACTED {category.upper()}]"

            sanitized = pattern.sub(replace_ip, sanitized)
            continue
        if pattern.search(sanitized):
            findings.append(category)
            sanitized = pattern.sub(f"[REDACTED {category.upper()}]", sanitized)

    toxic_matches = _TOXIC_RE.findall(sanitized)
    if toxic_matches:
        findings.append("toxic_language")
    return ModerationResult(
        text=sanitized,
        findings=tuple(dict.fromkeys(findings)),
        blocked=bool(toxic_matches),
    )
