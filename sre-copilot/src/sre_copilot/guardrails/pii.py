import re
from typing import Iterable

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
KEY_RE = re.compile(r"(sk-|pk_|rk_)[A-Za-z0-9]{16,}")
IP_RE = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
PHONE_RE = re.compile(r"\+?\d[\d\s\-]{6,}\d")


def find_pii(text: str) -> list[str]:
    matches: list[str] = []
    matches.extend(EMAIL_RE.findall(text))
    matches.extend(KEY_RE.findall(text))
    matches.extend(IP_RE.findall(text))
    matches.extend(PHONE_RE.findall(text))
    return list(dict.fromkeys(matches))  # de-duplicate while preserving order


def mask_matches(text: str, matches: Iterable[str]) -> str:
    masked = text
    for match in matches:
        masked = masked.replace(match, "[redacted]")
    return masked
