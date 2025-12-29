import re

PATTERNS = [
    re.compile(pat, re.IGNORECASE)
    for pat in [
        r"ignore\s+previous",
        r"disregard\s+above",
        r"system\s+prompt",
        r"you\s+are\s+now\s+.*?model",
        r"drop\s+all\s+rules",
        r"delete\s+your\s+instructions",
        r"runtime\s+reconfigure",
        r"sudo\s+rm",
    ]
]


def detect_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in PATTERNS)
