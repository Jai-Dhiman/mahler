import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from email_types import EmailMessage

_NOISE_FROM_PATTERNS = re.compile(
    r"(noreply|no-reply|no_reply|donotreply|mailer-daemon|postmaster)",
    re.IGNORECASE,
)

_ADDR_EXTRACT = re.compile(r"<([^>]+)>|(\S+@\S+)")

_JOB_BOARD_DOMAINS = frozenset([
    "linkedin.com",
    "indeed.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "monster.com",
    "dice.com",
    "careerbuilder.com",
    "lever.co",
    "greenhouse.io",
    "workday.com",
    "smartrecruiters.com",
])

_JOB_ALERT_SUBJECTS = re.compile(
    r"(job alert|new jobs for you|jobs matching|recommended jobs|jobs you might like)",
    re.IGNORECASE,
)

_BULK_PRECEDENCE = frozenset(["bulk", "list", "junk"])


def _extract_domain(from_addr: str) -> str:
    match = _ADDR_EXTRACT.search(from_addr)
    if not match:
        return ""
    raw = match.group(1) or match.group(2)
    if not raw:
        return ""
    at = raw.rfind("@")
    if at == -1:
        return ""
    return raw[at + 1:].lower().strip()


def is_noise(email: "EmailMessage") -> bool:
    headers = email["headers"]

    if "list-unsubscribe" in headers:
        return True

    precedence = headers.get("precedence", "").lower().strip()
    if precedence in _BULK_PRECEDENCE:
        return True

    auto_submitted = headers.get("auto-submitted", "no").lower()
    if auto_submitted != "no":
        return True

    if _NOISE_FROM_PATTERNS.search(email["from_addr"]):
        return True

    domain = _extract_domain(email["from_addr"])
    if domain in _JOB_BOARD_DOMAINS:
        return True

    if _JOB_ALERT_SUBJECTS.search(email["subject"]):
        return True

    return False
