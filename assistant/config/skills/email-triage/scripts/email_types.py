from typing import TypedDict, Literal


class EmailMessage(TypedDict):
    message_id: str        # globally unique ID (gmail id or "outlook:{uid}:{folder}")
    source: Literal["gmail", "outlook"]
    from_addr: str         # "Name <email@example.com>" or just "email@example.com"
    subject: str
    received_at: str       # ISO8601 UTC
    body_preview: str      # first 500 chars plain text, stripped of HTML
    is_junk_rescue: bool   # True if fetched from Junk/Spam folder
    headers: dict          # raw headers dict, lowercase keys, for pre-filter use
