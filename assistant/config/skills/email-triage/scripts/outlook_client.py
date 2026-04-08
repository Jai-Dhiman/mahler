import imaplib
import email
import email.header
import email.utils
import email.message
import re
import ssl
from datetime import datetime, timezone
from typing import Optional

from email_types import EmailMessage


def _decode_header_value(raw_value: Optional[str]) -> str:
    if not raw_value:
        return ""
    parts = email.header.decode_header(raw_value)
    decoded_parts = []
    for fragment, charset in parts:
        if isinstance(fragment, bytes):
            if charset:
                decoded_parts.append(fragment.decode(charset, errors="replace"))
            else:
                decoded_parts.append(fragment.decode("utf-8", errors="replace"))
        else:
            decoded_parts.append(fragment)
    return "".join(decoded_parts)


def _strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


def _extract_body_preview(msg: email.message.Message) -> str:
    plain_body: Optional[str] = None
    html_body: Optional[str] = None

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain" and plain_body is None:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        plain_body = payload.decode(charset, errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        plain_body = payload.decode("latin-1", errors="replace")
            elif content_type == "text/html" and html_body is None:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        html_body = payload.decode(charset, errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        html_body = payload.decode("latin-1", errors="replace")
    else:
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                text = payload.decode(charset, errors="replace")
            except (LookupError, UnicodeDecodeError):
                text = payload.decode("latin-1", errors="replace")
            if content_type == "text/plain":
                plain_body = text
            elif content_type == "text/html":
                html_body = text

    if plain_body is not None:
        return plain_body.strip()[:500]
    if html_body is not None:
        return _strip_html(html_body).strip()[:500]
    return ""


def _parse_received_at(date_str: Optional[str]) -> str:
    if date_str:
        try:
            dt = email.utils.parsedate_to_datetime(date_str)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            pass
    return datetime.now(tz=timezone.utc).isoformat()


def _fetch_from_folder(
    imap: imaplib.IMAP4_SSL,
    folder_name: str,
    is_junk: bool,
) -> list[EmailMessage]:
    try:
        status, _ = imap.select(folder_name, readonly=False)
    except imaplib.IMAP4.error:
        return []

    if status != "OK":
        return []

    status, data = imap.search(None, "UNSEEN")
    if status != "OK" or not data or not data[0]:
        return []

    uid_list = data[0].split()
    if not uid_list:
        return []

    results: list[EmailMessage] = []

    for uid in uid_list:
        try:
            status, msg_data = imap.fetch(uid, "(RFC822)")
            if status != "OK" or not msg_data:
                continue

            raw_email: Optional[bytes] = None
            for part in msg_data:
                if isinstance(part, tuple):
                    raw_email = part[1]
                    break

            if raw_email is None:
                continue

            msg = email.message_from_bytes(raw_email)

            imap.store(uid, "+FLAGS", "\\Seen")

            raw_message_id = msg.get("Message-ID", "").strip()
            if raw_message_id:
                message_id = raw_message_id.strip("<>")
            else:
                message_id = f"outlook:{uid.decode()}"

            raw_from = _decode_header_value(msg.get("From", ""))
            name, addr = email.utils.parseaddr(raw_from)
            if addr:
                from_addr = f"{name} <{addr}>" if name else addr
            else:
                from_addr = raw_from

            subject = _decode_header_value(msg.get("Subject", ""))
            received_at = _parse_received_at(msg.get("Date"))
            body_preview = _extract_body_preview(msg)

            headers: dict = {k.lower(): v for k, v in msg.items()}

            result: EmailMessage = {
                "message_id": message_id,
                "source": "outlook",
                "from_addr": from_addr,
                "subject": subject,
                "received_at": received_at,
                "body_preview": body_preview,
                "is_junk_rescue": is_junk,
                "headers": headers,
            }
            results.append(result)

        except Exception:
            continue

    return results


def fetch_unread_emails(
    host: str, email_addr: str, app_password: str
) -> list[EmailMessage]:
    """Fetch unread emails from Outlook INBOX and Junk folders via IMAP SSL.
    Marks fetched messages as seen. Raises on connection/auth failure."""
    imap = imaplib.IMAP4_SSL(host, 993)
    try:
        try:
            imap.login(email_addr, app_password)
        except imaplib.IMAP4.error as exc:
            raise RuntimeError(f"IMAP login failed for {email_addr}: {exc}") from exc

        results: list[EmailMessage] = []
        results.extend(_fetch_from_folder(imap, "INBOX", is_junk=False))
        results.extend(_fetch_from_folder(imap, "Junk", is_junk=True))
        return results
    finally:
        try:
            imap.logout()
        except Exception:
            pass
