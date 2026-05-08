_THIN_RECENT_MIN = 3
_THIN_CONTEXT_MIN = 5
_SECTION_MAX = 600
_TOTAL_MAX = 2000


def _is_thin(bundle) -> bool:
    recent_n = len(bundle.recent_items)
    total_n = recent_n + len(bundle.context_items)
    return recent_n < _THIN_RECENT_MIN and total_n < _THIN_CONTEXT_MIN


def _qualifying_connections(brief: dict, identifiers: set) -> int:
    n = 0
    for c in brief.get("connections", []):
        cites = c.get("citations", []) or []
        valid = {cit.get("id", "") for cit in cites if cit.get("id") in identifiers}
        if len(valid) >= 2:
            n += 1
    return n


def _check_length(brief: dict) -> bool:
    pattern = brief.get("pattern") or ""
    question = brief.get("question") or ""
    connections_text = "\n".join(c.get("summary", "") for c in brief.get("connections", []))
    sections = [pattern, question, connections_text]
    if any(len(s) > _SECTION_MAX for s in sections):
        return False
    if sum(len(s) for s in sections) > _TOTAL_MAX:
        return False
    return True


def validate(brief: dict, bundle) -> tuple[bool, str]:
    if _is_thin(bundle):
        return (False, "thin_context")
    if _qualifying_connections(brief, bundle.identifiers) < 2:
        return (False, "insufficient_citations")
    if not _check_length(brief):
        return (False, "length_exceeded")
    return (True, "")
