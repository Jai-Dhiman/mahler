_THIN_RECENT_MIN = 3
_THIN_CONTEXT_MIN = 5
_SECTION_MAX = 600
_TOTAL_MAX = 2000


def _is_thin(bundle) -> bool:
    recent_n = len(bundle.recent_items)
    total_n = recent_n + len(bundle.context_items)
    return recent_n < _THIN_RECENT_MIN and total_n < _THIN_CONTEXT_MIN


def _resolve_id(cit: dict, identifiers: set) -> str:
    """Return the matching identifier string, or '' if none matches.

    Accepts both the full 'source:id' form (e.g. 'git:422') and the split
    form the LLM sometimes produces ('source': 'git', 'id': '422').
    """
    cit_id = cit.get("id", "")
    if cit_id in identifiers:
        return cit_id
    composed = f"{cit.get('source', '')}:{cit_id}"
    if composed in identifiers:
        return composed
    return ""


def _qualifying_connections(brief: dict, identifiers: set) -> int:
    n = 0
    for c in brief.get("connections", []):
        cites = c.get("citations", []) or []
        valid = {r for cit in cites if (r := _resolve_id(cit, identifiers))}
        if len(valid) >= 2:
            n += 1
    return n


def _check_length(brief: dict) -> bool:
    pattern = brief.get("pattern") or ""
    question = brief.get("question") or ""
    summaries = [c.get("summary", "") for c in brief.get("connections", [])]
    if len(pattern) > _SECTION_MAX or len(question) > _SECTION_MAX:
        return False
    if any(len(s) > _SECTION_MAX for s in summaries):
        return False
    total = len(pattern) + len(question) + sum(len(s) for s in summaries)
    return total <= _TOTAL_MAX


def validate(brief: dict, bundle) -> tuple[bool, str]:
    if _is_thin(bundle):
        return (False, "thin_context")
    if _qualifying_connections(brief, bundle.identifiers) < 2:
        return (False, "insufficient_citations")
    if not _check_length(brief):
        return (False, "length_exceeded")
    return (True, "")
