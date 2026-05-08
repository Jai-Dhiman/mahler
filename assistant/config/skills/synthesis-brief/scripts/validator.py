_THIN_RECENT_MIN = 3
_THIN_CONTEXT_MIN = 5


def _is_thin(bundle) -> bool:
    recent_n = len(bundle.recent_items)
    total_n = recent_n + len(bundle.context_items)
    return recent_n < _THIN_RECENT_MIN and total_n < _THIN_CONTEXT_MIN


def validate(brief: dict, bundle) -> tuple[bool, str]:
    if _is_thin(bundle):
        return (False, "thin_context")
    return (True, "")
