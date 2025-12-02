"""Main entry point for Cloudflare Workers.

Routes incoming requests (HTTP and cron) to appropriate handlers.
"""

from datetime import datetime

from src.workers.health import handle_health
from src.workers.morning_scan import handle_morning_scan
from src.workers.midday_check import handle_midday_check
from src.workers.afternoon_scan import handle_afternoon_scan
from src.workers.eod_summary import handle_eod_summary
from src.workers.position_monitor import handle_position_monitor
from src.workers.discord_webhook import handle_discord_webhook


async def on_fetch(request, env):
    """Handle HTTP requests."""
    url = request.url
    method = request.method

    try:
        # Health check
        if "/health" in url:
            return await handle_health(request, env)

        # Discord webhook
        if "/discord" in url and method == "POST":
            return await handle_discord_webhook(request, env)

        # Default response
        return Response(
            '{"status": "ok", "service": "mahler"}',
            headers={"Content-Type": "application/json"},
        )

    except Exception as e:
        print(f"Error handling request: {e}")
        return Response(
            f'{{"error": "{str(e)}"}}',
            status=500,
            headers={"Content-Type": "application/json"},
        )


async def on_scheduled(event, env, ctx):
    """Handle cron triggers.

    Cron schedule (from wrangler.toml):
    - "35 14 * * 1-5"     -> morning_scan (9:35 AM ET)
    - "0 17 * * 1-5"      -> midday_check (12:00 PM ET)
    - "30 20 * * 1-5"     -> afternoon_scan (3:30 PM ET)
    - "15 21 * * 1-5"     -> eod_summary (4:15 PM ET)
    - "*/5 14-21 * * 1-5" -> position_monitor (every 5 min)
    """
    cron = event.cron

    try:
        print(f"Cron triggered: {cron} at {datetime.now().isoformat()}")

        # Route based on cron pattern
        if cron == "35 14 * * 1-5":
            await handle_morning_scan(env)
        elif cron == "0 17 * * 1-5":
            await handle_midday_check(env)
        elif cron == "30 20 * * 1-5":
            await handle_afternoon_scan(env)
        elif cron == "15 21 * * 1-5":
            await handle_eod_summary(env)
        elif "*/5" in cron:
            await handle_position_monitor(env)
        else:
            print(f"Unknown cron pattern: {cron}")

    except Exception as e:
        print(f"Error in scheduled handler: {e}")
        # Re-raise to ensure error is logged
        raise


# Cloudflare Workers entry points
def fetch(request, env):
    """Fetch handler for HTTP requests."""
    return on_fetch(request, env)


def scheduled(event, env, ctx):
    """Scheduled handler for cron triggers."""
    ctx.wait_until(on_scheduled(event, env, ctx))
