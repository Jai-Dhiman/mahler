"""Main entry point for Cloudflare Workers.

Routes incoming requests (HTTP and cron) to appropriate handlers.
Uses lazy imports due to Cloudflare Python Workers module loading quirks.
"""

from datetime import datetime
from workers import Response


async def on_fetch(request, env):
    """Handle HTTP requests."""
    url = request.url
    method = request.method

    try:
        # Health check
        if "/health" in url:
            from workers.health import handle_health
            return await handle_health(request, env)

        # Discord webhook
        if "/discord" in url and method == "POST":
            from workers.discord_webhook import handle_discord_webhook
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
    """Handle cron triggers."""
    cron = event.cron

    try:
        print(f"Cron triggered: {cron} at {datetime.now().isoformat()}")

        # Route based on cron pattern
        if cron == "35 14 * * 1-5":
            from workers.morning_scan import handle_morning_scan
            await handle_morning_scan(env)
        elif cron == "0 17 * * 1-5":
            from workers.midday_check import handle_midday_check
            await handle_midday_check(env)
        elif cron == "30 20 * * 1-5":
            from workers.afternoon_scan import handle_afternoon_scan
            await handle_afternoon_scan(env)
        elif cron == "15 21 * * 1-5":
            from workers.eod_summary import handle_eod_summary
            await handle_eod_summary(env)
        elif "*/5" in cron:
            from workers.position_monitor import handle_position_monitor
            await handle_position_monitor(env)
        else:
            print(f"Unknown cron pattern: {cron}")

    except Exception as e:
        print(f"Error in scheduled handler: {e}")
        raise


# Cloudflare Workers entry points
def fetch(request, env):
    """Fetch handler for HTTP requests."""
    return on_fetch(request, env)


def scheduled(event, env, ctx):
    """Scheduled handler for cron triggers."""
    ctx.wait_until(on_scheduled(event, env, ctx))
