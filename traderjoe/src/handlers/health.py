"""Health check endpoint."""

from datetime import datetime

from workers import Response


async def handle_health(request, env):
    """Handle health check requests.

    Returns basic system status and connectivity info.
    """
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": env.ENVIRONMENT,
        "checks": {},
    }

    # Check D1 connectivity
    try:
        result = await env.MAHLER_DB.prepare("SELECT 1 as test").first()
        status["checks"]["d1"] = "ok" if result else "error"
    except Exception as e:
        status["checks"]["d1"] = f"error: {str(e)}"
        status["status"] = "degraded"

    # Check KV connectivity
    try:
        await env.MAHLER_KV.get("health_check_test")
        status["checks"]["kv"] = "ok"
    except Exception as e:
        status["checks"]["kv"] = f"error: {str(e)}"
        status["status"] = "degraded"

    # Check Alpaca API connectivity
    try:
        from core.broker.alpaca import AlpacaClient
        alpaca = AlpacaClient(
            api_key=env.ALPACA_API_KEY,
            secret_key=env.ALPACA_SECRET_KEY,
            paper=(env.ENVIRONMENT == "paper"),
        )
        account = await alpaca.get_account()
        status["checks"]["alpaca"] = "ok"
        status["alpaca_account"] = {
            "equity": account.equity,
            "cash": account.cash,
            "status": account.status,
        }
    except Exception as e:
        status["checks"]["alpaca"] = f"error: {str(e)}"
        status["status"] = "degraded"

    return Response(
        __import__("json").dumps(status),
        headers={"Content-Type": "application/json"},
    )
