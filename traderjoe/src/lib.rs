use worker::*;

mod analysis;
mod broker;
mod config;
mod db;
mod handlers;
mod notifications;
mod risk;
mod types;

/// Verify the Authorization header matches the TRIGGER_SECRET env var.
///
/// Returns Err(Response) with 401 if auth fails so callers can propagate with `?`.
fn check_auth(req: &Request, env: &Env) -> std::result::Result<(), Response> {
    let secret = env.var("TRIGGER_SECRET")
        .map(|v| v.to_string())
        .unwrap_or_default();

    if secret.is_empty() {
        return Err(Response::error("TRIGGER_SECRET not configured", 500).unwrap());
    }

    let expected = format!("Bearer {}", secret);
    let auth = req.headers().get("Authorization").ok().flatten().unwrap_or_default();

    if auth != expected {
        return Err(Response::error("Unauthorized", 401).unwrap());
    }

    Ok(())
}

#[event(fetch)]
async fn fetch(mut req: Request, env: Env, _ctx: Context) -> Result<Response> {
    // Discord ping/pong for bot verification
    if req.method() == Method::Post {
        if let Ok(body) = req.text().await {
            if body.contains("\"type\":1") {
                return Response::from_json(&serde_json::json!({ "type": 1 }));
            }

            // Restore the consumed body for downstream use by routing on path
            let path = req.path();

            if path == "/trigger/morning-scan" {
                if let Err(r) = check_auth(&req, &env) { return Ok(r); }
                handlers::morning_scan::run(&env).await?;
                return Response::from_json(&serde_json::json!({ "ok": true, "handler": "morning_scan" }));
            }

            if path == "/trigger/position-monitor" {
                if let Err(r) = check_auth(&req, &env) { return Ok(r); }
                handlers::position_monitor::run(&env).await?;
                return Response::from_json(&serde_json::json!({ "ok": true, "handler": "position_monitor" }));
            }

            if path == "/trigger/eod-summary" {
                if let Err(r) = check_auth(&req, &env) { return Ok(r); }
                handlers::eod_summary::run(&env).await?;
                return Response::from_json(&serde_json::json!({ "ok": true, "handler": "eod_summary" }));
            }

            if path == "/circuit-breaker/reset" {
                if let Err(r) = check_auth(&req, &env) { return Ok(r); }
                let kv = db::kv::KvClient::new(env.kv("KV")?);
                kv.reset_circuit_breaker().await?;
                return Response::from_json(&serde_json::json!({ "ok": true, "reset": true }));
            }
        }
    }

    let path = req.path();

    if path == "/health" {
        use chrono::Utc;
        let db = db::d1::D1Client::new(env.d1("DB")?);
        let kv = db::kv::KvClient::new(env.kv("KV")?);

        let trade_count = db.get_open_trades().await.map(|t| t.len() as i64).unwrap_or(-1);
        let cb_state = kv.get_circuit_breaker().await;
        let cb_halted = cb_state.as_ref().map(|s| s.halted).unwrap_or(false);
        let kv_ok = cb_state.is_ok();

        // Check each secret/var by presence only — never call .to_string() on Var,
        // which panics in worker-rs 0.4 when the JS value can't be coerced to a string.
        let alpaca_key_ok = env.var("ALPACA_API_KEY").is_ok();
        let alpaca_secret_ok = env.var("ALPACA_SECRET_KEY").is_ok();
        let discord_token_ok = env.var("DISCORD_BOT_TOKEN").is_ok();
        let discord_channel_ok = env.var("DISCORD_CHANNEL_ID").is_ok();
        let trigger_secret_ok = env.var("TRIGGER_SECRET").is_ok();

        return Response::from_json(&serde_json::json!({
            "ok": true,
            "db_reachable": trade_count >= 0,
            "open_trades": trade_count,
            "kv_reachable": kv_ok,
            "circuit_breaker_halted": cb_halted,
            "env_vars": {
                "ALPACA_API_KEY": alpaca_key_ok,
                "ALPACA_SECRET_KEY": alpaca_secret_ok,
                "DISCORD_BOT_TOKEN": discord_token_ok,
                "DISCORD_CHANNEL_ID": discord_channel_ok,
                "TRIGGER_SECRET": trigger_secret_ok,
            }
        }));
    }

    if path == "/circuit-breaker" {
        if let Err(r) = check_auth(&req, &env) { return Ok(r); }
        let kv = db::kv::KvClient::new(env.kv("KV")?);
        let state = kv.get_circuit_breaker().await?;
        return Response::from_json(&serde_json::json!({
            "halted": state.halted,
            "reason": state.reason,
            "triggered_at": state.triggered_at,
        }));
    }

    Response::ok("trader-joe")
}

#[event(scheduled)]
async fn scheduled(event: ScheduledEvent, env: Env, _ctx: ScheduleContext) {
    let cron = event.cron();
    console_log!("Scheduled event: {}", cron);

    let result = match cron.as_str() {
        "0 14 * * MON-FRI" => handlers::morning_scan::run(&env).await,
        "*/5 14-20 * * MON-FRI" => handlers::position_monitor::run(&env).await,
        "15 20 * * MON-FRI" => handlers::eod_summary::run(&env).await,
        unknown => {
            console_error!("Unknown cron expression: {}", unknown);
            Ok(())
        }
    };

    if let Err(e) = result {
        console_error!("Handler error for cron {}: {:?}", cron, e);
        if let (Ok(token), Ok(channel)) = (
            env.var("DISCORD_BOT_TOKEN"),
            env.var("DISCORD_CHANNEL_ID"),
        ) {
            let discord = notifications::discord::DiscordClient::new(token.to_string(), channel.to_string());
            discord
                .send_error(
                    &format!("Cron Handler Error: {}", cron),
                    &format!("{:?}", e),
                )
                .await
                .ok();
        }
    }
}
