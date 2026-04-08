use worker::*;

mod analysis;
mod broker;
mod config;
mod db;
mod handlers;
mod notifications;
mod risk;
mod types;

#[event(fetch)]
async fn fetch(mut req: Request, _env: Env, _ctx: Context) -> Result<Response> {
    // Discord ping/pong for bot verification
    if req.method() == Method::Post {
        if let Ok(body) = req.text().await {
            if body.contains("\"type\":1") {
                return Response::from_json(&serde_json::json!({ "type": 1 }));
            }
        }
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
