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
async fn fetch(_req: Request, _env: Env, _ctx: Context) -> Result<Response> {
    Response::ok("trader-joe")
}

#[event(scheduled)]
async fn scheduled(event: ScheduledEvent, env: Env, _ctx: ScheduleContext) {
    let cron = event.cron();
    let result = match cron.as_str() {
        "0 14 * * MON-FRI" => handlers::morning_scan::run(&env).await,
        "*/5 14-20 * * MON-FRI" => handlers::position_monitor::run(&env).await,
        "15 20 * * MON-FRI" => handlers::eod_summary::run(&env).await,
        unknown => {
            console_error!("Unknown cron: {}", unknown);
            Ok(())
        }
    };

    if let Err(e) = result {
        console_error!("Handler error for {}: {:?}", cron, e);
    }
}
