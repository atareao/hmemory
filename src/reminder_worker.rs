use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use tokio::time::{Duration, interval};
use tracing::{error, info, warn};

use crate::api::AppState;

pub fn start(state: AppState) {
    tokio::spawn(async move {
        let mut tick = interval(Duration::from_secs(30));
        let log_path = std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".hermes").join("reminders.log"))
            .or_else(|| Some(PathBuf::from("/tmp/hmemory_reminders.log")));

        loop {
            tick.tick().await;
            let records = match state.store.get_due_reminders(50).await {
                Ok(r) => r,
                Err(e) => {
                    warn!("reminder poll failed: {e}");
                    continue;
                }
            };

            for rec in &records {
                let msg = format!(
                    "[REMINDER] [{}] {} (imp: {:.2})",
                    rec.profile, rec.content, rec.importance
                );
                info!("{msg}");

                if let Some(ref path) = log_path
                    && let Ok(mut file) = tokio::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                        .await
                {
                    let _ = file.write_all(format!("{}\n", &msg).as_bytes()).await;
                }

                if let Err(e) = state.store.mark_reminder_sent(rec.id).await {
                    error!("failed marking reminder {} sent: {e}", rec.id);
                }
            }
        }
    });
}
