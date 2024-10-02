//! Tool.

use log::{error, info};
use std::{io::Write, sync::Once};

/// Init log. Set log format.
pub fn init_log() {
    env_logger::builder()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] {}:{} - {}",
                chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.level(),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        })
        .filter_level(log::LevelFilter::Info)
        .init();
}

static INIT_LOG: Once = Once::new();

/// Setup.
pub fn setup_log() {
    INIT_LOG.call_once(|| {
        init_log();
    });
}
