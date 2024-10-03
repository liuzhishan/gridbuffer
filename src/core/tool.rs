//! Tool.

use anyhow::{bail, Result};
use likely_stable::{likely, unlikely};
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

#[macro_export]
macro_rules! error_bail {
    ($msg:literal $(,)?) => {
        error!($msg);
        bail!($msg)
    };
    ($err:expr $(,)?) => {
        error!($err);
        bail!(err)
    };
    ($fmt:expr, $($arg:tt)*) => {
        error!($fmt, $($arg)*);
        bail!($fmt, $($arg)*)
    };
}

/// Check range.
#[inline]
pub fn check_range(range_start: usize, range_end: usize, total_num_values: usize) -> Result<()> {
    if unlikely(range_start > range_end) {
        error_bail!("invalid range: {}..{}", range_start, range_end);
    }

    if unlikely(total_num_values == 0) {
        return Ok(());
    }

    if unlikely(range_start >= total_num_values || range_end > total_num_values) {
        error_bail!(
            "invalid range: {}..{}, total_num_values: {}",
            range_start,
            range_end,
            total_num_values
        );
    }

    Ok(())
}

/// Check data length.
#[inline]
pub fn check_data_length(data_length: usize, total_num_values: usize) -> Result<()> {
    if unlikely(data_length != total_num_values) {
        error_bail!(
            "invalid data length: {}, expected: {}",
            data_length,
            total_num_values
        );
    }

    Ok(())
}
