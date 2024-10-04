//! Timer for performance test.

use anyhow::{anyhow, bail, Result};
use log::{error, info};

use coarsetime::{Duration, Instant, Updater};

/// Timer.
pub struct Timer {
    /// Keyword to print.
    keyword: String,

    /// Start time.
    start: Instant,
}

impl Timer {
    /// Create a new timer.
    pub fn new(keyword: String) -> Self {
        Self {
            keyword,
            start: Instant::now(),
        }
    }

    /// Print the elapsed time.
    pub fn print_elapsed(&self) {
        info!(
            "{}, elapsed: {}us",
            self.keyword,
            self.start.elapsed().as_micros()
        );
    }
}
