//! Timer for performance test.

use coarsetime::Instant;
use log::info;

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

    /// Get time elapsed.
    pub fn get_elapsed_micros(&self) -> u64 {
        self.start.elapsed().as_micros()
    }

    /// Print the elapsed time.
    pub fn print_elapsed(&self) {
        info!(
            "{}, elapsed: {} us",
            self.keyword,
            self.start.elapsed().as_micros()
        );
    }
}
