//! Tool.

use anyhow::{bail, Result};
use likely_stable::{likely, unlikely};
use log::{error, info};
use std::{io::Write, sync::Once};

use super::gridbuffer::CompressionType;

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

/// Check prefix and suffix of alignment result.
#[inline]
pub fn check_alignment_result(prefix: usize, suffix: usize) -> Result<()> {
    if unlikely(prefix > 0 || suffix > 0) {
        error_bail!(
            "invalid alignment result: prefix: {}, suffix: {}",
            prefix,
            suffix
        );
    }

    Ok(())
}

/// Check compression type.
#[inline]
pub fn check_compression_type(
    compression_type: CompressionType,
    expected_compression_type: CompressionType,
) -> Result<()> {
    if unlikely(compression_type != expected_compression_type) {
        error_bail!(
            "invalid compression type: {}, expected: {}",
            compression_type.to_string(),
            expected_compression_type.to_string()
        );
    }

    Ok(())
}

/// Sort u32 data and index simultaneously.
///
/// Use `U32Sorter` to avoid `malloc` to store the sorted data.
pub struct U32Sorter {
    indexes: Vec<u32>,
    data: Vec<u32>,
}

impl U32Sorter {
    pub fn new() -> Self {
        Self {
            indexes: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            indexes: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.indexes.len()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.indexes.clear();
        self.data.clear();
    }

    #[inline]
    pub fn indexes(&self) -> &[u32] {
        &self.indexes
    }

    #[inline]
    pub fn data(&self) -> &[u32] {
        &self.data
    }

    /// Sort the data and keep the original index.
    ///
    /// Sort by value in data, index arranged by value order.
    ///
    /// For example:
    ///
    /// before sort:
    /// data: [100, 0, 1, 0, 50, 0, 2, 0], indexes: [0, 1, 2, 3, 4, 5, 6, 7]
    ///
    /// after sort:
    /// data: [0, 0, 0, 0, 1, 2, 50, 100], indexes: [1, 3, 5, 7, 2, 6, 4, 0]
    pub fn sort(&mut self, data: &[u32]) {
        self.indexes.clear();
        self.data.clear();

        self.indexes.extend((0..data.len()).into_iter().map(|i| i as u32));
        self.data.extend(data.iter().copied());

        self.indexes.sort_by_key(|&x| self.data[x as usize]);
        self.data.sort_by(|a, b| a.cmp(b));
    }

    /// Sort the data and index inplace by index.
    ///
    /// For example:
    ///
    /// before sort:
    /// data: [0, 0, 0, 0, 1, 2, 50, 100], indexes: [1, 3, 5, 7, 2, 6, 4, 0]
    ///
    /// after sort:
    /// data: [100, 0, 1, 0, 50, 0, 2, 0], indexes: [0, 1, 2, 3, 4, 5, 6, 7]
    pub fn sort_by_index(data: &mut [u32], indexes: &mut [u32]) {
        Self::quick_sort_by_index(data, indexes);
    }

    /// Quick sort to simultaneously sort indexes and data by index.
    /// The data is sorted accordingly.
    pub fn quick_sort_by_index(data: &mut [u32], indexes: &mut [u32]) {
        if data.len() <= 1 {
            return;
        }
        let pivot = Self::partition(data, indexes);

        let (left_data, right_data) = data.split_at_mut(pivot);
        let (left_indexes, right_indexes) = indexes.split_at_mut(pivot);

        Self::quick_sort_by_index(left_data, left_indexes);
        Self::quick_sort_by_index(&mut right_data[1..], &mut right_indexes[1..]);
    }

    fn partition(data: &mut [u32], indexes: &mut [u32]) -> usize {
        let pivot = indexes.len() - 1;
        let mut i = 0;

        for j in 0..pivot {
            if indexes[j] <= indexes[pivot] {
                indexes.swap(i, j);
                data.swap(i, j);
                i += 1;
            }
        }

        indexes.swap(i, pivot);
        data.swap(i, pivot);

        i
    }

}
