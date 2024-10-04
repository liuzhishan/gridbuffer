//! GridBuffer.
//!
//! GridBuffer is a format which is specifically designed for storing data, which is used for
//! ad sparse model.
//!
//! GridBuffer use `simd` to speedup both compression and IO operations. The data is stored
//! by column-first. And interface is similar to `ndarray`.
//!
//! Why design a new format?
//!
//! The short answer is: The data format is the core part for data storage and IO performance,
//! it's definitely worth it to design a new format for extreme efficiency and performance.
//!
//! The long answer is about some critical reasons for model traing:
//!
//! 1. Because sample for training is large, we need high performance format to speed up IO, and
//! also high compression ratio to reduce the size of data. It's very important for large model
//! training, because it will be the whole bottleneck of training easily if the IO is not fast
//! enough.
//!
//! 2. The Sample is usually constructed as `row`, but when it comes to training, the samples are
//! assembled as fixed batch, which will feeding to training framework later for training. The
//! assembly prcess is also time-consuming and need more processing logic.
//!
//! 3. Each model has its own features to train, which means for each row, we select a subset
//! of features to feed into model. So if the total number of features is large, the number of
//! features for each sample is just a small number. It's a waste that we need to read the whole
//! row to get a small number of features.
//!
//! 4. We usually has a group of features for base model, and when we want to add new features,
//! we often need to join new data to base features according to keys, such as `request_id` or
//! `user_id` and so on. The join work is usually done by `hive`, but `join` in `hive` is often
//! slow and resource consuming when the data is large. Because we need to shuffle the base features
//! and new features globally.
//!
//! There are some solutions currently:
//!
//! 1. protobuf
//! 2. flatbuffers
//! 3. arrow
//! 4. self-defined batch format.
//! 5. merge-on-read using bucket join.
//!
//! Let me explain them detailly.
//!
//! 1. protobuf
//!
//! 2. flatbuffers
//!
//! 3. arrow
//!
//! 4. self-defined batch format.
//!
//! 5. merge-on-read using bucket join.
//!
//! Each solution above has its own pros and cons, and cannot meet the requirements of all the
//! scenarios.
//!
//! `GridBuffer` is designed to solve all the critical problems above for sparse model training.
//!
#![feature(portable_simd)]
use core::simd::prelude::*;

use anyhow::{bail, Result};
use bitpacking::BitPacker4x;
use likely_stable::{likely, unlikely};
use log::{error, info};
use std::ops::Range;
use strum::{EnumCount, EnumDiscriminants, EnumString, FromRepr, ToString};

use bitpacking::BitPacker;

use crate::error_bail;

use super::{
    simd::{compress_bitpacker4x, decompress_bitpacker4x},
    tool::{check_data_length, check_range},
};

#[derive(
    Default, Clone, FromRepr, Debug, PartialEq, EnumCount, EnumDiscriminants, EnumString, ToString,
)]
#[repr(u8)]
pub enum GridDataType {
    #[default]
    U64,
    F32,
}

pub const GRID_BUFFER_VERSION: u8 = 1;

/// GridBuffer format.
pub struct GridBuffer {
    version: u8,
    num_rows: usize,
    num_cols: usize,
    u64_data: Vec<u64>,
    f32_data: Vec<f32>,
    cells: Vec<Vec<GridCell>>,
}

impl GridBuffer {
    /// Construct a new grid buffer.
    pub fn new(num_rows: usize, num_cols: usize, capacity: usize) -> Self {
        Self {
            version: GRID_BUFFER_VERSION,
            num_rows,
            num_cols,
            u64_data: Vec::with_capacity(capacity),
            f32_data: Vec::with_capacity(capacity),
            cells: vec![vec![GridCell::default(); num_cols]; num_rows],
        }
    }

    /// Construct a new grid buffer with all the fields.
    pub fn new_with_fields(
        num_rows: usize,
        num_cols: usize,
        u64_data: Vec<u64>,
        f32_data: Vec<f32>,
        cells: Vec<Vec<GridCell>>,
    ) -> Self {
        Self {
            version: GRID_BUFFER_VERSION,
            num_rows,
            num_cols,
            u64_data,
            f32_data,
            cells,
        }
    }

    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    #[inline]
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Push a cell into the grid buffer.
    #[inline]
    pub fn push_cell(&mut self, row: usize, col: usize, value: GridCell) {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row, col);
            return;
        }

        self.cells[row][col] = value;
    }

    /// Get a cell from the grid buffer.
    #[inline]
    pub fn get_cell(&self, row: usize, col: usize) -> Option<&GridCell> {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return None;
        }

        Some(&self.cells[row][col])
    }

    /// Push a u64 value into the grid buffer.
    #[inline]
    pub fn push_u64_values(&mut self, row: usize, col: usize, values: &[u64]) {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return;
        }

        self.u64_data.extend_from_slice(values);
        let range = self.u64_data.len() - values.len()..self.u64_data.len();

        self.cells[row][col] = GridCell::U64Cell(GridCellU64 { range });
    }

    /// Push a f32 value into the grid buffer.
    #[inline]
    pub fn push_f32_values(&mut self, row: usize, col: usize, values: &[f32]) {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return;
        }

        self.f32_data.extend_from_slice(values);
        let range = self.f32_data.len() - values.len()..self.f32_data.len();

        self.cells[row][col] = GridCell::F32Cell(GridCellF32 { range });
    }

    /// Get u64 values from the cell of grid buffer.
    #[inline]
    pub fn get_u64_values(&self, row: usize, col: usize) -> Option<&[u64]> {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return None;
        }

        match &self.cells[row][col] {
            GridCell::U64Cell(cell) => Some(&self.u64_data[cell.range.clone()]),
            _ => None,
        }
    }

    /// Get f32 values from the cell of grid buffer.
    #[inline]
    pub fn get_f32_values(&self, row: usize, col: usize) -> Option<&[f32]> {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return None;
        }

        match &self.cells[row][col] {
            GridCell::F32Cell(cell) => Some(&self.f32_data[cell.range.clone()]),
            _ => None,
        }
    }

    /// Get the number of values in u64 data.
    #[inline]
    pub fn total_num_u64_values(&self) -> usize {
        self.u64_data.len()
    }

    /// Get the number of values in f32 data.
    #[inline]
    pub fn total_num_f32_values(&self) -> usize {
        self.f32_data.len()
    }

    /// Get the number of values in the cell.
    #[inline]
    pub fn get_cell_num_values(&self, row: usize, col: usize) -> usize {
        match &self.cells[row][col] {
            GridCell::U64Cell(cell) => cell.range.len(),
            GridCell::F32Cell(cell) => cell.range.len(),
        }
    }

    /// Get the total bytes of the grid buffer.
    #[inline]
    pub fn estimated_bytes(&self) -> usize {
        self.u64_data.len() * 8
            + self.f32_data.len() * 4
            + self.cells.len() * self.num_cols * 9
            + 10
    }

    /// Serialize the GridBuffer to bytes.
    ///
    /// The format is the following concatenation:
    ///
    /// 1. version: u8
    /// 2. num_rows: usize
    /// 3. num_cols: usize
    /// 4. total_num_u64_values: usize
    /// 5. total_num_f32_values: usize
    /// 6. num_rows * num_cols * 2 usize:
    ///      for each cell, it's stored as:
    ///          data_type: u8
    ///          range.start: usize
    ///          range.end: usize
    /// 7. compressed_u64_total_num_bytes: usize
    /// 8. num_bits: u8
    /// 9. compressed u64 bytes
    /// 10. compressed_f32_total_num_bytes: usize
    /// 11. num_bits: u8
    /// 12. compressed f32 bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.estimated_bytes());
        
        // version
        self.push_u8_le(self.version, &mut buf);

        // num_rows, num_cols
        // Store as u32.
        self.push_usize_le(self.num_rows, &mut buf);
        self.push_usize_le(self.num_cols, &mut buf);

        // total_num_u64_values, total_num_f32_values
        self.push_usize_le(self.total_num_u64_values(), &mut buf);
        self.push_usize_le(self.total_num_f32_values(), &mut buf);

        // Cells.
        self.cells_to_bytes(&mut buf);

        // Compressed u64 data.
        self.u64_data_to_bytes(&mut buf);

        // Compressed f32 data.
        self.f32_data_to_bytes(&mut buf);

        buf
    }

    /// Cells to bytes.
    fn cells_to_bytes(&self, buf: &mut Vec<u8>) {
        // Cell ranges.
        for row in &self.cells {
            for cell in row {
                let range = cell.range();

                // data_type
                self.push_u8_le(cell.data_type() as u8, buf);

                // range
                self.push_usize_le(range.start, buf);
                self.push_usize_le(range.end, buf);
            }
        }
    }

    /// U64 data to bytes.
    #[inline]
    fn u64_data_to_bytes(&self, buf: &mut Vec<u8>) {
        let compressed_u64_data = self.compress_u64_data(self.u64_data.as_slice());

        self.push_usize_le(compressed_u64_data.len(), buf);
        buf.extend_from_slice(compressed_u64_data.as_slice());
    }

    /// F32 data to bytes.
    #[inline]
    fn f32_data_to_bytes(&self, buf: &mut Vec<u8>) {
        let compressed_f32_data = self.compress_f32_data(self.f32_data.as_slice());

        self.push_usize_le(compressed_f32_data.len(), buf);
        buf.extend_from_slice(compressed_f32_data);
    }

    /// Parse num_rows and num_cols from bytes.
    #[inline]
    fn parse_num_rows_and_cols(bytes: &[u8]) -> (usize, usize) {
        (
            Self::parse_usize_le_unchecked(bytes, 1),
            Self::parse_usize_le_unchecked(bytes, 5),
        )
    }

    /// Parse num_u64_values and num_f32_values from bytes.
    #[inline]
    fn parse_num_u64_and_f32_values(bytes: &[u8]) -> (usize, usize) {
        (
            Self::parse_usize_le_unchecked(bytes, 9),
            Self::parse_usize_le_unchecked(bytes, 13),
        )
    }

    /// Parse cells from bytes.
    fn parse_cells(
        bytes: &[u8],
        num_rows: usize,
        num_cols: usize,
        total_num_u64_values: usize,
        total_num_f32_values: usize,
    ) -> Result<(Vec<Vec<GridCell>>, usize)> {
        let mut cells = vec![vec![GridCell::default(); num_cols]; num_rows];

        let usize_size = 4;
        let mut pos = 4 * usize_size + 1;

        for row in 0..num_rows {
            for col in 0..num_cols {
                let data_type = match GridDataType::from_repr(bytes[pos]) {
                    Some(data_type) => data_type,
                    None => {
                        error_bail!("invalid data type: {}", bytes[pos]);
                    }
                };

                let range_start = Self::parse_usize_le_unchecked(bytes, pos + 1);
                let range_end = Self::parse_usize_le_unchecked(bytes, pos + 1 + usize_size);

                if data_type == GridDataType::U64 {
                    check_range(range_start, range_end, total_num_u64_values)?;

                    cells[row][col] = GridCell::U64Cell(GridCellU64 {
                        range: range_start..range_end,
                    });
                } else if data_type == GridDataType::F32 {
                    check_range(range_start, range_end, total_num_f32_values)?;

                    cells[row][col] = GridCell::F32Cell(GridCellF32 {
                        range: range_start..range_end,
                    });
                } else {
                    error_bail!("invalid data type: {}", data_type.to_string());
                }

                pos += 2 * usize_size + 1;
            }
        }

        Ok((cells, pos))
    }

    /// Parse u64 data from bytes.
    /// 
    /// Return the `u64_data`position of the next field.
    fn parse_u64_data(bytes: &[u8], start_pos: usize, total_num_u64_values: usize) -> Result<(Vec<u64>, usize)> {
        // compressed_u64_data_len.
        let usize_size = 4;

        let mut pos = start_pos;

        let compressed_u64_data_len = Self::parse_usize_le_unchecked(bytes, pos);
        let u64_end = pos + usize_size + compressed_u64_data_len;

        let mut u32_data = vec![0u32; compressed_u64_data_len / 4];

        let mut u32_pos = 0;
        let mut total_u32_len = 0;

        // num_bits.
        pos += usize_size;

        while pos < u64_end {
            let len = Self::parse_usize_le_unchecked(bytes, pos);
            pos += usize_size;

            let num_bits = bytes[pos];
            pos += 1;

            if num_bits > 0 {
                // Use `simd` to decompress.
                decompress_bitpacker4x(
                    &bytes[pos..pos + len],
                    num_bits,
                    &mut u32_data[u32_pos..u32_pos + BitPacker4x::BLOCK_LEN],
                );

                u32_pos += BitPacker4x::BLOCK_LEN;
                pos += len;
                total_u32_len += BitPacker4x::BLOCK_LEN;
            } else {
                // No decompression. Just copy the data.
                //
                // If we use `bytes[pos..pos + len].align_to::<u32>()`, the `prefix.len()` will be `3`,
                // and the `suffix.len()` will be `1`. But the `len` is exactly multiple of `4`.
                //
                // Weird. It should be `prefix.len() == 0` and `suffix.len() == 0`.
                // But why?
                //
                // TODO: Find why `bytes[pos..pos + len].align_to::<u32>()` is not working.
                let vec = bytes[pos..pos + len].to_vec();
                let (prefix, middle, suffix) = unsafe { vec.align_to::<u32>() };
                u32_data[u32_pos..u32_pos + middle.len()].copy_from_slice(middle);

                if unlikely(prefix.len() > 0 || suffix.len() > 0) {
                    error_bail!(
                        "invalid u32 data, prefix: {}, suffix: {}",
                        prefix.len(),
                        suffix.len()
                    );
                }

                u32_pos += middle.len();
                pos += len;
                total_u32_len += middle.len();
            }
        }

        u32_data.resize(total_u32_len, 0);

        let u64_data = Self::align_to_u64(u32_data.as_slice())?;

        Ok((u64_data, pos))
    }

    /// Align u32 data to u64.
    fn align_to_u64(data: &[u32]) -> Result<Vec<u64>> {
        let (u64_prefix, u64_middle, u64_suffix) = unsafe { data.align_to::<u64>() };

        if unlikely(u64_prefix.len() > 0 || u64_suffix.len() > 0) {
            error_bail!(
                "invalid u64 data, prefix: {}, suffix: {}",
                u64_prefix.len(),
                u64_suffix.len()
            );
        }

        let u64_data = u64_middle.to_vec();

        Ok(u64_data)
    }

    /// Parse f32 data from bytes.
    /// 
    /// Return the `f32_data`position of the next field.
    fn parse_f32_data(bytes: &[u8], start_pos: usize, total_num_f32_values: usize) -> Result<(Vec<f32>, usize)> {
        let usize_size = 4;

        let mut pos = start_pos;
        
        // f32 data.
        let compressed_f32_data_len = Self::parse_usize_le_unchecked(bytes, pos);
        pos += usize_size;

        let f32_data_start = pos;
        let f32_data_end = pos + compressed_f32_data_len;

        // Same as above `u32_data`.
        // TODO: Find why `bytes.align_to::<f32>()` is not working.
        let f32_vec = bytes[f32_data_start..f32_data_end].to_vec();

        let (prefix, middle, suffix) = unsafe { f32_vec.align_to::<f32>() };

        if unlikely(prefix.len() > 0 || suffix.len() > 0) {
            error_bail!(
                "invalid f32 data, prefix: {}, suffix: {}",
                prefix.len(),
                suffix.len()
            );
        }

        let f32_data = middle.to_vec();

        Ok((f32_data, pos))
    }

    /// Deserialize the GridBuffer from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // version
        let version = bytes[0];

        // num_rows, num_cols
        let (num_rows, num_cols) = Self::parse_num_rows_and_cols(bytes);

        // total_num_u64_values, total_num_f32_values
        let (total_num_u64_values, total_num_f32_values) =
            Self::parse_num_u64_and_f32_values(bytes);

        // Cell ranges.
        let (cells, pos) = Self::parse_cells(
            bytes,
            num_rows,
            num_cols,
            total_num_u64_values,
            total_num_f32_values,
        )?;

        let (u64_data, pos) = Self::parse_u64_data(bytes, pos, total_num_u64_values)?;
        check_data_length(u64_data.len(), total_num_u64_values)?;

        let (f32_data, pos) = Self::parse_f32_data(bytes, pos, total_num_f32_values)?;
        check_data_length(f32_data.len(), total_num_f32_values)?;

        Ok(GridBuffer::new_with_fields(
            num_rows, num_cols, u64_data, f32_data, cells,
        ))
    }

    /// Serialize the GridBuffer to base64.
    pub fn to_base64(&self) -> String {
        let bytes = self.to_bytes();
        base64::encode(&bytes)
    }

    /// Deserialize the GridBuffer from base64.
    pub fn from_base64(base64: &str) -> Result<Self> {
        let bytes = base64::decode(base64)?;
        Self::from_bytes(&bytes)
    }

    /// Helper method to push a `usize` as little-endian bytes
    ///
    /// Store as u32.
    #[inline]
    fn push_usize_le(&self, value: usize, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&(value as u32).to_le_bytes());
    }

    /// Parse u32 from little-endian bytes.
    #[inline]
    fn parse_u32_le_unchecked(bytes: &[u8], pos: usize) -> u32 {
        u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
    }

    /// Parse usize from little-endian bytes.
    ///
    /// Use u32 as usize.
    #[inline]
    fn parse_usize_le_unchecked(bytes: &[u8], pos: usize) -> usize {
        Self::parse_u32_le_unchecked(bytes, pos) as usize
    }

    /// Helper method to push a `u8` as little-endian bytes
    #[inline]
    fn push_u8_le(&self, value: u8, buffer: &mut Vec<u8>) {
        buffer.push(value);
    }

    /// Compress the u64 data.
    #[inline]
    fn compress_u64_data(&self, data: &[u64]) -> Vec<u8> {
        let (prefix, middle, suffix) = unsafe { data.align_to::<u32>() };

        compress_bitpacker4x(middle)
    }

    /// Compress f32 in the future.
    #[inline]
    fn compress_f32_data<'a>(&self, data: &'a [f32]) -> &'a [u8] {
        let (prefix, middle, suffix) = unsafe { data.align_to::<u8>() };

        middle
    }
}

/// GridCell.
///
/// Meta data for each cell.
#[derive(Clone, Default)]
pub struct GridCellU64 {
    pub range: Range<usize>,
}

#[derive(Clone, Default)]
pub struct GridCellF32 {
    pub range: Range<usize>,
}

/// Use enum to represent different types of cells.
#[derive(Clone)]
pub enum GridCell {
    U64Cell(GridCellU64),
    F32Cell(GridCellF32),
}

impl Default for GridCell {
    fn default() -> Self {
        Self::U64Cell(GridCellU64::default())
    }
}

impl GridCell {
    /// Get the range of the cell.
    pub fn range(&self) -> Range<usize> {
        match self {
            GridCell::U64Cell(cell) => cell.range.clone(),
            GridCell::F32Cell(cell) => cell.range.clone(),
        }
    }

    /// Get the data type of the cell.
    pub fn data_type(&self) -> GridDataType {
        match self {
            GridCell::U64Cell(_) => GridDataType::U64,
            GridCell::F32Cell(_) => GridDataType::F32,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::tool::setup_log;

    use super::*;
    use log::info;

    #[test]
    fn test_gridbuffer_creation() {
        setup_log();

        let num_rows = 5;
        let num_cols = 5;
        let capacity = 500;

        let buffer = GridBuffer::new(num_rows, num_cols, capacity);
        assert_eq!(buffer.num_rows(), num_rows);
        assert_eq!(buffer.num_cols(), num_cols);
        assert_eq!(buffer.total_num_u64_values(), 0);
        assert_eq!(buffer.total_num_f32_values(), 0);
    }

    #[test]
    fn test_gridbuffer_set_and_get_u64() {
        setup_log();

        let num_rows = 2;
        let num_cols = 2;
        let capacity = 500;

        let mut buffer = GridBuffer::new(num_rows, num_cols, capacity);

        let u64_values = vec![1, 2, 3];
        let f32_values = vec![1.0, 2.0, 3.0];

        buffer.push_u64_values(0, 0, &u64_values);
        buffer.push_f32_values(1, 1, &f32_values);

        assert_eq!(buffer.get_u64_values(0, 0), Some(u64_values.as_slice()));
        assert_eq!(buffer.get_f32_values(1, 1), Some(f32_values.as_slice()));
        assert_eq!(buffer.total_num_u64_values(), 3);
        assert_eq!(buffer.total_num_f32_values(), 3);
    }

    #[test]
    fn test_gridbuffer_set_and_get_f32() {
        setup_log();

        let num_rows = 2;
        let num_cols = 2;
        let capacity = 500;

        let mut buffer = GridBuffer::new(num_rows, num_cols, capacity);

        let f32_values = vec![1.0, 2.0, 3.0];
        let f32_values2 = vec![4.0, 5.0];

        buffer.push_f32_values(0, 1, &f32_values);
        buffer.push_f32_values(1, 0, &f32_values2);

        assert_eq!(buffer.get_f32_values(0, 1), Some(f32_values.as_slice()));
        assert_eq!(buffer.get_f32_values(1, 0), Some(f32_values2.as_slice()));
        assert_eq!(buffer.total_num_f32_values(), 5);
    }

    #[test]
    fn test_gridbuffer_get_cell_num_values() {
        setup_log();

        let num_rows = 2;
        let num_cols = 2;
        let capacity = 500;

        let mut buffer = GridBuffer::new(num_rows, num_cols, capacity);

        let u64_values = vec![1, 2, 3];
        let u64_values2 = vec![4, 5];

        let f32_values = vec![1.0, 2.0];
        let f32_values2 = vec![3.0, 4.0, 5.0];

        buffer.push_u64_values(0, 0, &u64_values);
        buffer.push_f32_values(0, 1, &f32_values);
        buffer.push_u64_values(1, 0, &u64_values2);
        buffer.push_f32_values(1, 1, &f32_values2);

        assert_eq!(buffer.get_cell_num_values(0, 0), 3);
        assert_eq!(buffer.get_cell_num_values(0, 1), 2);
        assert_eq!(buffer.get_cell_num_values(1, 0), 2);
        assert_eq!(buffer.get_cell_num_values(1, 1), 3);
    }

    #[test]
    fn test_gridbuffer_out_of_bounds_row() {
        let buffer = GridBuffer::new(2, 2, 500);
        assert_eq!(buffer.get_u64_values(2, 0), None);
    }

    #[test]
    fn test_gridbuffer_out_of_bounds_col() {
        let buffer = GridBuffer::new(2, 2, 500);
        assert_eq!(buffer.get_f32_values(0, 2), None);
    }

    #[test]
    fn test_gridbuffer_to_bytes() {
        setup_log();

        let num_rows = 2;
        let num_cols = 2;
        let capacity = 500;

        let mut buffer = GridBuffer::new(num_rows, num_cols, capacity);

        // Add some u64 and f32 values
        let u64_values = vec![1, 2, 3];
        let f32_values = vec![1.0, 2.0, 3.0];

        buffer.push_u64_values(0, 0, &u64_values);
        buffer.push_f32_values(1, 1, &f32_values);

        // Convert to bytes
        let bytes = buffer.to_bytes();

        // Create a new GridBuffer from the bytes
        let restored_buffer = GridBuffer::from_bytes(&bytes).unwrap();

        // Check if the restored buffer matches the original
        assert_eq!(restored_buffer.num_rows(), num_rows);
        assert_eq!(restored_buffer.num_cols(), num_cols);

        assert_eq!(
            restored_buffer.get_u64_values(0, 0),
            Some(u64_values.as_slice())
        );
        assert_eq!(
            restored_buffer.get_f32_values(1, 1),
            Some(f32_values.as_slice())
        );

        // Check if empty cells are still empty
        let empty_u64_values = vec![];

        assert_eq!(
            restored_buffer.get_u64_values(0, 1),
            Some(empty_u64_values.as_slice())
        );
        assert_eq!(restored_buffer.get_f32_values(1, 0), None);

        // Check total number of values
        assert_eq!(restored_buffer.total_num_u64_values(), 3);
        assert_eq!(restored_buffer.total_num_f32_values(), 3);
    }

    #[test]
    fn test_gridbuffer_to_bytes_empty() {
        setup_log();

        let buffer = GridBuffer::new(2, 2, 500);

        let bytes = buffer.to_bytes();

        let restored_buffer = GridBuffer::from_bytes(&bytes).unwrap();

        assert_eq!(restored_buffer.num_rows(), 2);
        assert_eq!(restored_buffer.num_cols(), 2);
        assert_eq!(restored_buffer.total_num_u64_values(), 0);
        assert_eq!(restored_buffer.total_num_f32_values(), 0);
    }
}
