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
use bitpacking::{BitPacker4x, BitPacker8x};
use likely_stable::{likely, unlikely};
use log::{error, info};
use std::{arch::is_riscv_feature_detected, default, ops::Range};
use strum::{EnumCount, EnumDiscriminants, EnumString, FromRepr, ToString};

use crate::core::tool::check_alignment_result;

use bitpacking::BitPacker;

use crate::error_bail;

use super::{
    simd::{
        compress_bitpacker, compress_bitpacker4x, compress_bitpacker4x_sorted,
        compress_bitpacker8x, decompress_bitpacker, decompress_bitpacker4x,
    },
    tool::{check_compression_type, check_data_length, check_range, U32Sorter},
};

/// The data type of grid buffer.
#[derive(
    Default, Clone, FromRepr, Debug, PartialEq, EnumCount, EnumDiscriminants, EnumString, ToString,
)]
#[repr(u8)]
pub enum GridDataType {
    /// No data type.
    #[default]
    None,

    /// U64 data type.
    U64,

    /// F32 data type.
    F32,
}

/// The compression type of data.
#[derive(
    Default, Clone, FromRepr, Debug, PartialEq, EnumCount, EnumDiscriminants, EnumString, ToString,
)]
#[repr(u8)]
pub enum CompressionType {
    /// No compression.
    #[default]
    None,

    /// Bit packing 4x.
    BitPacking4x,

    /// Bit packing 8x.
    BitPacking8x,

    /// Differential coding.
    DifferentialCoding,

    /// Bit packing 4x and differential coding.
    BitPacking4xDiffCoding,
}

pub trait GetCompressionType {
    fn get_compression_type() -> CompressionType;
}

impl GetCompressionType for BitPacker4x {
    fn get_compression_type() -> CompressionType {
        CompressionType::BitPacking4x
    }
}

impl GetCompressionType for BitPacker8x {
    fn get_compression_type() -> CompressionType {
        CompressionType::BitPacking8x
    }
}

pub const GRID_BUFFER_VERSION: u8 = 1;

pub const FOUR: usize = 4;

pub struct GridBasicInfo {
    pub version: u8,
    pub num_rows: usize,
    pub num_cols: usize,
    pub total_num_u64_values: usize,
    pub total_num_f32_values: usize,
    pub cells: Vec<Vec<GridCell>>,
}

/// GridBuffer format.
///
/// Use `simd` to compress the `u64_data`.
#[derive(Clone)]
pub struct GridBuffer {
    version: u8,
    num_rows: usize,
    num_cols: usize,
    u64_data: Vec<u64>,
    f32_data: Vec<f32>,
    cells: Vec<Vec<GridCell>>,
}

impl GridBuffer {
    /// Construct an empty new grid buffer.
    /// 
    /// To be more flexible, we don't set the grid size here.
    pub fn new() -> Self {
        Self {
            version: GRID_BUFFER_VERSION,
            num_rows: 0,
            num_cols: 0,
            u64_data: vec![],
            f32_data: vec![],
            cells: vec![],
        }
    }

    /// Construct a new grid buffer.
    pub fn new_with_num_rows_cols(num_rows: usize, num_cols: usize) -> Self {
        let capacity = num_rows * num_cols * 5;

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

    /// Clear the grid buffer.
    pub fn clear(&mut self) {
        self.u64_data.clear();
        self.f32_data.clear();

        for row in &mut self.cells {
            for cell in row {
                cell.clear();
            }
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

    #[inline]
    pub fn set_num_cols(&mut self, num_cols: usize) {
        self.num_cols = num_cols;
    }

    /// Set the number of rows and columns.
    #[inline]
    pub fn set_num_rows_cols(&mut self, num_rows: usize, num_cols: usize) {
        self.num_rows = num_rows;
        self.num_cols = num_cols;
    }

    /// Extend rows.
    #[inline]
    pub fn extend_rows(&mut self, num_rows: usize) {
        self.num_rows += num_rows;
        self.cells.extend(vec![vec![GridCell::default(); self.num_cols]; num_rows]);
    }

    /// Extend columns.
    #[inline]
    pub fn extend_cols(&mut self, num_cols: usize) {
        self.num_cols += num_cols;
        for row in &mut self.cells {
            row.extend(vec![GridCell::default(); num_cols]);
        }
    }

    /// Extend rows and columns.
    #[inline]
    pub fn extend_rows_cols(&mut self, num_rows: usize, num_cols: usize) {
        self.extend_rows(num_rows);
        self.extend_cols(num_cols);
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

    #[inline]
    pub fn push_u64(&mut self, row: usize, col: usize, value: u64) {
        self.push_u64_values(row, col, &[value]);
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

    #[inline]
    pub fn push_f32(&mut self, row: usize, col: usize, value: f32) {
        self.push_f32_values(row, col, &[value]);
    }

    /// Get u64 values from the cell of grid buffer.
    ///
    /// For better ease of use, return the slice of `u64_data` directly. If the data is not exists,
    /// return empty slice.
    #[inline]
    pub fn get_u64_values(&self, row: usize, col: usize) -> &[u64] {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return &[];
        }

        match &self.cells[row][col] {
            GridCell::U64Cell(cell) => &self.u64_data[cell.range.clone()],
            _ => &[],
        }
    }

    #[inline]
    pub fn get_u64(&self, row: usize, col: usize) -> Option<u64> {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return None;
        }

        match &self.cells[row][col] {
            GridCell::U64Cell(cell) => {
                if cell.range.len() == 1 {
                    Some(self.u64_data[cell.range.start])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get f32 values from the cell of grid buffer.
    #[inline]
    pub fn get_f32_values(&self, row: usize, col: usize) -> &[f32] {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return &[];
        }

        match &self.cells[row][col] {
            GridCell::F32Cell(cell) => &self.f32_data[cell.range.clone()],
            _ => &[],
        }
    }

    #[inline]
    pub fn get_f32(&self, row: usize, col: usize) -> Option<f32> {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return None;
        }

        match &self.cells[row][col] {
            GridCell::F32Cell(cell) => {
                if cell.range.len() == 1 {
                    Some(self.f32_data[cell.range.start])
                } else {
                    None
                }
            }
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
            GridCell::None => 0,
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
    /// 7. compression_type: u8
    /// 8. compressed_u64_total_num_bytes: usize
    /// 9. num_bits: u8
    /// 10. compressed u64 bytes
    /// 11. compresson_type: u8
    /// 12. compressed_f32_total_num_bytes: usize
    /// 14. compressed f32 bytes
    #[inline]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_bitpacking4x()
    }

    /// Serialize the GridBuffer to bytes with `BitPacking` compression.
    pub fn to_bytes_with_bitpacking<T: BitPacker + GetCompressionType>(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.estimated_bytes());

        self.serialize_basic_info(&mut buf);

        // Compressed u64 data.
        self.u64_data_to_bytes_bitpacking::<T>(&mut buf);

        // Compressed f32 data.
        self.f32_data_to_bytes(&mut buf);

        buf
    }

    /// Serialize the GridBuffer to bytes with `BitPacking4x` compression.
    #[inline]
    pub fn to_bytes_bitpacking4x(&self) -> Vec<u8> {
        self.to_bytes_with_bitpacking::<BitPacker4x>()
    }

    /// Serialize the GridBuffer to bytes with `BitPacking8x` compression.
    #[inline]
    pub fn to_bytes_bitpacking8x(&self) -> Vec<u8> {
        self.to_bytes_with_bitpacking::<BitPacker8x>()
    }

    /// Using differential coding to compress the u64 data.
    ///
    /// The u64 data is sorted before compression. To avoid `malloc` to store the sorted data,
    /// we pass a new parameter `U32Sorter` to reuse the memory.
    pub fn to_bytes_with_sorted(&self, sorter: &mut U32Sorter) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.estimated_bytes());

        // Serialize basic info.
        self.serialize_basic_info(&mut buf);

        // Compressed u64 data.
        self.u64_data_to_bytes_with_sorted(&mut buf, sorter);

        // Compressed f32 data.
        self.f32_data_to_bytes(&mut buf);

        buf
    }

    /// Serialize basic info of the grid buffer, including:
    /// 1. version: u8
    /// 2. num_rows: usize
    /// 3. num_cols: usize
    /// 4. total_num_u64_values: usize
    /// 5. total_num_f32_values: usize
    /// 6. cells: Vec<Vec<GridCell>>
    pub fn serialize_basic_info(&self, buf: &mut Vec<u8>) {
        // version
        self.push_u8_le(self.version, buf);

        // num_rows, num_cols
        // Store as u32.
        self.push_usize_le(self.num_rows, buf);
        self.push_usize_le(self.num_cols, buf);

        // total_num_u64_values, total_num_f32_values
        self.push_usize_le(self.total_num_u64_values(), buf);
        self.push_usize_le(self.total_num_f32_values(), buf);

        // Cells.
        self.cells_to_bytes(buf);
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

    /// U64 data to bytes using `BitPacking`.
    fn u64_data_to_bytes_bitpacking<T: BitPacker + GetCompressionType>(&self, buf: &mut Vec<u8>) {
        let compressed_u64_data = self.compress_u64_data_bitpacking::<T>(self.u64_data.as_slice());

        self.push_u8_le(T::get_compression_type() as u8, buf);
        self.push_usize_le(compressed_u64_data.len(), buf);

        buf.extend_from_slice(compressed_u64_data.as_slice());
    }

    /// U64 data to bytes using `BitPacking4x`.
    fn u64_data_to_bytes_bitpacking4x(&self, buf: &mut Vec<u8>) {
        self.u64_data_to_bytes_bitpacking::<BitPacker4x>(buf);
    }

    /// U64 data to bytes using `BitPacking8x`.
    fn u64_data_to_bytes_bitpacking8x(&self, buf: &mut Vec<u8>) {
        self.u64_data_to_bytes_bitpacking::<BitPacker8x>(buf);
    }

    /// U64 data to bytes with differential coding.
    ///
    /// First we build the u64 index for each u64 value, which is the position in the original `u64_data`.
    /// Then we use the `compress_u64_data_with_bitpacking4x_sorted` to compress the u64 data.
    /// And use `BitPacking4x` to compress the index. Then we store both the compressed u64 data and
    /// the compressed index in bytes.
    fn u64_data_to_bytes_with_sorted(&self, buf: &mut Vec<u8>, sorter: &mut U32Sorter) {
        // Align to u32.
        let (prefix, middle, suffix) = unsafe { self.u64_data.align_to::<u32>() };

        sorter.sort(middle);

        let compressed_data = self.compress_u64_data_with_bitpacking4x_sorted(sorter.data());
        let compressed_index = self.compress_u32_data(sorter.indexes());

        // Store the compression type.
        self.push_u8_le(CompressionType::BitPacking4xDiffCoding as u8, buf);

        // The data and index is seperated.
        // Store the length of compressed u64 data.
        self.push_usize_le(compressed_data.len(), buf);

        // Store the compressed u64 data.
        buf.extend_from_slice(compressed_data.as_slice());

        // Store the length of compressed u64 index.
        self.push_usize_le(compressed_index.len(), buf);

        // Store the compressed u64 index.
        buf.extend_from_slice(compressed_index.as_slice());
    }

    /// F32 data to bytes.
    fn f32_data_to_bytes(&self, buf: &mut Vec<u8>) {
        let compressed_f32_data = self.compress_f32_data(self.f32_data.as_slice());

        self.push_u8_le(CompressionType::None as u8, buf);
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

                if data_type == GridDataType::None {
                    cells[row][col] = GridCell::None;
                } else if data_type == GridDataType::U64 {
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
    fn parse_u64_data(
        bytes: &[u8],
        start_pos: usize,
        total_num_u64_values: usize,
    ) -> Result<(Vec<u64>, usize)> {
        // compressed_u64_data_len.
        let usize_size = 4;

        let compression_type = match CompressionType::from_repr(bytes[start_pos]) {
            Some(compression_type) => compression_type,
            None => {
                error_bail!("invalid compression type: {}", bytes[start_pos]);
            }
        };

        match compression_type {
            CompressionType::None => {
                Self::parse_u64_data_without_compression(bytes, start_pos + 1, total_num_u64_values)
            }
            CompressionType::BitPacking4x => {
                Self::parse_u64_data_with_bitpacking4x(bytes, start_pos + 1, total_num_u64_values)
            }
            CompressionType::BitPacking8x => {
                Self::parse_u64_data_with_bitpacking8x(bytes, start_pos + 1, total_num_u64_values)
            }
            CompressionType::BitPacking4xDiffCoding => {
                Self::parse_u64_data_with_bitpacking4x_sorted(
                    bytes,
                    start_pos + 1,
                    total_num_u64_values,
                )
            }
            _ => {
                error_bail!(
                    "unsupported compression_type: {}",
                    compression_type.to_string()
                );
            }
        }
    }

    /// Parse u64 data without compression.
    fn parse_u64_data_without_compression(
        bytes: &[u8],
        start_pos: usize,
        total_num_u64_values: usize,
    ) -> Result<(Vec<u64>, usize)> {
        let mut pos = start_pos;

        let compressed_data_len = Self::parse_usize_le_unchecked(bytes, pos);
        pos += 4;

        let u8_data = bytes[pos..pos + compressed_data_len].to_vec();

        let (prefix, middle, suffix) = unsafe { u8_data.align_to::<u64>() };
        check_alignment_result(prefix.len(), suffix.len())?;

        check_data_length(middle.len(), total_num_u64_values)?;

        Ok((middle.to_vec(), pos + compressed_data_len))
    }

    /// Parse u32 data with bitpacking.
    fn parse_u32_data_with_bitpacking<T: BitPacker + GetCompressionType>(
        bytes: &[u8],
        start_pos: usize,
        compressed_data_len: usize,
        total_num_u32_values: usize,
        is_sorted: bool,
    ) -> Result<(Vec<u32>, usize)> {
        // usize_size, fixed as 4.
        let usize_size = 4;

        // pos in bytes.
        let mut pos = start_pos;

        // end position of `u64_data` compressed in bytes.
        let u32_end = pos + compressed_data_len;

        // decompressed u32 data.
        let mut u32_data = vec![0u32; total_num_u32_values];

        // position in `u32_data`.
        let mut u32_pos = 0;

        // length of `u32_data`.
        let mut total_u32_len = 0;

        if compressed_data_len > 0 {
            while pos < u32_end {
                let len = Self::parse_usize_le_unchecked(bytes, pos);
                pos += FOUR;

                let num_bits = bytes[pos];
                pos += 1;

                // Be careful of the `num_bits == 0 && len == 0` case.
                //
                // We use `num_bits` and `len` to determine if the data is empty.
                //
                // If `num_bits` and `len` are both `0`, the data is empty. We need to fill the data with `0`.
                // But if `num_bits` is `0` and `len` is not `0`, the data is not compressed. We just copy the data.
                //
                // TODO: rewrite the logic to make it more clear. Maybe using an `enum` to represent the different cases.
                if num_bits > 0 && len > 0 {
                    // Use `simd` to decompress.
                    decompress_bitpacker::<T>(
                        &bytes[pos..pos + len],
                        num_bits,
                        &mut u32_data[u32_pos..u32_pos + T::BLOCK_LEN],
                        is_sorted,
                    );

                    u32_pos += T::BLOCK_LEN;
                    pos += len;
                    total_u32_len += T::BLOCK_LEN;
                } else if num_bits == 0 && len == 0 {
                    // The data is all empty, but is compressed using `bitpacking`. So the `num_bits` and `len` are both `0`.
                    u32_data[u32_pos..u32_pos + T::BLOCK_LEN].fill(0);

                    u32_pos += T::BLOCK_LEN;
                    pos += 0;
                    total_u32_len += T::BLOCK_LEN;
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
        }

        u32_data.resize(total_u32_len, 0);

        Ok((u32_data, pos))
    }

    /// Parse u32 data with bitpacking4x.
    fn parse_u32_data_with_bitpacking4x(
        bytes: &[u8],
        start_pos: usize,
        total_num_u32_values: usize,
    ) -> Result<(Vec<u32>, usize)> {
        let compressed_data_len = Self::parse_usize_le_unchecked(bytes, start_pos);

        Self::parse_u32_data_with_bitpacking::<BitPacker4x>(
            bytes,
            start_pos + FOUR,
            compressed_data_len,
            total_num_u32_values,
            false,
        )
    }

    /// Parse u32 data with bitpacking4x and differential coding.
    fn parse_u32_data_with_bitpacking4x_sorted(
        bytes: &[u8],
        start_pos: usize,
        num_values: usize,
    ) -> Result<(Vec<u32>, usize)> {
        let compressed_data_len = Self::parse_usize_le_unchecked(bytes, start_pos);

        let (mut u32_data, pos) = Self::parse_u32_data_with_bitpacking::<BitPacker4x>(
            bytes,
            start_pos + FOUR,
            compressed_data_len,
            num_values,
            false,
        )?;

        let compressed_index_len = Self::parse_usize_le_unchecked(bytes, pos);

        let (mut u32_index, pos) = Self::parse_u32_data_with_bitpacking::<BitPacker4x>(
            bytes,
            pos + FOUR,
            compressed_index_len,
            num_values,
            true,
        )?;

        U32Sorter::sort_by_index(u32_data.as_mut_slice(), u32_index.as_mut_slice());

        Ok((u32_data, pos))
    }

    /// Parse u64 data with bitpacking.
    fn parse_u64_data_with_bitpacking<T: BitPacker + GetCompressionType>(
        bytes: &[u8],
        start_pos: usize,
        num_values: usize,
        is_sorted: bool,
    ) -> Result<(Vec<u64>, usize)> {
        let compressed_data_len = Self::parse_usize_le_unchecked(bytes, start_pos);

        let (u32_data, pos) = Self::parse_u32_data_with_bitpacking::<T>(
            bytes,
            start_pos + FOUR,
            compressed_data_len,
            num_values * 2,
            is_sorted,
        )?;

        let u64_data = Self::align_to_u64(u32_data.as_slice())?;

        Ok((u64_data, pos))
    }

    /// Parse u64 data with bitpacking4x.
    #[inline]
    fn parse_u64_data_with_bitpacking4x(
        bytes: &[u8],
        start_pos: usize,
        num_values: usize,
    ) -> Result<(Vec<u64>, usize)> {
        Self::parse_u64_data_with_bitpacking::<BitPacker4x>(bytes, start_pos, num_values, false)
    }

    /// Parse u64 data with bitpacking8x.
    #[inline]
    fn parse_u64_data_with_bitpacking8x(
        bytes: &[u8],
        start_pos: usize,
        num_values: usize,
    ) -> Result<(Vec<u64>, usize)> {
        Self::parse_u64_data_with_bitpacking::<BitPacker8x>(bytes, start_pos, num_values, false)
    }

    /// Parse u64 data with bitpacking4x and differential coding.
    ///
    /// TODO: redundant with `Self::parse_u64_data_with_bitpacking4x`. Need merge.
    fn parse_u64_data_with_bitpacking4x_sorted(
        bytes: &[u8],
        start_pos: usize,
        num_values: usize,
    ) -> Result<(Vec<u64>, usize)> {
        let (u32_data, pos) =
            Self::parse_u32_data_with_bitpacking4x_sorted(bytes, start_pos, num_values * 2)?;

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
    fn parse_f32_data(
        bytes: &[u8],
        start_pos: usize,
        total_num_f32_values: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let mut pos = start_pos;

        let compression_type = match CompressionType::from_repr(bytes[pos]) {
            Some(compression_type) => compression_type,
            None => {
                error_bail!("invalid compression type: {}", bytes[pos]);
            }
        };
        pos += 1;

        check_compression_type(compression_type, CompressionType::None)?;

        // f32 data.
        let compressed_f32_data_len = Self::parse_usize_le_unchecked(bytes, pos);
        pos += FOUR;

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
        let (basic_info, pos) = Self::parse_basic_info(bytes)?;

        let (u64_data, pos) = Self::parse_u64_data(bytes, pos, basic_info.total_num_u64_values)?;
        check_data_length(u64_data.len(), basic_info.total_num_u64_values)?;

        let (f32_data, pos) = Self::parse_f32_data(bytes, pos, basic_info.total_num_f32_values)?;
        check_data_length(f32_data.len(), basic_info.total_num_f32_values)?;

        Ok(GridBuffer::new_with_fields(
            basic_info.num_rows,
            basic_info.num_cols,
            u64_data,
            f32_data,
            basic_info.cells,
        ))
    }

    /// Serialize the GridBuffer to base64.
    pub fn to_base64(&self) -> String {
        let bytes = self.to_bytes();
        base64::encode(&bytes)
    }

    /// Serialize the GridBuffer to base64 with BitPacker.
    pub fn to_base64_with_bitpacking<T: BitPacker + GetCompressionType>(&self) -> String {
        let bytes = self.to_bytes_with_bitpacking::<T>();
        base64::encode(&bytes)
    }

    /// Serialize the GridBuffer to bytes with sorted, then to base64.
    pub fn to_base64_with_sorted(&self, sorter: &mut U32Sorter) -> String {
        let bytes = self.to_bytes_with_sorted(sorter);
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

    /// Parse basic info from bytes.
    fn parse_basic_info(bytes: &[u8]) -> Result<(GridBasicInfo, usize)> {
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

        let basic_info = GridBasicInfo {
            version,
            num_rows,
            num_cols,
            total_num_u64_values,
            total_num_f32_values,
            cells,
        };

        Ok((basic_info, pos))
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

    /// Compress the u64 data using bitpacking.
    #[inline]
    fn compress_u64_data_bitpacking<T: BitPacker + GetCompressionType>(
        &self,
        data: &[u64],
    ) -> Vec<u8> {
        let (prefix, middle, suffix) = unsafe { data.align_to::<u32>() };

        compress_bitpacker::<T>(middle)
    }

    /// Compress the u64 data using bitpacking4x.
    #[inline]
    fn compress_u64_data_bitpacking4x(&self, data: &[u64]) -> Vec<u8> {
        let (prefix, middle, suffix) = unsafe { data.align_to::<u32>() };

        compress_bitpacker4x(middle)
    }

    /// Compress the u64 data using bitpacking8x.
    #[inline]
    fn compress_u64_data_bitpacking8x(&self, data: &[u64]) -> Vec<u8> {
        let (prefix, middle, suffix) = unsafe { data.align_to::<u32>() };

        compress_bitpacker8x(middle)
    }

    #[inline]
    fn compress_u32_data(&self, data: &[u32]) -> Vec<u8> {
        compress_bitpacker4x(data)
    }

    /// Compress u64 data with bitpacking4x and differential coding.
    #[inline]
    fn compress_u64_data_with_bitpacking4x_sorted(&self, data: &[u32]) -> Vec<u8> {
        compress_bitpacker4x_sorted(data)
    }

    /// Compress f32 in the future.
    #[inline]
    fn compress_f32_data<'a>(&self, data: &'a [f32]) -> &'a [u8] {
        let (prefix, middle, suffix) = unsafe { data.align_to::<u8>() };

        middle
    }

    /// Sort the rows by function by `f`.
    fn sort_rows_by<F, T>(&mut self, f: F)
    where
        F: Fn(&Vec<GridCell>) -> T,
        T: Ord,
    {
        self.cells.sort_by_key(|k| f(k));
    }
}

/// GridCell.
///
/// Meta data for each cell.
#[derive(Clone, Default)]
pub struct GridCellU64 {
    pub range: Range<usize>,
}

impl GridCellU64 {
    /// Clear the cell.
    #[inline]
    pub fn clear(&mut self) {
        self.range = 0..0;
    }
}

#[derive(Clone, Default)]
pub struct GridCellF32 {
    pub range: Range<usize>,
}

impl GridCellF32 {
    /// Clear the cell.
    #[inline]
    pub fn clear(&mut self) {
        self.range = 0..0;
    }
}

/// Use enum to represent different types of cells.
#[derive(Clone)]
pub enum GridCell {
    /// No data type.
    None,

    /// U64 data type.
    U64Cell(GridCellU64),

    /// F32 data type.
    F32Cell(GridCellF32),
}

impl Default for GridCell {
    fn default() -> Self {
        Self::None
    }
}

impl GridCell {
    /// Get the range of the cell.
    pub fn range(&self) -> Range<usize> {
        match self {
            GridCell::None => 0..0,
            GridCell::U64Cell(cell) => cell.range.clone(),
            GridCell::F32Cell(cell) => cell.range.clone(),
        }
    }

    /// Get the data type of the cell.
    pub fn data_type(&self) -> GridDataType {
        match self {
            GridCell::None => GridDataType::None,
            GridCell::U64Cell(_) => GridDataType::U64,
            GridCell::F32Cell(_) => GridDataType::F32,
        }
    }

    /// Clear the cell.
    #[inline]
    pub fn clear(&mut self) {
        match self {
            GridCell::None => (),
            GridCell::U64Cell(cell) => cell.clear(),
            GridCell::F32Cell(cell) => cell.clear(),
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

        let buffer = GridBuffer::new_with_num_rows_cols(num_rows, num_cols);
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

        let mut buffer = GridBuffer::new_with_num_rows_cols(num_rows, num_cols);

        let u64_values = vec![1, 2, 3];
        let f32_values = vec![1.0, 2.0, 3.0];

        buffer.push_u64_values(0, 0, &u64_values);
        buffer.push_f32_values(1, 1, &f32_values);

        assert_eq!(buffer.get_u64_values(0, 0), u64_values.as_slice());
        assert_eq!(buffer.get_f32_values(1, 1), f32_values.as_slice());
        assert_eq!(buffer.total_num_u64_values(), 3);
        assert_eq!(buffer.total_num_f32_values(), 3);
    }

    #[test]
    fn test_gridbuffer_set_and_get_f32() {
        setup_log();

        let num_rows = 2;
        let num_cols = 2;

        let mut buffer = GridBuffer::new_with_num_rows_cols(num_rows, num_cols);

        let f32_values = vec![1.0, 2.0, 3.0];
        let f32_values2 = vec![4.0, 5.0];

        buffer.push_f32_values(0, 1, &f32_values);
        buffer.push_f32_values(1, 0, &f32_values2);

        assert_eq!(buffer.get_f32_values(0, 1), f32_values.as_slice());
        assert_eq!(buffer.get_f32_values(1, 0), f32_values2.as_slice());
        assert_eq!(buffer.total_num_f32_values(), 5);
    }

    #[test]
    fn test_gridbuffer_get_cell_num_values() {
        setup_log();

        let num_rows = 2;
        let num_cols = 2;

        let mut buffer = GridBuffer::new_with_num_rows_cols(num_rows, num_cols);

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
        let buffer = GridBuffer::new_with_num_rows_cols(2, 2);
        assert_eq!(buffer.get_u64_values(2, 0), &[]);
    }

    #[test]
    fn test_gridbuffer_out_of_bounds_col() {
        let buffer = GridBuffer::new_with_num_rows_cols(2, 2);
        assert_eq!(buffer.get_f32_values(0, 2), &[]);
    }

    #[test]
    fn test_gridbuffer_to_bytes() {
        setup_log();

        let num_rows = 2;
        let num_cols = 2;

        let mut buffer = GridBuffer::new_with_num_rows_cols(num_rows, num_cols);

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

        assert_eq!(restored_buffer.get_u64_values(0, 0), u64_values.as_slice());
        assert_eq!(restored_buffer.get_f32_values(1, 1), f32_values.as_slice());

        // Check if empty cells are still empty
        assert_eq!(restored_buffer.get_u64_values(0, 1), &[]);
        assert_eq!(restored_buffer.get_f32_values(1, 0), &[]);

        // Check total number of values
        assert_eq!(restored_buffer.total_num_u64_values(), 3);
        assert_eq!(restored_buffer.total_num_f32_values(), 3);
    }

    #[test]
    fn test_gridbuffer_to_bytes_empty() {
        setup_log();

        let buffer = GridBuffer::new_with_num_rows_cols(2, 2);

        let bytes = buffer.to_bytes();

        let restored_buffer = GridBuffer::from_bytes(&bytes).unwrap();

        assert_eq!(restored_buffer.num_rows(), 2);
        assert_eq!(restored_buffer.num_cols(), 2);
        assert_eq!(restored_buffer.total_num_u64_values(), 0);
        assert_eq!(restored_buffer.total_num_f32_values(), 0);
    }

    #[test]
    fn test_gridbuffer_to_bytes_sorted_and_from_bytes_sorted() {
        setup_log();

        let mut sorter = U32Sorter::new();

        let num_rows = 2;
        let num_cols = 2;

        let mut buffer = GridBuffer::new_with_num_rows_cols(num_rows, num_cols);

        // Add some u64 values in unsorted order
        let u64_values = vec![100, 1, 50, 2];
        buffer.push_u64_values(0, 0, &u64_values);

        // Add some f32 values
        let f32_values = vec![1.0, 2.0, 3.0];
        buffer.push_f32_values(1, 1, &f32_values);

        // Convert to bytes using sorted method
        let bytes = buffer.to_bytes_with_sorted(&mut sorter);

        // Create a new GridBuffer from the sorted bytes
        let restored_buffer = GridBuffer::from_bytes(&bytes).unwrap();

        // Check if the restored buffer matches the original
        assert_eq!(restored_buffer.num_rows(), num_rows);
        assert_eq!(restored_buffer.num_cols(), num_cols);

        // Check if the u64 values are correctly restored (should be sorted)
        let restored_u64_values = restored_buffer.get_u64_values(0, 0);
        assert_eq!(restored_u64_values, &[100, 1, 50, 2]);

        // Check if the f32 values are correctly restored
        assert_eq!(restored_buffer.get_f32_values(1, 1), f32_values.as_slice());

        // Check if empty cells are still empty
        assert_eq!(restored_buffer.get_u64_values(0, 1), &[]);
        assert_eq!(restored_buffer.get_f32_values(1, 0), &[]);

        // Check total number of values
        assert_eq!(restored_buffer.total_num_u64_values(), 4);
        assert_eq!(restored_buffer.total_num_f32_values(), 3);
    }

    #[test]
    fn test_gridbuffer_to_bytes_sorted_empty() {
        setup_log();

        let buffer = GridBuffer::new_with_num_rows_cols(2, 2);

        let mut sorter = U32Sorter::new();

        let bytes = buffer.to_bytes_with_sorted(&mut sorter);

        let restored_buffer = GridBuffer::from_bytes(&bytes).unwrap();

        assert_eq!(restored_buffer.num_rows(), 2);
        assert_eq!(restored_buffer.num_cols(), 2);
        assert_eq!(restored_buffer.total_num_u64_values(), 0);
        assert_eq!(restored_buffer.total_num_f32_values(), 0);
    }
}
