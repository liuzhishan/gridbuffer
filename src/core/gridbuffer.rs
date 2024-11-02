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
use anyhow::{bail, Result};
use bitpacking::{BitPacker4x, BitPacker8x};
use likely_stable::unlikely;
use log::error;
use std::ops::Range;
use std::slice;
use strum::{Display, EnumCount, EnumDiscriminants, EnumString, FromRepr};

use dashmap::DashMap;

use crate::core::tool::check_alignment_result;

use base64::{engine::general_purpose, Engine as _};
use bitpacking::BitPacker;

use crate::error_bail;
use crate::sniper::SimpleFeatures;

use super::{
    simd::{
        compress_bitpacker, compress_bitpacker4x, compress_bitpacker4x_sorted,
        compress_bitpacker8x, decompress_bitpacker,
    },
    tool::{check_compression_type, check_data_length, check_range, gxhash32_u32_slice, U32Sorter},
};

/// The data type of grid buffer.
#[derive(
    Default, Clone, FromRepr, Debug, PartialEq, EnumCount, EnumDiscriminants, EnumString, Display,
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
    Default, Clone, FromRepr, Debug, PartialEq, EnumCount, EnumDiscriminants, EnumString, Display,
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

/// The basic information of the grid buffer.
pub struct GridBasicInfo {
    pub version: u8,
    pub num_rows: usize,
    pub num_cols: usize,
    pub total_num_u64_values: usize,
    pub total_num_f32_values: usize,
    pub col_ids: Vec<u32>,
    pub col_ids_hash: u32,
    pub cells: Vec<Vec<GridCell>>,
}

/// GridBuffer format.
///
/// Use `simd` to compress the `u64_data`.
///
/// For storage efficiency, we use `u32` to represent the `String` of column name.
/// The bigger the `num_rows` is, the more efficient the storage will be.
///
/// We support two ways of access the data:
/// 1. access by column index. It's faster, but you need to known which column index you want to access.
/// 2. access by column id. It's more flexible, but slower than access by column index. It needs to build column index first.
///     And when accessing the data, the column id need to be translated to column index by `col_index` mapping.
///
/// For the two accessing ways, we provide two `from_bytes` methods.
/// 1. `from_bytes` is used for access by column index. It only restore the data, and do not build `col_index` mapping.
/// 2. `from_bytes_with_col_index` is used for access by column id. It will restore the data, and build `col_index` mapping.
///
/// Choose the right way based on your actual usage.
///
/// TODO: Too much redundancy for get values and push values. Refactor to macros.
#[derive(Clone)]
pub struct GridBuffer {
    /// The version of the grid buffer.
    version: u8,

    /// The number of rows in the grid buffer.
    num_rows: usize,

    /// The number of columns in the grid buffer.
    num_cols: usize,

    /// The underlying u64 data in the grid buffer.
    u64_data: Vec<u64>,

    /// The underlying f32 data in the grid buffer.
    f32_data: Vec<f32>,

    /// The column id. Each id is an mapping of the column name.
    col_ids: Vec<u32>,

    /// For fast checking col ids, we use `gxhash` to hash the col ids.
    ///
    /// This should be passed as parameter from outside with `col_ids` parameter.
    col_ids_hash: u32,

    /// The column index.
    col_index: DashMap<u32, usize>,

    /// The cells in the grid buffer.
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
            col_ids: vec![],
            col_ids_hash: 0,
            col_index: DashMap::new(),
            cells: vec![],
        }
    }

    /// Construct a new grid buffer.
    ///
    /// To avoid unnecessary copy, the `cols` is not cloned, but moved.
    pub fn new_with_num_rows_col_ids(num_rows: usize, col_ids: Vec<u32>) -> Self {
        let num_cols = col_ids.len();
        let capacity = num_rows * num_cols * 5;

        let col_index = Self::get_col_index(&col_ids);
        let col_ids_hash = gxhash32_u32_slice(&col_ids);

        Self {
            version: GRID_BUFFER_VERSION,
            num_rows,
            num_cols,
            u64_data: Vec::with_capacity(capacity),
            f32_data: Vec::with_capacity(capacity),
            col_ids,
            col_ids_hash,
            col_index,
            cells: vec![vec![GridCell::default(); num_cols]; num_rows],
        }
    }

    pub fn new_with_num_rows_col_ids_hash(
        num_rows: usize,
        col_ids: Vec<u32>,
        col_ids_hash: u32,
    ) -> Self {
        let num_cols = col_ids.len();
        let capacity = num_rows * num_cols * 5;

        let col_index = Self::get_col_index(&col_ids);

        Self {
            version: GRID_BUFFER_VERSION,
            num_rows,
            num_cols,
            u64_data: Vec::with_capacity(capacity),
            f32_data: Vec::with_capacity(capacity),
            col_ids,
            col_ids_hash,
            col_index,
            cells: vec![vec![GridCell::default(); num_cols]; num_rows],
        }
    }

    /// Construct a new grid buffer with all the fields.
    pub fn new_with_fields(
        num_rows: usize,
        num_cols: usize,
        u64_data: Vec<u64>,
        f32_data: Vec<f32>,
        col_ids: Vec<u32>,
        col_ids_hash: u32,
        cells: Vec<Vec<GridCell>>,
    ) -> Self {
        let col_index = Self::get_col_index(&col_ids);

        Self {
            version: GRID_BUFFER_VERSION,
            num_rows,
            num_cols,
            u64_data,
            f32_data,
            col_ids,
            col_ids_hash,
            col_index,
            cells,
        }
    }

    /// Initialize the grid buffer with `num_rows` and `cols`.
    pub fn init(&mut self, num_rows: usize, col_ids: Vec<u32>) {
        self.num_rows = num_rows;
        self.num_cols = col_ids.len();
        self.col_ids_hash = gxhash32_u32_slice(&col_ids);
        self.col_ids = col_ids;
        self.col_index = Self::get_col_index(&self.col_ids);
        self.cells = vec![vec![GridCell::default(); self.num_cols]; self.num_rows];
    }

    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.num_rows > 0 && self.num_cols > 0
    }

    #[inline]
    pub fn col_ids_hash(&self) -> u32 {
        self.col_ids_hash
    }

    #[inline]
    pub fn is_col_ids_hash_equal(&self, col_ids_hash: u32) -> bool {
        self.col_ids_hash == col_ids_hash
    }

    /// Get `col_index` from `cols`.
    #[inline]
    fn get_col_index(col_ids: &Vec<u32>) -> DashMap<u32, usize> {
        col_ids
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, col)| (col, id))
            .collect::<DashMap<_, _>>()
    }

    /// Build `col_index` from `cols`.
    #[inline]
    fn build_col_index(&mut self, col_ids: &Vec<u32>) {
        self.col_index = Self::get_col_index(col_ids);
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

    /// Get the pointer of the u64 data.
    #[inline]
    pub fn u64_values(&self) -> &[u64] {
        &self.u64_data
    }

    /// Get the pointer of the f32 data.
    #[inline]
    pub fn f32_values(&self) -> &[f32] {
        &self.f32_data
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
    pub fn col_ids(&self) -> &Vec<u32> {
        &self.col_ids
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

    #[inline]
    pub fn set_col_ids(&mut self, col_ids: &Vec<u32>) {
        self.col_ids = col_ids.clone();
        self.col_index = Self::get_col_index(&self.col_ids);
    }

    /// Extend rows.
    #[inline]
    pub fn extend_rows(&mut self, num_rows: usize) {
        self.num_rows += num_rows;
        self.cells
            .extend(vec![vec![GridCell::default(); self.num_cols]; num_rows]);
    }

    /// Extend columns.
    ///
    /// Must update the `col_index` after extend columns.
    pub fn extend_cols(&mut self, col_ids: &Vec<u32>) -> Result<()> {
        let n = self.num_cols;

        self.num_cols += col_ids.len();
        self.col_ids.extend_from_slice(col_ids);

        for (i, col) in col_ids.iter().enumerate() {
            if self.col_index.contains_key(col) {
                error_bail!("column id already exists: {}", col);
            }

            self.col_index.insert(col.clone(), n + i);
        }

        for row in &mut self.cells {
            row.extend(vec![GridCell::default(); col_ids.len()]);
        }

        Ok(())
    }

    /// Extend rows and columns.
    #[inline]
    pub fn extend_rows_cols(&mut self, num_rows: usize, col_ids: &Vec<u32>) -> Result<()> {
        self.extend_rows(num_rows);
        self.extend_cols(col_ids)
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

    /// Get the column index by the column id.
    ///
    /// TODO: speed up the search.
    #[inline]
    pub fn get_col_by_id(&self, col_id: u32) -> Option<usize> {
        match self.col_index.get(&col_id) {
            Some(id) => Some(id.value().clone()),
            None => None,
        }
    }

    #[inline]
    pub fn push_cell_by_col_id(&mut self, row: usize, col_id: u32, value: GridCell) -> Result<()> {
        match self.get_col_by_id(col_id) {
            Some(col) => {
                self.push_cell(row, col, value);
                Ok(())
            }
            None => {
                error_bail!("column id not found: {}", col_id);
            }
        }
    }

    #[inline]
    pub fn get_cell(&self, row: usize, col: usize) -> Option<&GridCell> {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return None;
        }

        Some(&self.cells[row][col])
    }

    /// Get the cell by the column id.
    #[inline]
    pub fn get_cell_by_col_id(&self, row: usize, col_id: u32) -> Option<&GridCell> {
        match self.get_col_by_id(col_id) {
            Some(col) => self.get_cell(row, col),
            None => {
                error!("column id not found: {}", col_id);
                None
            }
        }
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

    pub fn push_u64_values_by_col_id(
        &mut self,
        row: usize,
        col_id: u32,
        values: &[u64],
    ) -> Result<()> {
        match self.get_col_by_id(col_id) {
            Some(col) => {
                self.push_u64_values(row, col, values);
                Ok(())
            }
            None => {
                error_bail!("column id not found: {}", col_id);
            }
        }
    }

    #[inline]
    pub fn push_u64(&mut self, row: usize, col: usize, value: u64) {
        self.push_u64_values(row, col, &[value]);
    }

    #[inline]
    pub fn push_u64_by_col_id(&mut self, row: usize, col_id: u32, value: u64) -> Result<()> {
        match self.get_col_by_id(col_id) {
            Some(col) => {
                self.push_u64(row, col, value);
                Ok(())
            }
            None => {
                error_bail!("column id not found: {}", col_id);
            }
        }
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

    pub fn push_f32_values_by_col_id(
        &mut self,
        row: usize,
        col_id: u32,
        values: &[f32],
    ) -> Result<()> {
        match self.get_col_by_id(col_id) {
            Some(col) => {
                self.push_f32_values(row, col, values);
                Ok(())
            }
            None => {
                error_bail!("column id not found: {}", col_id);
            }
        }
    }

    #[inline]
    pub fn push_f32(&mut self, row: usize, col: usize, value: f32) {
        self.push_f32_values(row, col, &[value]);
    }

    #[inline]
    pub fn push_f32_by_col_id(&mut self, row: usize, col_id: u32, value: f32) -> Result<()> {
        match self.get_col_by_id(col_id) {
            Some(col) => {
                self.push_f32(row, col, value);
                Ok(())
            }
            None => {
                error_bail!("column id not found: {}", col_id);
            }
        }
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

    /// Get u64 values from the cell by the column id.
    #[inline]
    pub fn get_u64_values_by_col_id(&self, row: usize, col_id: u32) -> &[u64] {
        match self.get_col_by_id(col_id) {
            Some(col) => self.get_u64_values(row, col),
            None => &[],
        }
    }

    /// Get u64 values from the cell.
    #[inline]
    pub fn get_u64_values_by_cell_u64(&self, cell: &GridCellU64) -> &[u64] {
        if unlikely(cell.range.start >= self.u64_data.len() || cell.range.end > self.u64_data.len())
        {
            error!(
                "out of bounds, range: {:?}, len: {}",
                cell.range,
                self.u64_data.len()
            );
            return &[];
        }

        &self.u64_data[cell.range.clone()]
    }

    /// Get u64 values from the cell.
    #[inline]
    pub fn get_u64_values_by_cell(&self, cell: &GridCell) -> &[u64] {
        match cell {
            GridCell::U64Cell(cell) => self.get_u64_values_by_cell_u64(cell),
            _ => &[],
        }
    }

    /// Get u64 value by row and col.
    #[inline]
    pub fn get_u64(&self, row: usize, col: usize) -> Option<u64> {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return None;
        }

        match &self.cells[row][col] {
            GridCell::U64Cell(cell) => {
                if cell.range.len() == 1 {
                    if cell.range.start >= self.u64_data.len() {
                        error!(
                            "out of bounds, range.start: {}, len: {}",
                            cell.range.start,
                            self.u64_data.len()
                        );
                        return None;
                    }

                    Some(self.u64_data[cell.range.start])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get u64 value by the column id.
    #[inline]
    pub fn get_u64_by_col_id(&self, row: usize, col_id: u32) -> Option<u64> {
        match self.get_col_by_id(col_id) {
            Some(col) => self.get_u64(row, col),
            None => None,
        }
    }

    /// Get u64 value by the cell u64.
    #[inline]
    pub fn get_u64_by_cell_u64(&self, cell: &GridCellU64) -> Option<u64> {
        self.get_u64_values_by_cell_u64(cell).first().cloned()
    }

    /// Get u64 value by the cell.
    #[inline]
    pub fn get_u64_by_cell(&self, cell: &GridCell) -> Option<u64> {
        match cell {
            GridCell::U64Cell(cell) => self.get_u64_by_cell_u64(cell),
            _ => None,
        }
    }

    /// Get mut u64 values by the cell.
    #[inline]
    pub fn get_u64_values_by_cell_mut(&mut self, cell: &GridCell) -> &mut [u64] {
        match cell {
            GridCell::U64Cell(cell) => &mut self.u64_data[cell.range.clone()],
            _ => &mut [],
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

    /// Get f32 values from the cell by the column id.
    #[inline]
    pub fn get_f32_values_by_col_id(&self, row: usize, col_id: u32) -> &[f32] {
        match self.get_col_by_id(col_id) {
            Some(col) => self.get_f32_values(row, col),
            None => &[],
        }
    }

    /// Get f32 values from the cell.
    #[inline]
    pub fn get_f32_values_by_cell_f32(&self, cell: &GridCellF32) -> &[f32] {
        if unlikely(cell.range.start >= self.f32_data.len() || cell.range.end > self.f32_data.len())
        {
            error!(
                "out of bounds, range: {:?}, len: {}",
                cell.range,
                self.f32_data.len()
            );
            return &[];
        }

        &self.f32_data[cell.range.clone()]
    }

    /// Get f32 values from the cell f32.
    #[inline]
    pub fn get_f32_values_by_cell(&self, cell: &GridCell) -> &[f32] {
        match cell {
            GridCell::F32Cell(cell) => self.get_f32_values_by_cell_f32(cell),
            _ => &[],
        }
    }

    /// Get f32 value by row and col.
    #[inline]
    pub fn get_f32(&self, row: usize, col: usize) -> Option<f32> {
        if unlikely(row >= self.num_rows || col >= self.num_cols) {
            error!("out of bounds, row: {}, col: {}", row.clone(), col.clone());
            return None;
        }

        match &self.cells[row][col] {
            GridCell::F32Cell(cell) => {
                if cell.range.len() == 1 {
                    if cell.range.start >= self.f32_data.len() {
                        error!(
                            "out of bounds, range.start: {}, len: {}",
                            cell.range.start,
                            self.f32_data.len()
                        );
                        return None;
                    }

                    Some(self.f32_data[cell.range.start])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get f32 value by the column id.
    #[inline]
    pub fn get_f32_by_col_id(&self, row: usize, col_id: u32) -> Option<f32> {
        match self.get_col_by_id(col_id) {
            Some(col) => self.get_f32(row, col),
            None => None,
        }
    }

    /// Get f32 value by the cell f32.
    #[inline]
    pub fn get_f32_by_cell_f32(&self, cell: &GridCellF32) -> Option<f32> {
        self.get_f32_values_by_cell_f32(cell).first().cloned()
    }

    /// Get f32 value by the cell.
    #[inline]
    pub fn get_f32_by_cell(&self, cell: &GridCell) -> Option<f32> {
        match cell {
            GridCell::F32Cell(cell) => self.get_f32_by_cell_f32(cell),
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
    /// 6. col_ids: Vec<u32>
    /// 7. col_ids_hash: u32
    /// 8. num_rows * num_cols * 2 usize:
    ///      for each cell, it's stored as:
    ///          data_type: u8
    ///          range.start: usize
    ///          range.end: usize
    /// 9. compression_type: u8
    /// 10. compressed_u64_total_num_bytes: usize
    /// 11. num_bits: u8
    /// 12. compressed u64 bytes
    /// 13. compresson_type: u8
    /// 14. compressed_f32_total_num_bytes: usize
    /// 15. compressed f32 bytes
    #[inline]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_bitpacking4x()
    }

    /// Serialize the GridBuffer to bytes with `BitPacking` compression.
    pub fn to_bytes_with_bitpacking<T: BitPacker + GetCompressionType>(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.estimated_bytes());

        self.serialize_basic_info::<T>(&mut buf);

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
        self.serialize_basic_info::<BitPacker4x>(&mut buf);

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
    /// 6. col_ids: Vec<u32>
    /// 7. col_ids_hash: u32
    /// 8. cells: Vec<Vec<GridCell>>
    pub fn serialize_basic_info<T: BitPacker + GetCompressionType>(&self, buf: &mut Vec<u8>) {
        // version
        self.push_u8_le(self.version, buf);

        // num_rows, num_cols
        // Store as u32.
        self.push_usize_le(self.num_rows, buf);
        self.push_usize_le(self.num_cols, buf);

        // total_num_u64_values, total_num_f32_values.
        self.push_usize_le(self.total_num_u64_values(), buf);
        self.push_usize_le(self.total_num_f32_values(), buf);

        // col_ids.
        self.col_ids_to_bytes::<T>(buf);

        // Cells.
        self.cells_to_bytes(buf);
    }

    /// `col_ids` to bytes.
    fn col_ids_to_bytes<T: BitPacker + GetCompressionType>(&self, buf: &mut Vec<u8>) {
        let compressed_col_ids = compress_bitpacker::<T>(self.col_ids.as_slice());

        self.push_u8_le(T::get_compression_type() as u8, buf);
        self.push_usize_le(compressed_col_ids.len(), buf);

        buf.extend_from_slice(compressed_col_ids.as_slice());

        self.push_u32_le(self.col_ids_hash, buf);
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
        let (_prefix, middle, _suffix) = unsafe { self.u64_data.align_to::<u32>() };

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
        start_pos: usize,
        num_rows: usize,
        num_cols: usize,
        total_num_u64_values: usize,
        total_num_f32_values: usize,
    ) -> Result<(Vec<Vec<GridCell>>, usize)> {
        let mut cells = vec![vec![GridCell::default(); num_cols]; num_rows];

        let mut pos = start_pos;

        for row in 0..num_rows {
            for col in 0..num_cols {
                let data_type = match GridDataType::from_repr(bytes[pos]) {
                    Some(data_type) => data_type,
                    None => {
                        error_bail!("invalid data type: {}", bytes[pos]);
                    }
                };

                let range_start = Self::parse_usize_le_unchecked(bytes, pos + 1);
                let range_end = Self::parse_usize_le_unchecked(bytes, pos + 1 + FOUR);

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

                pos += 2 * FOUR + 1;
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

    /// Parse u32 data from bytes.
    ///
    /// Return the `u32_data`position of the next field.
    fn parse_u32_data(
        bytes: &[u8],
        start_pos: usize,
        total_num_u32_values: usize,
    ) -> Result<(Vec<u32>, usize)> {
        // compressed_u32_data_len.
        let compression_type = match CompressionType::from_repr(bytes[start_pos]) {
            Some(compression_type) => compression_type,
            None => {
                error_bail!("invalid compression type: {}", bytes[start_pos]);
            }
        };

        match compression_type {
            CompressionType::None => {
                Self::parse_u32_data_without_compression(bytes, start_pos + 1, total_num_u32_values)
            }
            CompressionType::BitPacking4x => {
                Self::parse_u32_data_with_bitpacking4x(bytes, start_pos + 1, total_num_u32_values)
            }
            CompressionType::BitPacking8x => {
                Self::parse_u32_data_with_bitpacking8x(bytes, start_pos + 1, total_num_u32_values)
            }
            CompressionType::BitPacking4xDiffCoding => {
                Self::parse_u32_data_with_bitpacking4x_sorted(
                    bytes,
                    start_pos + 1,
                    total_num_u32_values,
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

    /// Parse int data without compression.
    fn parse_int_data_without_compression<T: Sized + Clone>(
        bytes: &[u8],
        start_pos: usize,
        total_num_int_values: usize,
    ) -> Result<(Vec<T>, usize)> {
        let mut pos = start_pos;

        let compressed_data_len = Self::parse_usize_le_unchecked(bytes, pos);
        pos += 4;

        let u8_data = bytes[pos..pos + compressed_data_len].to_vec();

        let (prefix, middle, suffix) = unsafe { u8_data.align_to::<T>() };
        check_alignment_result(prefix.len(), suffix.len())?;

        check_data_length(middle.len(), total_num_int_values)?;

        Ok((middle.to_vec(), pos + compressed_data_len))
    }

    /// Parse u64 data without compression.
    fn parse_u64_data_without_compression(
        bytes: &[u8],
        start_pos: usize,
        total_num_u64_values: usize,
    ) -> Result<(Vec<u64>, usize)> {
        Self::parse_int_data_without_compression::<u64>(bytes, start_pos, total_num_u64_values)
    }

    /// Parse u32 data without compression.
    fn parse_u32_data_without_compression(
        bytes: &[u8],
        start_pos: usize,
        total_num_u32_values: usize,
    ) -> Result<(Vec<u32>, usize)> {
        Self::parse_int_data_without_compression::<u32>(bytes, start_pos, total_num_u32_values)
    }

    /// Parse u32 data with bitpacking.
    fn parse_u32_data_with_bitpacking<T: BitPacker + GetCompressionType>(
        bytes: &[u8],
        start_pos: usize,
        compressed_data_len: usize,
        total_num_u32_values: usize,
        is_sorted: bool,
    ) -> Result<(Vec<u32>, usize)> {
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

    /// Parse u32 data with bitpacking8x.
    fn parse_u32_data_with_bitpacking8x(
        bytes: &[u8],
        start_pos: usize,
        total_num_u32_values: usize,
    ) -> Result<(Vec<u32>, usize)> {
        let compressed_data_len = Self::parse_usize_le_unchecked(bytes, start_pos);

        Self::parse_u32_data_with_bitpacking::<BitPacker8x>(
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
        _total_num_f32_values: usize,
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

        let (f32_data, _pos) = Self::parse_f32_data(bytes, pos, basic_info.total_num_f32_values)?;
        check_data_length(f32_data.len(), basic_info.total_num_f32_values)?;

        Ok(GridBuffer::new_with_fields(
            basic_info.num_rows,
            basic_info.num_cols,
            u64_data,
            f32_data,
            basic_info.col_ids.clone(),
            basic_info.col_ids_hash,
            basic_info.cells.clone(),
        ))
    }

    /// Serialize the GridBuffer to base64.
    pub fn to_base64(&self) -> String {
        let bytes = self.to_bytes();
        general_purpose::STANDARD.encode(&bytes)
    }

    /// Serialize the GridBuffer to base64 with BitPacker.
    pub fn to_base64_with_bitpacking<T: BitPacker + GetCompressionType>(&self) -> String {
        let bytes = self.to_bytes_with_bitpacking::<T>();
        general_purpose::STANDARD.encode(&bytes)
    }

    /// Serialize the GridBuffer to bytes with sorted, then to base64.
    pub fn to_base64_with_sorted(&self, sorter: &mut U32Sorter) -> String {
        let bytes = self.to_bytes_with_sorted(sorter);
        general_purpose::STANDARD.encode(&bytes)
    }

    /// Deserialize the GridBuffer from base64.
    pub fn from_base64(base64: &str) -> Result<Self> {
        let bytes = general_purpose::STANDARD.decode(base64)?;
        Self::from_bytes(&bytes)
    }

    /// Helper method to push a `usize` as little-endian bytes
    ///
    /// Store as u32.
    #[inline]
    fn push_usize_le(&self, value: usize, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&(value as u32).to_le_bytes());
    }

    #[inline]
    fn push_u32_le(&self, value: u32, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&value.to_le_bytes());
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

        // col_ids
        let (col_ids, col_ids_hash, pos) = Self::parse_col_ids(bytes, num_cols)?;

        // Cell ranges.
        let (cells, pos) = Self::parse_cells(
            bytes,
            pos,
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
            col_ids,
            col_ids_hash,
            cells,
        };

        Ok((basic_info, pos))
    }

    /// Parse col_ids from bytes.
    ///
    /// Return (col_ids, col_ids_hash, pos).
    #[inline]
    fn parse_col_ids(bytes: &[u8], num_cols: usize) -> Result<(Vec<u32>, u32, usize)> {
        let pos = 4 * FOUR + 1;

        let (col_ids, pos) = Self::parse_u32_data(bytes, pos, num_cols)?;

        let col_ids_hash = Self::parse_u32_le_unchecked(bytes, pos);

        Ok((col_ids, col_ids_hash, pos + FOUR))
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
        let (_prefix, middle, _suffix) = unsafe { data.align_to::<u32>() };

        compress_bitpacker::<T>(middle)
    }

    /// Compress the u64 data using bitpacking4x.
    #[inline]
    fn compress_u64_data_bitpacking4x(&self, data: &[u64]) -> Vec<u8> {
        let (_prefix, middle, _suffix) = unsafe { data.align_to::<u32>() };

        compress_bitpacker4x(middle)
    }

    /// Compress the u64 data using bitpacking8x.
    #[inline]
    fn compress_u64_data_bitpacking8x(&self, data: &[u64]) -> Vec<u8> {
        let (_prefix, middle, _suffix) = unsafe { data.align_to::<u32>() };

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
        let (_prefix, middle, _suffix) = unsafe { data.align_to::<u8>() };

        middle
    }

    /// Sort the rows by function by `f`.
    pub fn sort_rows_by<F, T>(&mut self, f: F)
    where
        F: Fn(&Vec<GridCell>) -> T,
        T: Ord,
    {
        self.cells.sort_by_key(|k| f(k));
    }

    /// Convert `SimpleFeatures` to `GridBuffer` with `1` row.
    ///
    /// The first four columns are preserved for `SampleKey` fields.
    ///
    /// The `sample_key_ids` are global ids for `SampleKey` fields.
    ///
    /// The `feature_ids` are global ids for sparse feature and dense feature names.
    pub fn from_simple_features(
        sample_key_ids: &'static [u32],
        feature_ids: &[u32],
        simple_features: &SimpleFeatures,
    ) -> Self {
        let ids = sample_key_ids
            .iter()
            .chain(feature_ids.iter())
            .cloned()
            .collect::<Vec<_>>();

        let mut res = Self::new_with_num_rows_col_ids(1, ids);

        res.push_u64(0, 0, simple_features.timestamp);
        res.push_u64(0, 1, simple_features.user_id);
        res.push_u64(0, 2, simple_features.item_id);
        res.push_u64(0, 3, simple_features.llsid);

        for (i, sparse_feature) in simple_features.sparse_feature.iter().enumerate() {
            res.push_u64_values(0, 4 + i as usize, &sparse_feature.values);
        }

        let num_sparse_features = simple_features.sparse_feature.len() as usize;
        for (i, dense_feature) in simple_features.dense_feature.iter().enumerate() {
            res.push_f32_values(
                0,
                4 + num_sparse_features + i as usize,
                &dense_feature.values,
            );
        }

        res
    }
}

/// GridCell.
///
/// Meta data for each cell.
#[derive(Clone, Default, Debug)]
pub struct GridCellU64 {
    /// The range of the cell in the u64 data.
    pub range: Range<usize>,
}

impl GridCellU64 {
    /// Clear the cell.
    #[inline]
    pub fn clear(&mut self) {
        self.range = 0..0;
    }

    /// Get the u64 values by the cell.
    ///
    /// Used when sorting.
    ///
    /// The `gridbuffer` is borrow as mutable, but the compare function is borrowing each
    /// cell as immutable. We cannot borrow `gridbuffer` as mutable and immutable the same time.
    /// So we pass the pointer to each cell and the length of the u64 data to this function.
    #[inline]
    pub fn get_u64_values(&self, u64_ptr: *const u64, len: usize) -> &[u64] {
        if unlikely(self.range.start >= len || self.range.end > len) {
            error!("out of bounds, range: {:?}, len: {}", self.range, len);
            return &[];
        }

        unsafe { slice::from_raw_parts(u64_ptr.add(self.range.start), self.range.len()) }
    }
}

#[derive(Clone, Default, Debug)]
pub struct GridCellF32 {
    /// The range of the cell in the f32 data.
    pub range: Range<usize>,
}

impl GridCellF32 {
    /// Clear the cell.
    #[inline]
    pub fn clear(&mut self) {
        self.range = 0..0;
    }

    #[inline]
    pub fn get_f32_values(&self, f32_ptr: *const f32, len: usize) -> &[f32] {
        if unlikely(self.range.start >= len || self.range.end > len) {
            error!("out of bounds, range: {:?}, len: {}", self.range, len);
            return &[];
        }

        unsafe { slice::from_raw_parts(f32_ptr.add(self.range.start), self.range.len()) }
    }
}

/// Use enum to represent different types of cells.
#[derive(Clone, Debug)]
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

    /// Get the u64 values by the cell.
    #[inline]
    pub fn get_u64_values(&self, u64_ptr: *const u64, len: usize) -> &[u64] {
        match self {
            GridCell::U64Cell(cell) => cell.get_u64_values(u64_ptr, len),
            _ => &[],
        }
    }

    /// Get the u64 value by the cell.
    #[inline]
    pub fn get_u64(&self, u64_ptr: *const u64, len: usize) -> Option<u64> {
        match self {
            GridCell::U64Cell(cell) => cell.get_u64_values(u64_ptr, len).first().cloned(),
            _ => None,
        }
    }

    /// Get the f32 values by the cell.
    #[inline]
    pub fn get_f32_values(&self, f32_ptr: *const f32, len: usize) -> &[f32] {
        match self {
            GridCell::F32Cell(cell) => cell.get_f32_values(f32_ptr, len),
            _ => &[],
        }
    }

    /// Get the f32 value by the cell.
    #[inline]
    pub fn get_f32(&self, f32_ptr: *const f32, len: usize) -> Option<f32> {
        match self {
            GridCell::F32Cell(cell) => cell.get_f32_values(f32_ptr, len).first().cloned(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::tool::setup_log;

    use super::*;

    #[test]
    fn test_gridbuffer_creation() {
        setup_log();

        let num_rows = 5;
        let num_cols = 5;

        let col_ids = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();

        let buffer = GridBuffer::new_with_num_rows_col_ids(num_rows, col_ids);
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

        let col_ids = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();

        let mut buffer = GridBuffer::new_with_num_rows_col_ids(num_rows, col_ids);

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

        let col_ids = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();

        let mut buffer = GridBuffer::new_with_num_rows_col_ids(num_rows, col_ids);

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

        let col_ids = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();

        let mut buffer = GridBuffer::new_with_num_rows_col_ids(num_rows, col_ids);

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
        let num_cols = 2;
        let col_ids = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();

        let buffer = GridBuffer::new_with_num_rows_col_ids(2, col_ids);
        assert_eq!(buffer.get_u64_values(2, 0), &[]);
    }

    #[test]
    fn test_gridbuffer_out_of_bounds_col() {
        let num_cols = 2;
        let col_ids = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();

        let buffer = GridBuffer::new_with_num_rows_col_ids(2, col_ids);
        assert_eq!(buffer.get_f32_values(0, 2), &[]);
    }

    #[test]
    fn test_gridbuffer_to_bytes() {
        setup_log();

        let num_rows = 2;
        let num_cols = 2;

        let col_ids = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();

        let mut buffer = GridBuffer::new_with_num_rows_col_ids(num_rows, col_ids);

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

        let num_rows = 2;
        let num_cols = 2;

        let col_ids = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();

        let buffer = GridBuffer::new_with_num_rows_col_ids(num_rows, col_ids);

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

        let col_ids = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();

        let mut buffer = GridBuffer::new_with_num_rows_col_ids(num_rows, col_ids);

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

        let num_rows = 2;
        let num_cols = 2;

        let col_ids = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();

        let buffer = GridBuffer::new_with_num_rows_col_ids(num_rows, col_ids);

        let mut sorter = U32Sorter::new();

        let bytes = buffer.to_bytes_with_sorted(&mut sorter);

        let restored_buffer = GridBuffer::from_bytes(&bytes).unwrap();

        assert_eq!(restored_buffer.num_rows(), num_rows);
        assert_eq!(restored_buffer.num_cols(), num_cols);
        assert_eq!(restored_buffer.total_num_u64_values(), 0);
        assert_eq!(restored_buffer.total_num_f32_values(), 0);
    }
}
