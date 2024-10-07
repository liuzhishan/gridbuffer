use anyhow::{anyhow, bail, Result};
use bitpacking::{BitPacker, BitPacker4x, BitPacker8x};
use log::{error, info};

use prost::Message;

use std::io::BufWriter;
use std::io::{BufRead, BufReader, Read, Write};

use super::gridbuffer::GetCompressionType;
use super::{gridbuffer::GridBuffer, timer::Timer};
use crate::core::tool::U32Sorter;
use crate::sniper::SimpleFeatures;

/// SimpleFeatures iterator to GridBuffer iterator.
///
/// return iterator for the result.
pub struct SimpleFeaturesBatcher<I: Iterator<Item = SimpleFeatures>> {
    features: I,
    buffer: GridBuffer,
    row: usize,
    num_rows: usize,
    num_cols: usize,
}

impl<I: Iterator<Item = SimpleFeatures>> SimpleFeaturesBatcher<I> {
    pub fn new(features: I, num_rows: usize, num_cols: usize) -> Self {
        Self {
            features,
            buffer: GridBuffer::new_with_num_rows_cols(num_rows, num_cols),
            row: 0,
            num_rows,
            num_cols,
        }
    }
}

impl<I: Iterator<Item = SimpleFeatures>> Iterator for SimpleFeaturesBatcher<I> {
    type Item = GridBuffer;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(features) = self.features.next() {
            let mut col = 0;

            for (i, sparse_feature) in features.sparse_feature.iter().enumerate() {
                // Remove prefix.
                let u64_values = &sparse_feature
                    .values
                    .iter()
                    .map(|x| *x & (1_u64 << 52 - 1))
                    .collect::<Vec<_>>();

                self.buffer.push_u64_values(self.row, col, &u64_values);

                col += 1;
            }

            for (i, dense_feature) in features.dense_feature.iter().enumerate() {
                let f32_values = &dense_feature.values;
                self.buffer.push_f32_values(self.row, col, f32_values);

                col += 1;
            }

            self.row += 1;

            if self.row == self.num_rows {
                let gridbuffer = self.buffer.clone();
                self.buffer.clear();

                self.row = 0;
                return Some(gridbuffer);
            }
        }

        None
    }
}
