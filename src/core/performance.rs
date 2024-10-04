//! Test performance of GridBuffer with protobuf data.
//!
//! Parse protobuf data, store data in GridBuffer, and then serialize to bytes.
//! Compare the performance with protobuf's own serialization and deserialization.
//! Also the size of the serialized data is the same.

use anyhow::{anyhow, bail, Result};
use log::{error, info};

use prost::Message;

use std::io::BufWriter;
use std::io::{BufRead, BufReader, Read, Write};

use super::{gridbuffer::GridBuffer, timer::Timer};
use crate::sniper::SimpleFeatures;
use crate::core::tool::U32Sorter;

/// Parse SimpleFeatures from base64 encoded protobuf data.
pub fn parse_simple_features(data: &str) -> Result<SimpleFeatures> {
    let decoded_data = base64::decode(data)?;
    let simple_features = SimpleFeatures::decode(&*decoded_data)?;
    Ok(simple_features)
}

/// Read SimpleFeatures from a file line by line, return an iterator.
pub fn read_simple_features_from_reader(
    reader: impl BufRead,
) -> impl Iterator<Item = Result<SimpleFeatures>> {
    reader.lines().map(|line| {
        line.map_err(|e| e.into())
            .and_then(|line| parse_simple_features(&line))
    })
}

/// Time parsing SimpleFeatures from a file.
pub fn time_parse_simple_features_from_file(file_path: &str) -> Result<()> {
    let file = std::fs::File::open(file_path)?;
    let reader = BufReader::new(file);

    let timer = Timer::new("parse_simple_features_from_file".to_string());

    let mut count = 0;

    for simple_features in read_simple_features_from_reader(reader) {
        count += 1;
    }

    timer.print_elapsed();

    Ok(())
}

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
            buffer: GridBuffer::new(num_rows, num_cols, 10000),
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

/// Time converting SimpleFeatures to GridBuffer.
pub fn convert_simple_features_to_gridbuffer(
    file_path: &str,
    num_rows: usize,
    num_cols: usize,
) -> Result<SimpleFeaturesBatcher<impl Iterator<Item = SimpleFeatures>>> {
    let file = std::fs::File::open(file_path)?;
    let reader = BufReader::new(file);

    let simple_features = read_simple_features_from_reader(reader)
        .filter(|x| match x {
            Ok(_) => true,
            Err(e) => {
                error!("Error: {}", e);
                false
            }
        })
        .map(|x| x.unwrap());

    Ok(SimpleFeaturesBatcher::new(
        simple_features,
        num_rows,
        num_cols,
    ))
}

/// Read SimpleFeatures from a file.
pub fn read_simple_features_from_file(
    file_path: &str,
) -> Result<impl Iterator<Item = SimpleFeatures>> {
    let file = std::fs::File::open(file_path)?;
    let reader = BufReader::new(file);

    Ok(read_simple_features_from_reader(reader)
        .filter(|x| match x {
            Ok(_) => true,
            Err(e) => {
                error!("Error: {}", e);
                false
            }
        })
        .map(|x| x.unwrap()))
}

/// Time converting simple features to gridbuffer.
pub fn time_convert_simple_features_to_gridbuffer(
    file_path: &str,
    num_rows: usize,
    num_cols: usize,
) -> Result<()> {
    let timer = Timer::new("time_convert_simple_features_to_gridbuffer".to_string());

    let gridbuffers = convert_simple_features_to_gridbuffer(file_path, num_rows, num_cols)?;

    for gridbuffer in gridbuffers {
        info!("bytes len: {}", gridbuffer.estimated_bytes());
    }

    timer.print_elapsed();

    Ok(())
}

/// Save gridbuffers to file.
pub fn save_gridbuffers_to_file(
    gridbuffers: impl Iterator<Item = GridBuffer>,
    file_path: &str,
    is_sorted: bool,
) -> Result<()> {
    let file = std::fs::File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    let mut sorter = U32Sorter::new();

    for gridbuffer in gridbuffers {
        if is_sorted {
            let serialized = gridbuffer.to_base64_with_sorted(&mut sorter);
            writer.write_all(serialized.as_bytes())?;
        } else {
            let serialized = gridbuffer.to_base64();
            writer.write_all(serialized.as_bytes())?;
        }

        writer.write("\n".as_bytes())?;
    }

    Ok(())
}

pub fn convert_simple_features_to_gridbuffer_file(
    file_path: &str,
    num_rows: usize,
    num_cols: usize,
    res_file_path: &str,
    is_sorted: bool,
) -> Result<()> {
    let gridbuffers = convert_simple_features_to_gridbuffer(file_path, num_rows, num_cols)?;
    save_gridbuffers_to_file(gridbuffers, res_file_path, is_sorted)?;

    Ok(())
}

pub fn convert_simple_features_to_gridbuffer_file_without_sorted(
    file_path: &str,
    num_rows: usize,
    num_cols: usize,
    res_file_path: &str,
) -> Result<()> {
    convert_simple_features_to_gridbuffer_file(file_path, num_rows, num_cols, res_file_path, false)
}

pub fn convert_simple_features_to_gridbuffer_file_with_sorted(
    file_path: &str,
    num_rows: usize,
    num_cols: usize,
    res_file_path: &str,
) -> Result<()> {
    convert_simple_features_to_gridbuffer_file(file_path, num_rows, num_cols, res_file_path, true)
}

/// Time serializing GridBuffer to bytes.
fn run_performance_test() {}
