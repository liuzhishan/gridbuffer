//! Test performance of GridBuffer with protobuf data.
//!
//! Parse protobuf data, store data in GridBuffer, and then serialize to bytes.
//! Compare the performance with protobuf's own serialization and deserialization.
//! Also the size of the serialized data is the same.

use anyhow::Result;
use bitpacking::{BitPacker, BitPacker4x, BitPacker8x};
use log::{error, info};

use prost::Message;

use std::io::BufWriter;
use std::io::{BufRead, BufReader, Write};

use base64::{engine::general_purpose, Engine as _};

use super::gridbuffer::GetCompressionType;
use super::{gridbuffer::GridBuffer, timer::Timer};

use crate::core::feature_batcher::SimpleFeaturesBatcher;
use crate::core::tool::U32Sorter;
use crate::sniper::SimpleFeatures;

/// Parse SimpleFeatures from base64 encoded protobuf data.
pub fn parse_simple_features(data: &str) -> Result<SimpleFeatures> {
    let decoded_data = general_purpose::STANDARD.decode(data)?;
    let simple_features = SimpleFeatures::decode(&*decoded_data)?;
    Ok(simple_features)
}

/// Read SimpleFeatures from line iterator.
pub fn read_simple_features_from_lines(
    lines: impl Iterator<Item = String>,
) -> impl Iterator<Item = Result<SimpleFeatures>> {
    lines.map(|line| parse_simple_features(&line))
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

    let mut _count = 0;

    for _simple_features in read_simple_features_from_reader(reader) {
        _count += 1;
    }

    timer.print_elapsed();

    Ok(())
}

/// Time converting SimpleFeatures to GridBuffer.
pub fn convert_simple_features_to_gridbuffer(
    file_path: &str,
    num_rows: usize,
    _num_cols: usize,
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

    Ok(SimpleFeaturesBatcher::new(simple_features, num_rows))
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

/// Read GridBuffer from a file.
pub fn read_gridbuffer_from_file(file_path: &str) -> Result<impl Iterator<Item = GridBuffer>> {
    let file = std::fs::File::open(file_path)?;
    let reader = BufReader::new(file);

    let res = reader
        .lines()
        .filter(|x| match x {
            Ok(_) => true,
            Err(e) => {
                error!("Read line Error: {}", e);
                false
            }
        })
        .map(|x| GridBuffer::from_base64(&x.unwrap()))
        .filter(|x| match x {
            Ok(_) => true,
            Err(e) => {
                error!("GridBuffer from base64 Error: {}", e);
                false
            }
        })
        .map(|x| x.unwrap());

    Ok(res)
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

/// Save gridbuffers to file using bitpacking.
pub fn save_gridbuffers_to_file_with_bitpacking<T: BitPacker + GetCompressionType>(
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
            let serialized = gridbuffer.to_base64_with_bitpacking::<T>();
            writer.write_all(serialized.as_bytes())?;
        }

        writer.write("\n".as_bytes())?;
    }

    writer.flush()?;

    Ok(())
}

/// Save gridbuffers to file using bitpacking4x.
#[inline]
pub fn save_gridbuffers_to_file_with_bitpacking4x(
    gridbuffers: impl Iterator<Item = GridBuffer>,
    file_path: &str,
    is_sorted: bool,
) -> Result<()> {
    save_gridbuffers_to_file_with_bitpacking::<BitPacker4x>(gridbuffers, file_path, is_sorted)
}

/// Save gridbuffers to file using bitpacking8x.
#[inline]
pub fn save_gridbuffers_to_file_with_bitpacking8x(
    gridbuffers: impl Iterator<Item = GridBuffer>,
    file_path: &str,
    is_sorted: bool,
) -> Result<()> {
    save_gridbuffers_to_file_with_bitpacking::<BitPacker8x>(gridbuffers, file_path, is_sorted)
}

/// Convert simple features to gridbuffer file using bitpacking.
pub fn convert_simple_features_to_gridbuffer_file_with_bitpacking<
    T: BitPacker + GetCompressionType,
>(
    file_path: &str,
    num_rows: usize,
    num_cols: usize,
    res_file_path: &str,
    is_sorted: bool,
) -> Result<()> {
    let gridbuffers = convert_simple_features_to_gridbuffer(file_path, num_rows, num_cols)?;
    save_gridbuffers_to_file_with_bitpacking::<T>(gridbuffers, res_file_path, is_sorted)?;

    Ok(())
}

/// Convert simple features to gridbuffer file using bitpacking4x.
#[inline]
pub fn convert_simple_features_to_gridbuffer_file_with_bitpacking4x(
    file_path: &str,
    num_rows: usize,
    num_cols: usize,
    res_file_path: &str,
    is_sorted: bool,
) -> Result<()> {
    convert_simple_features_to_gridbuffer_file_with_bitpacking::<BitPacker4x>(
        file_path,
        num_rows,
        num_cols,
        res_file_path,
        is_sorted,
    )
}

/// Convert simple features to gridbuffer file using bitpacking8x.
#[inline]
pub fn convert_simple_features_to_gridbuffer_file_with_bitpacking8x(
    file_path: &str,
    num_rows: usize,
    num_cols: usize,
    res_file_path: &str,
    is_sorted: bool,
) -> Result<()> {
    convert_simple_features_to_gridbuffer_file_with_bitpacking::<BitPacker8x>(
        file_path,
        num_rows,
        num_cols,
        res_file_path,
        is_sorted,
    )
}

/// Convert simple features to gridbuffer file without sorting.
#[inline]
pub fn convert_simple_features_to_gridbuffer_file_without_sorted(
    file_path: &str,
    num_rows: usize,
    num_cols: usize,
    res_file_path: &str,
) -> Result<()> {
    convert_simple_features_to_gridbuffer_file_with_bitpacking::<BitPacker4x>(
        file_path,
        num_rows,
        num_cols,
        res_file_path,
        false,
    )
}

/// Convert simple features to gridbuffer file with sorting.
#[inline]
pub fn convert_simple_features_to_gridbuffer_file_with_sorted(
    file_path: &str,
    num_rows: usize,
    num_cols: usize,
    res_file_path: &str,
) -> Result<()> {
    convert_simple_features_to_gridbuffer_file_with_bitpacking::<BitPacker4x>(
        file_path,
        num_rows,
        num_cols,
        res_file_path,
        true,
    )
}

/// Time serializing GridBuffer to bytes.
fn run_performance_test() {}
