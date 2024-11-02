//! Test.
#![feature(portable_simd)]
use anyhow::Result;

use bitpacking::BitPacker;
use bitpacking::BitPacker4x;

use gridbuffer::core::performance::convert_simple_features_to_gridbuffer_file_with_bitpacking4x;
use gridbuffer::core::performance::convert_simple_features_to_gridbuffer_file_with_bitpacking8x;
use gridbuffer::core::performance::convert_simple_features_to_gridbuffer_file_with_sorted;
use gridbuffer::core::performance::read_gridbuffer_from_file;
use gridbuffer::core::performance::read_simple_features_from_file;
use gridbuffer::core::performance::time_convert_simple_features_to_gridbuffer;
use gridbuffer::core::timer::Timer;
use log::info;

use gridbuffer::core::tool::setup_log;

#[test]
fn test_compression() {
    setup_log();

    let my_data: Vec<u32> = vec![
        7, 7, 7, 7, 11, 10, 15, 13, 6, 5, 3, 14, 5, 7, 15, 12, 1, 10, 8, 10, 12, 14, 13, 1, 10, 1,
        1, 10, 4, 15, 12, 1, 2, 0, 8, 5, 14, 5, 2, 4, 1, 6, 14, 13, 5, 10, 10, 1, 6, 4, 1, 12, 1,
        1, 5, 15, 15, 2, 8, 6, 4, 3, 10, 8, 8, 9, 2, 6, 10, 5, 7, 9, 0, 13, 15, 5, 13, 10, 0, 2,
        10, 14, 5, 9, 12, 8, 5, 10, 8, 8, 10, 5, 13, 8, 11, 14, 7, 14, 4, 2, 9, 12, 14, 5, 15, 12,
        0, 12, 13, 3, 13, 5, 4, 15, 9, 8, 9, 3, 3, 3, 1, 12, 0, 6, 11, 11, 12, 4,
    ];

    // Detects if `SSE3` is available on the current computed
    // and uses the best available implementation accordingly.
    let bitpacker = BitPacker4x::new();
    let bitpacker_sorted = BitPacker4x::new();

    // Computes the number of bits used for each integer in the blocks.
    // my_data is assumed to have a len of 128 for `BitPacker4x`.
    let num_bits: u8 = bitpacker.num_bits(&my_data);
    assert_eq!(num_bits, 4);

    info!("block len: {}", BitPacker4x::BLOCK_LEN);

    // The compressed array will take exactly `num_bits * BitPacker4x::BLOCK_LEN / 8`.
    // But it is ok to have an output with a different len as long as it is larger
    // than this.
    let mut compressed = vec![0u8; 4 * BitPacker4x::BLOCK_LEN];

    // Compress returns the len.
    let compressed_len = bitpacker.compress(&my_data, &mut compressed[..], num_bits);

    info!("compressed len: {}", compressed_len);

    assert_eq!(
        (num_bits as usize) * BitPacker4x::BLOCK_LEN / 8,
        compressed_len
    );

    // Decompressing
    let mut decompressed = vec![0u32; BitPacker4x::BLOCK_LEN];
    bitpacker.decompress(
        &compressed[..compressed_len],
        &mut decompressed[..],
        num_bits,
    );

    assert_eq!(&my_data, &decompressed);

    let mut my_data_sorted = my_data.clone();
    my_data_sorted.sort();

    let num_bits_sorted: u8 = bitpacker_sorted.num_bits(&my_data_sorted);
    assert_eq!(num_bits_sorted, 4);
    info!("num_bits_sorted: {}", num_bits_sorted);

    let mut compressed_sorted = vec![0u8; 4 * BitPacker4x::BLOCK_LEN];

    let compressed_len_sorted = bitpacker_sorted.compress_sorted(
        0,
        &my_data_sorted,
        &mut compressed_sorted[..],
        num_bits_sorted,
    );

    info!("compressed_len_sorted: {}", compressed_len_sorted);
}

/// Test SimpleFeatures size.
#[test]
fn test_simple_features_size() -> Result<()> {
    setup_log();

    let simple_features = read_simple_features_from_file("resources/simple_features_head_128.txt")?;

    let mut count = 0;
    for feature in simple_features {
        count += 1;

        info!(
            "feature.sparse_features.len(): {}, feature.dense_features.len(): {}",
            feature.sparse_feature.len(),
            feature.dense_feature.len()
        );
    }

    info!("count: {}", count);

    Ok(())
}

#[test]
fn test_convert_simple_features_to_gridbuffer() -> Result<()> {
    setup_log();

    time_convert_simple_features_to_gridbuffer("resources/simple_features_head_128.txt", 16, 81)?;

    Ok(())
}

#[test]
fn test_convert_simple_features_to_gridbuffer_file_with_bitpacking4x() -> Result<()> {
    setup_log();

    let filename = "resources/simple_features_nohash_96.txt";
    let res_filename = "resources/gridbuffers_nohash_row_16_col_81_bitpacking4x.txt";

    convert_simple_features_to_gridbuffer_file_with_bitpacking4x(
        filename,
        16,
        81,
        res_filename,
        false,
    )?;

    Ok(())
}

#[test]
fn test_convert_simple_features_to_gridbuffer_file_with_bitpacking4x_row_4_col_81() -> Result<()> {
    setup_log();

    let filename = "resources/simple_features_nohash_96.txt";
    let res_filename = "resources/gridbuffers_nohash_row_4_col_81_bitpacking4x.txt";

    convert_simple_features_to_gridbuffer_file_with_bitpacking4x(
        filename,
        4,
        81,
        res_filename,
        false,
    )?;

    Ok(())
}

#[test]
fn test_convert_simple_features_to_gridbuffer_file_with_bitpacking4x_row_1_col_81() -> Result<()> {
    setup_log();

    let filename = "resources/simple_features_nohash_96.txt";
    let res_filename = "resources/gridbuffers_nohash_row_1_col_81_bitpacking4x.txt";

    convert_simple_features_to_gridbuffer_file_with_bitpacking4x(
        filename,
        1,
        81,
        res_filename,
        false,
    )?;

    Ok(())
}

#[test]
fn test_convert_simple_features_to_gridbuffer_file_with_bitpacking8x() -> Result<()> {
    setup_log();

    let filename = "resources/simple_features_nohash_96.txt";
    let res_filename = "resources/gridbuffers_nohash_row_16_col_81_bitpacking8x.txt";

    convert_simple_features_to_gridbuffer_file_with_bitpacking8x(
        filename,
        16,
        81,
        res_filename,
        false,
    )?;

    Ok(())
}

#[test]
fn test_convert_simple_features_to_gridbuffer_file_with_sorted() -> Result<()> {
    setup_log();

    let filename = "resources/simple_features_nohash_96.txt";
    let res_filename = "resources/gridbuffers_nohash_row_16_col_81_sorted.txt";

    convert_simple_features_to_gridbuffer_file_with_sorted(filename, 16, 81, res_filename)?;

    Ok(())
}

/// Timing read simple features from file.
#[test]
fn timing_read_simple_features_from_file() -> Result<()> {
    setup_log();

    let loop_count = 10;
    let mut times = Vec::with_capacity(loop_count);

    let filename = "resources/simple_features_head_128.txt";

    for _ in 0..loop_count {
        let timer = Timer::new("read_simple_features_from_file".to_string());

        let simple_features = read_simple_features_from_file(filename)?;

        let mut _count_sparse = 0;
        for feature in simple_features {
            _count_sparse += feature.sparse_feature.len();
        }

        times.push(timer.get_elapsed_micros());
        timer.print_elapsed();
    }

    info!(
        "read_simple_features_from_file, loop_count: {}, times in micros: {:?}",
        loop_count, times
    );

    Ok(())
}

/// Timing read gridbuffer from file.
#[test]
fn timing_read_gridbuffer_from_file_with_bitpacking4x() -> Result<()> {
    setup_log();

    let loop_count = 10;
    let mut times = Vec::with_capacity(loop_count);

    let filename = "resources/gridbuffers_nohash_row_16_col_81_bitpacking4x.txt";

    for _ in 0..loop_count {
        let timer = Timer::new("read_gridbuffer_from_file_with_bitpacking4x".to_string());

        let gridbuffers = read_gridbuffer_from_file(filename)?;

        let mut _count = 0;
        for gridbuffer in gridbuffers {
            _count += gridbuffer.num_cols() * gridbuffer.num_rows();
        }

        times.push(timer.get_elapsed_micros());
        timer.print_elapsed();
    }

    info!(
        "read_gridbuffer_from_file, loop_count: {}, times in micros: {:?}",
        loop_count, times
    );

    Ok(())
}

#[test]
fn timing_read_gridbuffer_from_file_with_bitpacking8x() -> Result<()> {
    setup_log();

    let loop_count = 10;
    let mut times = Vec::with_capacity(loop_count);

    let filename = "resources/gridbuffers_nohash_row_16_col_81_bitpacking8x.txt";

    for _ in 0..loop_count {
        let timer = Timer::new("read_gridbuffer_from_file_with_bitpacking8x".to_string());

        let gridbuffers = read_gridbuffer_from_file(filename)?;

        let mut _count = 0;
        for gridbuffer in gridbuffers {
            _count += gridbuffer.num_cols() * gridbuffer.num_rows();
        }

        times.push(timer.get_elapsed_micros());
        timer.print_elapsed();
    }

    info!(
        "read_gridbuffer_from_file_with_bitpacking8x, loop_count: {}, times in micros: {:?}",
        loop_count, times
    );

    Ok(())
}
