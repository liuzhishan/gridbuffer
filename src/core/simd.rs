//! SIMD operations.

#![feature(portable_simd)]
use std::arch::x86_64::*;
use std::simd::prelude::*;

use likely_stable::{likely, unlikely};
use log::{error, info};

use bitpacking::BitPacker;
use bitpacking::BitPacker4x;
use bitpacking::BitPacker8x;

/// Compress the data using SIMD.
pub fn compress_bitpacker<T: BitPacker>(bitpacker: &T, is_sorted: bool, data: &[u32]) -> Vec<u8> {
    let block_len = T::BLOCK_LEN;

    // Need to store other info, such as length of each block.
    let estimated_len = data.len() * 4 + data.len() / block_len * 8 + 10;
    let mut compressed = vec![0u8; estimated_len];

    // Is usize possible to overflow?
    let mut pos = 0;
    let mut total_len = 0;

    let mut offset = 0;
    while offset < data.len() {
        if offset + block_len > data.len() {
            break;
        }

        if pos + 5 + block_len > compressed.len() {
            error!("Compressed array is too small");
            break;
        }

        let block = &data[offset..offset + block_len];
        let num_bits = bitpacker.num_bits(block);

        // Compress returns the len.
        let compressed_len = if is_sorted {
            bitpacker.compress_sorted(0, block, &mut compressed[pos + 5..], num_bits)
        } else {
            bitpacker.compress(block, &mut compressed[pos + 5..], num_bits)
        };

        // Store the total number of bytes compressed.
        compressed[pos..pos + 4].copy_from_slice(&(compressed_len as u32).to_le_bytes());

        // Store the num_bits.
        compressed[pos + 4] = num_bits;

        // Move the pos forward.
        pos += compressed_len + 5;

        // Move the offset forward.
        offset += block_len;

        total_len = pos + 1;
    }

    // Process the remaining data.
    // Use num_bits = 0 to indicate that the data is not compressed.
    if offset < data.len() {
        compressed[pos..pos + 4]
            .copy_from_slice(&(((data.len() - offset) * 4) as u32).to_le_bytes());
        compressed[pos + 4] = 0;
        pos += 5;

        let (prefix, middle, suffix) = unsafe { data[offset..].align_to::<u8>() };

        if likely(prefix.len() == 0 && suffix.len() == 0) {
            compressed[pos..pos + middle.len()].copy_from_slice(middle);
            total_len = pos + middle.len();
        } else {
            error!("Data is not aligned");
        }
    }

    compressed.resize(total_len, 0);

    compressed
}

/// Compress the data using SIMD.
pub fn compress_bitpacker4x(data: &[u32]) -> Vec<u8> {
    let bitpacker = BitPacker4x::new();
    compress_bitpacker(&bitpacker, false, data)
}

/// Compress the data using SIMD and differential coding.
pub fn compress_bitpacker4x_sorted(data: &[u32]) -> Vec<u8> {
    let bitpacker = BitPacker4x::new();
    compress_bitpacker(&bitpacker, true, data)
}

/// Decompress the data using SIMD.
pub fn decompress_bitpacker<T: BitPacker>(
    data: &[u8],
    num_bits: u8,
    res: &mut [u32],
    is_sorted: bool,
) -> usize {
    let bitpacker = T::new();

    if is_sorted {
        bitpacker.decompress_sorted(0, &data, res, num_bits)
    } else {
        bitpacker.decompress(&data, res, num_bits)
    }
}

pub fn decompress_bitpacker4x(data: &[u8], num_bits: u8, res: &mut [u32]) -> usize {
    decompress_bitpacker::<BitPacker4x>(data, num_bits, res, false)
}

/// Decompress the data using SIMD.
pub fn decompress_bitpacker8x(data: &[u8], num_bits: u8, res: &mut [u32]) -> usize {
    decompress_bitpacker::<BitPacker8x>(data, num_bits, res, false)
}

/// Decompress the data using SIMD and differential coding.
pub fn decompress_bitpacker4x_sorted(data: &[u8], num_bits: u8, res: &mut [u32]) -> usize {
    decompress_bitpacker::<BitPacker4x>(data, num_bits, res, true)
}

/// Decompress the data using SIMD and differential coding.
pub fn decompress_bitpacker8x_sorted(data: &[u8], num_bits: u8, res: &mut [u32]) -> usize {
    decompress_bitpacker::<BitPacker8x>(data, num_bits, res, true)
}
