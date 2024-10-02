//! Grid buffer.

#![feature(portable_simd)]
use core::simd::prelude::*;

/// GridBuffer is a format which is specifically designed for storing data, which is used for
/// ad sparse model.
///
/// GridBuffer use `simd` to speedup both compression and IO operations. The data is stored
/// by column-first. And interface is similar to `ndarray`.
pub struct GridBuffer<T> {
    grid: Vec<T>,
    width: usize,
    height: usize,
}
