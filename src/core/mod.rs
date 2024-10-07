//! Core of gridbuffer.
#![feature(portable_simd)]
use std::arch::x86_64::*;

pub mod gridbuffer;
pub mod performance;
pub mod simd;
pub mod timer;
pub mod tool;
pub mod batcher;