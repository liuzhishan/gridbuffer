//! Core of gridbuffer.
#![feature(portable_simd)]
use std::arch::x86_64::*;

pub mod feature_batcher;
pub mod gridbuffer;
pub mod gridrow;
pub mod performance;
pub mod simd;
pub mod timer;
pub mod tool;
