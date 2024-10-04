//! Gridbuffer.
#![feature(portable_simd)]
use std::arch::x86_64::*;
use std::simd::prelude::*;

pub mod sniper {
    include!(concat!(env!("OUT_DIR"), "/sniper.rs"));
}

pub mod core;
