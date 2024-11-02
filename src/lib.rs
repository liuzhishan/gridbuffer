//! Gridbuffer.
#![feature(portable_simd)]
#![allow(dead_code)]

pub mod sniper {
    include!(concat!(env!("OUT_DIR"), "/sniper.rs"));
}

pub mod core;
