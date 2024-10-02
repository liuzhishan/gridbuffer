//! SIMD operations.

#![feature(portable_simd)]
use std::arch::x86_64::*;
use std::simd::prelude::*;

use log::info;
