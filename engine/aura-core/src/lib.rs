//! AURA — an adaptive, utility- and runtime-aware cache.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod config;
pub mod features;
pub mod rng;
pub mod sketch;
pub mod types;

pub use config::Config;
pub use types::{Action, CostVector, Decision, KeyId, Layer, ObjectContext, Outcome, SlaClass};
