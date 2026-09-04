//! AURA — an adaptive, utility- and runtime-aware cache.
//!
//! The crate is split so that the *decision* logic and the *storage* logic never bleed
//! into each other: [`engine`] decides what should happen to an object, [`store`] carries
//! it out. That separation is what lets the same policy code run inside the discrete-event
//! simulator and inside the live server without a second implementation drifting away from
//! the first.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod config;
pub mod features;
pub mod rng;
pub mod sketch;
pub mod types;

pub use config::Config;
pub use types::{Action, CostVector, Decision, KeyId, Layer, ObjectContext, Outcome, SlaClass};
