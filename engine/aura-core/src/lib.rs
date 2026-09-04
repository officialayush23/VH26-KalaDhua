//! AURA — an adaptive, utility- and runtime-aware cache.
//!
//! The crate is split so that the *decision* logic and the *storage* logic never bleed into
//! each other: the policies and the engine decide what should happen to an object, the
//! store carries it out. That separation is what lets the same policy code run inside the
//! discrete-event simulator and inside the live server, instead of a second implementation
//! quietly drifting away from the first.
//!
//! Module map:
//!
//! - [`types`], [`config`] — the vocabulary and the numbers, shared with the trainer.
//! - [`sketch`], [`list`], [`rng`] — the primitives the hot path is built from.
//! - [`features`] — the 16-feature builder, the twin of `training/aura_train/features.py`.
//! - [`policies`] — every baseline, each a complete cache, plus the offline oracle.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod config;
pub mod features;
pub mod list;
pub mod policies;
pub mod rng;
pub mod sketch;
pub mod types;

pub use config::Config;
pub use policies::{CachePolicy, Request as CacheRequest};
pub use types::{Action, CostVector, Decision, KeyId, Layer, ObjectContext, Outcome, SlaClass};
