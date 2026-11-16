//! Count actual owned training/selection capacities. Fixed transient reserves
//! cover the pinned complete grammar generator/serializer and its two small
//! seven-power tables; they are not forecasts for decoder/head storage.
use pairs::{Training, selection::Selection};
use std::mem::size_of;

/// The pinned generator has 962806 fixed-size enum entries, a single serialized
/// bank, two geometrically growing partition vectors and bounded state scratch.
/// Reserve 1 GiB before invoking it. No model-sized array is built in this pass.
pub const SELECTION_SCRATCH: u128 = 1024 * 1024 * 1024;
/// Both pinned evaluator power tables have at most1024 seven-usize rows.
/// Include their containers, bounded interpreter stack and hashing scratch.
pub const INTERPRETER_SCRATCH: u128 = 1024 * 1024;
pub fn training(t: &Training) -> u128 {
    size_of::<Training>() as u128 + t.rows.capacity() as u128 * size_of::<[u32; 3]>() as u128
}
pub fn selection(s: &Selection) -> u128 {
    size_of::<Selection>() as u128
        + s.retained.capacity() as u128 * size_of::<usize>() as u128
        + s.conflicts.capacity() as u128 * size_of::<pairs::selection::Conflict>() as u128
        + [
            &s.family,
            &s.family_sha256,
            &s.original_ordered_training_sha256,
            &s.audit.dictionary_sha256,
            &s.audit.program_major_states_sha256,
        ]
        .iter()
        .map(|s| s.capacity() as u128)
        .sum::<u128>()
        + s.audit
            .admission_refusal
            .as_ref()
            .map_or(0, |s| s.capacity() as u128)
}
pub fn vector<T>(v: &Vec<T>) -> u128 {
    size_of::<Vec<T>>() as u128 + v.capacity() as u128 * size_of::<T>() as u128
}
