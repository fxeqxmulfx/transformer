//! Frozen eight-arm timing protocol. Each measurement starts from empty state.

use convex_channel_forward::model::{Model, Shape};
use convex_channel_minibatch::streaming::Streaming;
use convex_channels::task;
use convex_recall_compare::data::Sample;
use serde_json::{Value, json};
use std::{hint::black_box, time::Instant};

pub const SIZES: [usize; 7] = [1, 4, 7, 16, 32, 64, 256];
const WARMUP: usize = 5;
pub const TRIALS: usize = 31;
const REPEATS: usize = 64;

pub fn shape() -> Shape {
    Shape {
        queries: task::KEYS,
        keys: task::KEYS,
        values: task::VALUES,
        classes: task::CLASSES,
    }
}

pub fn protocol() -> Value {
    json!({"id":"mqar-minibatch-v1","tasks":["mqar","recoded"],"seeds":[0,1,2],
        "batch_dictionaries":SIZES,"keys":task::KEYS,"values":task::VALUES,
        "classes":task::CLASSES,"train_dictionaries":task::TRAIN,"test_dictionaries":task::TEST,
        "warmup_trials":WARMUP,"measured_trials":TRIALS,"fits_per_trial":REPEATS,
        "default_class":0,"timed_snapshots_per_fit":1,"arm_order":"rotated"})
}

pub fn fit(samples: &[Sample], batch_size: usize) -> (Model, usize, usize) {
    let mut state = Streaming::new(shape(), 0);
    for batch in samples.chunks(batch_size) {
        state.update_batch(batch);
    }
    (
        state.snapshot().expect("query has no consistent key"),
        state.candidate_visits(),
        state.state_payload_bytes(),
    )
}

pub fn distribution(raw: &[f64]) -> Value {
    assert_eq!(raw.len(), TRIALS);
    assert!(raw.iter().all(|v| v.is_finite() && *v > 0.0));
    let mut sorted = raw.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |p: usize| sorted[(sorted.len() - 1) * p / 100] * 1e6;
    json!({"seconds":raw,"median_us":at(50),"p10_us":at(10),"p90_us":at(90)})
}

pub fn measure(samples: &[Sample]) -> Vec<Value> {
    let mut raw = vec![Vec::new(); SIZES.len() + 1];
    for trial in 0..WARMUP + TRIALS {
        for offset in 0..raw.len() {
            let arm = (trial + offset) % raw.len();
            let start = Instant::now();
            for _ in 0..REPEATS {
                if arm == 0 {
                    black_box(Model::fit(black_box(shape()), black_box(samples), 0).unwrap());
                } else {
                    black_box(fit(black_box(samples), black_box(SIZES[arm - 1])));
                }
            }
            let elapsed = start.elapsed().as_secs_f64() / REPEATS as f64;
            if trial >= WARMUP {
                raw[arm].push(elapsed);
            }
        }
    }
    raw.iter().map(|row| distribution(row)).collect()
}

pub fn validate_timing(saved: &Value) {
    let raw: Vec<_> = saved["seconds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    crate::checks::equivalent(saved, &distribution(&raw));
}
