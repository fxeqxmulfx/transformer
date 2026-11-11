//! Frozen paired timing protocol. All fitting starts from fresh state.

use convex_channel_forward::model::{Model, Shape, dense_predict};
use convex_channels::{channel::Channel, task};
use convex_recall_compare::data::Sample;
use serde_json::{Value, json};
use std::{hint::black_box, time::Instant};

pub const WARMUP: usize = 5;
pub const TRIALS: usize = 31;
pub const FIT_REPEATS: usize = 64;
pub const PREDICT_REPEATS: usize = 128;
pub const ENDPOINT: u64 = 1000;

pub fn shape() -> Shape {
    Shape {
        queries: task::KEYS,
        keys: task::KEYS,
        values: task::VALUES,
        classes: task::CLASSES,
    }
}

pub fn protocol() -> Value {
    json!({"id":"mqar-forward-v1","seeds":[0,1,2],"tasks":["mqar","recoded"],
        "keys":task::KEYS,"values":task::VALUES,"classes":task::CLASSES,
        "train_dictionaries":task::TRAIN,"test_dictionaries":task::TEST,
        "baseline_steps":1,"endpoint_steps":ENDPOINT,"warmup_trials":WARMUP,
        "measured_trials":TRIALS,"fits_per_trial":FIT_REPEATS,
        "prediction_repeats_per_trial":PREDICT_REPEATS,"default_class":0,
        "frozen_protocol_commit":"071a5a49"})
}

pub fn fit_fw(samples: &[Sample], steps: u64) -> Channel {
    let s = shape();
    let mut channel = Channel::new(s.queries, s.keys, s.values, s.classes);
    let references: Vec<_> = samples.iter().collect();
    for step in 1..=steps {
        channel.step(&references, step);
    }
    channel
}

pub fn distribution(raw: &[f64]) -> Value {
    assert!(!raw.is_empty() && raw.iter().all(|x| x.is_finite() && *x > 0.0));
    let mut sorted = raw.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |percent: usize| sorted[(sorted.len() - 1) * percent / 100] * 1e6;
    json!({"seconds":raw,"median_us":at(50),"p10_us":at(10),"p90_us":at(90),
        "min_us":sorted[0]*1e6,"max_us":sorted[sorted.len()-1]*1e6})
}

fn timed_fit(samples: &[Sample], direct: bool) -> f64 {
    let start = Instant::now();
    for _ in 0..FIT_REPEATS {
        if direct {
            black_box(Model::fit(black_box(shape()), black_box(samples), 0).unwrap());
        } else {
            black_box(fit_fw(black_box(samples), black_box(1)));
        }
    }
    start.elapsed().as_secs_f64() / FIT_REPEATS as f64
}

fn timed_predict(samples: &[Sample], dense: &Channel, compact: &Model, arm: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..PREDICT_REPEATS {
        for s in black_box(samples) {
            if arm == 0 {
                black_box(dense_predict(black_box(dense), black_box(s)));
            } else {
                black_box(black_box(compact).predict(black_box(&s.values), black_box(&s.queries)));
            }
        }
    }
    start.elapsed().as_secs_f64() / (PREDICT_REPEATS * samples.len()) as f64
}

pub fn measure(train: &[Sample], test: &[Sample], direct: &Model, fw: &Channel) -> Value {
    let compiled = Model::compact(fw).expect("one-step baseline is not an exact vertex");
    let mut fit = [Vec::new(), Vec::new()];
    let mut prediction = [Vec::new(), Vec::new(), Vec::new()];
    for trial in 0..WARMUP + TRIALS {
        for offset in 0..2 {
            let arm = (trial + offset) % 2;
            let seconds = timed_fit(train, arm == 0);
            if trial >= WARMUP {
                fit[arm].push(seconds);
            }
        }
        for offset in 0..3 {
            let arm = (trial + offset) % 3;
            let model = if arm == 1 { direct } else { &compiled };
            let seconds = timed_predict(test, fw, model, arm);
            if trial >= WARMUP {
                prediction[arm].push(seconds);
            }
        }
    }
    json!({"fit":{"direct":distribution(&fit[0]),"fw_one":distribution(&fit[1])},
        "prediction_per_dictionary":{"fw_dense":distribution(&prediction[0]),
            "direct":distribution(&prediction[1]),"fw_compact":distribution(&prediction[2])}})
}

pub fn validate_timing(timing: &Value) {
    for group in ["fit", "prediction_per_dictionary"] {
        let arms: &[&str] = if group == "fit" {
            &["direct", "fw_one"]
        } else {
            &["fw_dense", "direct", "fw_compact"]
        };
        for &arm in arms {
            let saved = &timing[group][arm];
            let raw: Vec<_> = saved["seconds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            assert_eq!(raw.len(), TRIALS);
            crate::checks::equivalent(saved, &distribution(&raw));
        }
    }
}
