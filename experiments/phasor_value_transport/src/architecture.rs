//! Parameter layout and identity of the architecture used for every stress case.
use crate::{engine::Config, layout, pair_memory};
use serde_json::{Value, json};

// Historical binary selectors are explicit. Precision and the task never select a model.
pub const RAW: bool = !cfg!(any(
    feature = "pair-v2",
    feature = "prefix-xor",
    feature = "prefix-sum"
));
pub const STREAM_READ: bool = RAW && !cfg!(feature = "original-stream-mean");
pub const MODEL: &str = if STREAM_READ {
    "phasor-learned-stream-read-v1"
} else if RAW {
    "phasor-original-stream-mean-v1"
} else {
    pair_memory::MODEL
};

pub fn start(cfg: &Config) -> usize {
    layout::score_offset(cfg) + layout::score_count(cfg)
}

pub fn parameters(cfg: &Config) -> usize {
    if !RAW {
        return pair_memory::parameters(cfg);
    }
    if cfg.shared_values && STREAM_READ {
        4 * cfg.width
    } else {
        0
    }
}

pub fn initialize(cfg: &Config, parameters: &mut [f32]) {
    if !RAW {
        pair_memory::initialize_gains(cfg, parameters);
    }
}

pub fn record(result: &mut Value, cfg: &Config, values: &[f32]) {
    if RAW {
        result["readout_parameters"] = json!(parameters(cfg));
    } else {
        result["pair_parameters"] = json!(parameters(cfg));
        result["initial_gain"] = json!(pair_memory::INITIAL_GAIN);
        result["gain"] = json!(values[pair_memory::gain(cfg)]);
        if pair_memory::LEARN_BONUS {
            result["initial_bonus"] = json!(pair_memory::INITIAL_BONUS);
            result["successor_bonus"] = json!(values[pair_memory::gain(cfg) + 1]);
        }
    }
}

pub fn valid_record(result: &Value, width: u64) -> bool {
    if RAW {
        return result["readout_parameters"] == if STREAM_READ { 4 * width } else { 0 }
            && result.get("pair_parameters").is_none();
    }
    result["initial_gain"] == json!(pair_memory::INITIAL_GAIN)
        && result["pair_parameters"].as_u64().is_some_and(|n| n > 1)
        && result["gain"].as_f64().is_some_and(f64::is_finite)
        && (!pair_memory::LEARN_BONUS
            || (result["initial_bonus"] == json!(pair_memory::INITIAL_BONUS)
                && result["successor_bonus"]
                    .as_f64()
                    .is_some_and(f64::is_finite)))
}
