//! Parameter layout and identity of the architecture used for every stress case.
use crate::{engine::Config, head::Head, layout, pair_memory};
use serde_json::{Value, json};

// Historical binary selectors are explicit. Precision and the task never select a model.
pub const RAW: bool = !cfg!(any(
    feature = "pair-v2",
    feature = "prefix-xor",
    feature = "prefix-sum"
));
#[cfg(any(
    all(feature = "bounded-memory-read", feature = "causal-product-input"),
    all(
        any(feature = "bounded-memory-read", feature = "causal-product-input"),
        any(
            feature = "original-stream-mean",
            feature = "prefix-xor",
            feature = "prefix-sum",
            feature = "pair-v2"
        )
    )
))]
compile_error!("choose one architecture per stress binary");
pub const BOUNDED: bool = cfg!(feature = "bounded-memory-read");
pub const INPUT_PRODUCT: bool = cfg!(feature = "causal-product-input");
pub const STREAM_READ: bool =
    RAW && !BOUNDED && !INPUT_PRODUCT && !cfg!(feature = "original-stream-mean");
pub const HEAD: Head = if INPUT_PRODUCT {
    Head::InputProduct
} else if BOUNDED {
    Head::Bounded
} else if STREAM_READ {
    Head::Stream
} else if RAW {
    Head::Original
} else {
    Head::Relations
};
pub const MODEL: &str = if matches!(HEAD, Head::Original) {
    "phasor-original-stream-mean-v1"
} else {
    HEAD.id()
};

pub fn start(cfg: &Config) -> usize {
    layout::score_offset(cfg) + layout::score_count(cfg)
}

pub fn parameters(cfg: &Config) -> usize {
    cfg.head.parameters(cfg)
}

pub fn record(result: &mut Value, cfg: &Config, values: &[f32]) {
    if INPUT_PRODUCT {
        result["input_product_parameters"] = json!(parameters(cfg));
    } else if RAW {
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
    if INPUT_PRODUCT {
        return result["input_product_parameters"] == 4 * crate::head::MEMORY_WIDTH as u64 * width
            && result.get("pair_parameters").is_none()
            && result.get("readout_parameters").is_none();
    }
    if RAW {
        let per_channel = if BOUNDED {
            4 * crate::head::MEMORY_WIDTH as u64
        } else if STREAM_READ {
            4
        } else {
            0
        };
        return result["readout_parameters"] == per_channel * width
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
