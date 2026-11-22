//! Parameter layout and identity of the architecture used for every stress case.
use crate::{engine::Config, head::Head, layout, pair_memory};
use serde_json::{Value, json};

// Historical binary selectors are explicit. Precision and the task never select a model.
pub const RAW: bool = !cfg!(any(
    feature = "pair-v2",
    feature = "prefix-xor",
    feature = "prefix-sum"
));
pub const BOUNDED: bool = cfg!(feature = "bounded-memory-read");
pub const INPUT_PRODUCT: bool = cfg!(feature = "causal-product-input");
pub const COMPACT: bool = cfg!(feature = "compact-ffn");
pub const INPUT_PHASE: bool = cfg!(feature = "input-phase-read");
pub const AFFINE_PHASE: bool = cfg!(feature = "affine-phase-read");
pub const BILINEAR_PHASE: bool = cfg!(feature = "bilinear-phase-read");
pub const REFLECTION_PHASE: bool = cfg!(feature = "reflection-phase-read");
pub const SINGLE_REFLECTION: bool = cfg!(feature = "single-reflection-read");
pub const CROSSED_CONTEXT: bool = cfg!(feature = "crossed-context-rotation");
pub const CONTEXT_ROTATION: bool = cfg!(feature = "context-rotation-read") || CROSSED_CONTEXT;
pub const ASSOCIATIVE_REFLECTION: bool = cfg!(feature = "associative-reflection-read");
pub const PERIODIC_REFLECTION: bool = cfg!(feature = "periodic-reflection-read");
pub const PHASE_MEMORY: bool = cfg!(feature = "phase-prefix-read")
    || INPUT_PHASE
    || AFFINE_PHASE
    || BILINEAR_PHASE
    || REFLECTION_PHASE
    || SINGLE_REFLECTION
    || ASSOCIATIVE_REFLECTION
    || PERIODIC_REFLECTION;
const _: () = {
    let variants = BOUNDED as u8
        + INPUT_PRODUCT as u8
        + COMPACT as u8
        + cfg!(feature = "phase-prefix-read") as u8
        + INPUT_PHASE as u8
        + AFFINE_PHASE as u8
        + BILINEAR_PHASE as u8
        + REFLECTION_PHASE as u8
        + SINGLE_REFLECTION as u8
        + cfg!(feature = "context-rotation-read") as u8
        + CROSSED_CONTEXT as u8
        + ASSOCIATIVE_REFLECTION as u8
        + PERIODIC_REFLECTION as u8;
    assert!(
        variants <= 1 && (variants == 0 || RAW && !cfg!(feature = "original-stream-mean")),
        "choose one architecture per stress binary"
    );
};
pub const STREAM_READ: bool =
    RAW && !BOUNDED && !INPUT_PRODUCT && !cfg!(feature = "original-stream-mean");
pub const HEAD: Head = if CROSSED_CONTEXT {
    Head::CrossedContext
} else if PERIODIC_REFLECTION {
    Head::PeriodicReflection
} else if ASSOCIATIVE_REFLECTION {
    Head::AssociativeReflection
} else if CONTEXT_ROTATION {
    Head::ContextRotation
} else if SINGLE_REFLECTION {
    Head::SingleReflection
} else if REFLECTION_PHASE {
    Head::ReflectionPhase
} else if BILINEAR_PHASE {
    Head::BilinearPhase
} else if AFFINE_PHASE {
    Head::AffinePhase
} else if INPUT_PHASE {
    Head::InputPhase
} else if PHASE_MEMORY {
    Head::PhasePrefix
} else if COMPACT {
    Head::CompactStream
} else if INPUT_PRODUCT {
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

/// The phase read follows the original four learned stream coefficients.
/// Other hidden memories project the fixed-mean contextual vectors directly.
pub fn memory_start(cfg: &Config) -> usize {
    start(cfg)
        + if cfg.head.is_phase() || cfg.head.context_stages() > 0 {
            4 * cfg.width
        } else {
            0
        }
}

pub fn parameters(cfg: &Config) -> usize {
    cfg.head.parameters(cfg)
}

pub fn record(result: &mut Value, cfg: &Config, values: &[f32]) {
    if INPUT_PRODUCT {
        result["input_product_parameters"] = json!(parameters(cfg));
    } else if RAW {
        result["readout_parameters"] = json!(parameters(cfg));
        if COMPACT {
            result["ffn_complex_units"] = json!(crate::compact::hidden(cfg) / 2);
        }
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
        let per_channel = if ASSOCIATIVE_REFLECTION {
            4 + HEAD.phase_inputs() as u64
                + 2 * crate::head::MEMORY_WIDTH as u64
                + (crate::associative::PROJECTED + crate::associative::VALUES) as u64
        } else if CONTEXT_ROTATION {
            4 + HEAD.context_stages() as u64 * (width / 2)
        } else if PHASE_MEMORY {
            4 + HEAD.phase_inputs() as u64 + 2 * crate::head::MEMORY_WIDTH as u64
        } else if BOUNDED {
            4 * crate::head::MEMORY_WIDTH as u64
        } else if STREAM_READ {
            4
        } else {
            0
        };
        return (!COMPACT || result["ffn_complex_units"] == width)
            && result["readout_parameters"]
                == per_channel * width
                    + (HEAD.phase_gates() + HEAD.activation_parameters()) as u64
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
