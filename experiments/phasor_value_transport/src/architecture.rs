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
pub const SHARED_BILINEAR: bool = cfg!(feature = "shared-bilinear-phase");
pub const SCALAR_PHASE: bool = cfg!(feature = "scalar-phase-bottleneck");
pub const TIED_TOEPLITZ: bool = cfg!(feature = "tied-toeplitz");
pub const TWO_FRAME: bool = cfg!(feature = "two-sided-frame");
pub const TRIPLE_TRANSPORT: bool = HEAD.is_triple();
pub const FOURIER: bool = cfg!(feature = "fourier-transport");
pub const TOEPLITZ: bool = cfg!(feature = "prefix-toeplitz") || TIED_TOEPLITZ;
pub const REFLECTION_PHASE: bool = cfg!(feature = "reflection-phase-read");
pub const SINGLE_REFLECTION: bool = cfg!(feature = "single-reflection-read");
pub const CROSSED_CONTEXT: bool = cfg!(feature = "crossed-context-rotation");
pub const HOUSEHOLDER: bool = cfg!(feature = "householder-memory");
pub const CONTEXT_ROTATION: bool = cfg!(feature = "context-rotation-read") || CROSSED_CONTEXT;
pub const ASSOCIATIVE_REFLECTION: bool = cfg!(feature = "associative-reflection-read");
pub const PERIODIC_REFLECTION: bool = cfg!(feature = "periodic-reflection-read");
pub const PHASE_MEMORY: bool = cfg!(feature = "phase-prefix-read")
    || INPUT_PHASE
    || AFFINE_PHASE
    || BILINEAR_PHASE
    || SHARED_BILINEAR
    || SCALAR_PHASE
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
        + SHARED_BILINEAR as u8
        + SCALAR_PHASE as u8
        + cfg!(feature = "prefix-toeplitz") as u8
        + TIED_TOEPLITZ as u8
        + TWO_FRAME as u8
        + cfg!(feature = "triple-transport") as u8
        + cfg!(feature = "triple-bipartite") as u8
        + cfg!(feature = "triple-lookup-prototypes") as u8
        + cfg!(feature = "triple-phase-read") as u8
        + cfg!(feature = "triple-lookup-phase") as u8
        + cfg!(feature = "triple-peer-phase") as u8
        + cfg!(feature = "triple-witness-phase") as u8
        + cfg!(feature = "triple-iterated-witness") as u8
        + cfg!(feature = "triple-role-witness") as u8
        + cfg!(feature = "triple-graph-witness") as u8
        + cfg!(feature = "triple-unit-graph") as u8
        + cfg!(feature = "triple-shared-graph") as u8
        + cfg!(feature = "triple-current-graph") as u8
        + cfg!(feature = "triple-pair-classes") as u8
        + cfg!(feature = "triple-dense-induction") as u8
        + cfg!(feature = "triple-chart-induction") as u8
        + cfg!(feature = "triple-typed-lookup") as u8
        + cfg!(feature = "triple-context-residual") as u8
        + cfg!(feature = "learned-stream-read") as u8
        + FOURIER as u8
        + REFLECTION_PHASE as u8
        + SINGLE_REFLECTION as u8
        + cfg!(feature = "context-rotation-read") as u8
        + CROSSED_CONTEXT as u8
        + HOUSEHOLDER as u8
        + ASSOCIATIVE_REFLECTION as u8
        + PERIODIC_REFLECTION as u8;
    assert!(
        variants <= 1 && (variants == 0 || RAW && !cfg!(feature = "original-stream-mean")),
        "choose one architecture per stress binary"
    );
};
pub const STREAM_READ: bool =
    RAW && !BOUNDED && !INPUT_PRODUCT && !cfg!(feature = "original-stream-mean");
pub const HEAD: Head = if FOURIER {
    Head::FourierTransport
} else if cfg!(feature = "triple-lookup-prototypes") {
    Head::LookupTripleTransport
} else if cfg!(feature = "triple-phase-read") {
    Head::TriplePhase
} else if cfg!(feature = "triple-lookup-phase") {
    Head::LookupTriplePhase
} else if cfg!(feature = "triple-peer-phase") {
    Head::peer_phase(crate::head::PeerSearch::Aligned)
} else if cfg!(feature = "triple-witness-phase") {
    Head::peer_phase(crate::head::PeerSearch::Witnesses)
} else if cfg!(feature = "triple-iterated-witness") {
    Head::peer_phase(crate::head::PeerSearch::Iterated(None))
} else if cfg!(feature = "triple-unit-graph") {
    Head::PeerTriplePhase(crate::head::PeerRead::UNIT_GRAPH)
} else if cfg!(feature = "triple-shared-graph") {
    Head::PeerTriplePhase(crate::head::PeerRead::SHARED_GRAPH)
} else if cfg!(feature = "triple-current-graph") {
    Head::PeerTriplePhase(crate::head::PeerRead::CURRENT_GRAPH)
} else if cfg!(feature = "triple-pair-classes") {
    Head::PeerTriplePhase(crate::head::PeerRead::PAIR_GRAPH)
} else if cfg!(feature = "triple-chart-induction") {
    Head::PeerTriplePhase(crate::head::PeerRead::CHART_INDUCTION)
} else if cfg!(feature = "triple-typed-lookup") {
    Head::PeerTriplePhase(crate::head::PeerRead::TYPED_LOOKUP)
} else if cfg!(feature = "triple-dense-induction") {
    Head::PeerTriplePhase(crate::head::PeerRead::DENSE_INDUCTION)
} else if cfg!(feature = "triple-context-residual") {
    Head::PeerTriplePhase(crate::head::PeerRead::CONTEXT_RESIDUAL)
} else if cfg!(feature = "triple-graph-witness") {
    Head::peer_phase(crate::head::PeerSearch::Iterated(Some(
        crate::slot_orbits::RoleEvidence::WITH_GRAPH,
    )))
} else if cfg!(feature = "triple-role-witness") {
    Head::peer_phase(crate::head::PeerSearch::Iterated(Some(
        crate::slot_orbits::RoleEvidence::OBSERVED_ROLES,
    )))
} else if cfg!(feature = "triple-transport") {
    Head::TripleTransport
} else if cfg!(feature = "triple-bipartite") {
    Head::BipartiteTriple
} else if TWO_FRAME {
    Head::TwoFrame
} else if TIED_TOEPLITZ {
    Head::TiedToeplitz
} else if TOEPLITZ {
    Head::PrefixToeplitz
} else if HOUSEHOLDER {
    Head::Householder
} else if CROSSED_CONTEXT {
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
} else if SHARED_BILINEAR {
    Head::SharedBilinear
} else if SCALAR_PHASE {
    Head::ScalarPhase
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
} else if cfg!(feature = "learned-stream-read") {
    Head::Stream
} else if STREAM_READ {
    Head::PeerTriplePhase(crate::head::PeerRead::PAIR_GRAPH)
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
        + if cfg.head.has_stream_memory() || cfg.head.transforms_context() {
            4 * cfg.width
        } else {
            0
        }
}

pub fn parameters(cfg: &Config) -> usize {
    cfg.head.parameters(cfg)
}

pub fn record(result: &mut Value, cfg: &Config, values: &[f32]) {
    if cfg.head.uses_lookup_capacity() {
        result["prototype_slots"] =
            json!(crate::slot_memory::capacity(cfg.head, cfg.width, cfg.vocab));
    }
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

pub fn valid_record(result: &Value, width: u64, vocab: Option<usize>) -> bool {
    if INPUT_PRODUCT {
        return result["input_product_parameters"] == 4 * crate::head::MEMORY_WIDTH as u64 * width
            && result.get("pair_parameters").is_none()
            && result.get("readout_parameters").is_none();
    }
    if RAW {
        if FOURIER {
            return result["readout_parameters"] == crate::fourier::parameters(width as usize)
                && result.get("pair_parameters").is_none();
        }
        if TRIPLE_TRANSPORT {
            let slots = if HEAD.uses_lookup_capacity() {
                let Some(vocab) = vocab else { return false };
                let slots = crate::slot_memory::capacity(HEAD, width as usize, vocab);
                if result["prototype_slots"] != slots {
                    return false;
                }
                slots
            } else {
                width.min(128) as usize
            };
            return result["readout_parameters"]
                == crate::slot_memory::parameter_count(HEAD, width as usize, slots)
                && result.get("pair_parameters").is_none();
        }
        if TWO_FRAME {
            return result["readout_parameters"] == (2 * width + 8) * width
                && result.get("pair_parameters").is_none();
        }
        if TOEPLITZ {
            return result["readout_parameters"]
                == crate::toeplitz::parameter_count(width as usize, TIED_TOEPLITZ)
                && result.get("pair_parameters").is_none();
        }
        if HOUSEHOLDER {
            return result["readout_parameters"]
                == crate::householder::parameter_count(width as usize)
                && result.get("pair_parameters").is_none();
        }
        let per_channel = if ASSOCIATIVE_REFLECTION {
            4 + HEAD.phase_inputs() as u64
                + 2 * crate::head::MEMORY_WIDTH as u64
                + (crate::associative::PROJECTED + crate::associative::VALUES) as u64
        } else if CONTEXT_ROTATION {
            4 + HEAD.context_stages() as u64 * (width / 2)
        } else if PHASE_MEMORY {
            4 + HEAD.phase_inputs() as u64
                + 2 * crate::head::MEMORY_WIDTH as u64
                + u64::from(SCALAR_PHASE)
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
                    + (HEAD.phase_gates()
                        + HEAD.activation_parameters()
                        + HEAD.frequency_parameters()) as u64
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
