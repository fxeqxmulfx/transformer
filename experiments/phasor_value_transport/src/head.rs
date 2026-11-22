//! Model variants share the tied vocabulary matrix and the pinned optimizer.
use crate::{Config, Result, pair_memory, random::Random};

pub const MEMORY_WIDTH: usize = 32;
pub const SHARED_LATENTS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    Original,
    Stream,
    Relations,
    Transition,
    Recurrent,
    Bounded,
    InputProduct,
    CompactStream,
    PhasePrefix,
    InputPhase,
    AffinePhase,
    BilinearPhase,
    SharedBilinear,
    ScalarPhase,
    PrefixToeplitz,
    ReflectionPhase,
    SingleReflection,
    ContextRotation,
    CrossedContext,
    Householder,
    AssociativeReflection,
    PeriodicReflection,
}

impl Head {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "original" => Ok(Self::Original),
            "closed" if pair_memory::MODEL == "phasor-closed-relation-gain-v1" => {
                Ok(Self::Relations)
            }
            "closed" => Err("the closed control requires the default closed-relation build".into()),
            "transition" => Ok(Self::Transition),
            "recurrent" => Ok(Self::Recurrent),
            "bounded" => Ok(Self::Bounded),
            "input-product" => Ok(Self::InputProduct),
            "compact" => Ok(Self::CompactStream),
            "phase" => Ok(Self::PhasePrefix),
            "input-phase" => Ok(Self::InputPhase),
            "affine-phase" => Ok(Self::AffinePhase),
            "bilinear-phase" => Ok(Self::BilinearPhase),
            "shared-bilinear" => Ok(Self::SharedBilinear),
            "scalar-phase" => Ok(Self::ScalarPhase),
            "prefix-toeplitz" => Ok(Self::PrefixToeplitz),
            "reflection-phase" => Ok(Self::ReflectionPhase),
            "single-reflection" => Ok(Self::SingleReflection),
            "context-rotation" => Ok(Self::ContextRotation),
            "crossed-context" => Ok(Self::CrossedContext),
            "householder" => Ok(Self::Householder),
            "associative-reflection" => Ok(Self::AssociativeReflection),
            "periodic-reflection" => Ok(Self::PeriodicReflection),
            _ => Err(format!("unknown head {name}").into()),
        }
    }

    pub const fn id(self) -> &'static str {
        match self {
            Self::Original => "complex-phase-score-lerope",
            Self::Stream => "phasor-learned-stream-read-v1",
            Self::Relations => pair_memory::MODEL,
            Self::Transition => "phasor-hidden-transition-32-v1",
            Self::Recurrent => "phasor-hidden-recurrent-32-v1",
            Self::Bounded => "phasor-bounded-memory-32-v1",
            Self::InputProduct => "phasor-causal-product-input-32-v1",
            Self::CompactStream => "phasor-compact-ffn-stream-v1",
            Self::PhasePrefix => "phasor-learned-phase-prefix-32-v1",
            Self::InputPhase => "phasor-learned-input-phase-32-v1",
            Self::AffinePhase => "phasor-learned-affine-phase-32-v1",
            Self::BilinearPhase => "phasor-learned-bilinear-phase-32-v1",
            Self::SharedBilinear => "phasor-learned-shared-bilinear-8x32-v1",
            Self::ScalarPhase => "phasor-learned-scalar-phase-bottleneck-v1",
            Self::PrefixToeplitz => "phasor-learned-prefix-toeplitz-v1",
            Self::ReflectionPhase => "phasor-learned-reflection-phase-32-v1",
            Self::SingleReflection => "phasor-learned-single-reflection-32-v1",
            Self::ContextRotation => "phasor-learned-context-rotation-v1",
            Self::CrossedContext => "phasor-learned-crossed-context-rotation-v1",
            Self::Householder => "phasor-learned-householder-16x4-v1",
            Self::AssociativeReflection => "phasor-learned-associative-reflection-8x4-v1",
            Self::PeriodicReflection => "phasor-learned-periodic-reflection-32-v1",
        }
    }

    pub const fn is_phase(self) -> bool {
        matches!(
            self,
            Self::PhasePrefix
                | Self::InputPhase
                | Self::AffinePhase
                | Self::BilinearPhase
                | Self::SharedBilinear
                | Self::ScalarPhase
                | Self::ReflectionPhase
                | Self::SingleReflection
                | Self::AssociativeReflection
                | Self::PeriodicReflection
        )
    }

    pub const fn projects_embeddings(self) -> bool {
        matches!(
            self,
            Self::PrefixToeplitz
                | Self::InputPhase
                | Self::AffinePhase
                | Self::BilinearPhase
                | Self::SharedBilinear
                | Self::ScalarPhase
                | Self::ReflectionPhase
                | Self::SingleReflection
                | Self::ContextRotation
                | Self::CrossedContext
                | Self::Householder
                | Self::AssociativeReflection
                | Self::PeriodicReflection
        )
    }

    pub const fn has_stream_memory(self) -> bool {
        self.is_phase() || matches!(self, Self::Householder | Self::PrefixToeplitz)
    }

    pub const fn replaces_context(self) -> bool {
        matches!(self, Self::ScalarPhase | Self::PrefixToeplitz)
    }

    pub const fn phase_inputs(self) -> usize {
        match self {
            Self::AffinePhase => 2 * MEMORY_WIDTH,
            Self::BilinearPhase => 4 * MEMORY_WIDTH,
            Self::SharedBilinear => 4 * SHARED_LATENTS,
            Self::ScalarPhase => 4,
            _ => MEMORY_WIDTH,
        }
    }

    pub const fn phase_gates(self) -> usize {
        if matches!(
            self,
            Self::ReflectionPhase | Self::AssociativeReflection | Self::PeriodicReflection
        ) {
            MEMORY_WIDTH
        } else {
            0
        }
    }

    pub const fn frequency_parameters(self) -> usize {
        match self {
            Self::SharedBilinear => SHARED_LATENTS * MEMORY_WIDTH,
            Self::ScalarPhase => MEMORY_WIDTH,
            _ => 0,
        }
    }

    pub fn parameters(self, cfg: &Config) -> usize {
        match self {
            Self::PeriodicReflection => {
                Self::ReflectionPhase.parameters(cfg) + self.activation_parameters()
            }
            Self::Householder => crate::householder::parameter_count(cfg.width),
            Self::PrefixToeplitz => crate::toeplitz::parameter_count(cfg.width),
            Self::ScalarPhase => (4 + 4 + 2 * MEMORY_WIDTH + 1) * cfg.width + MEMORY_WIDTH,
            Self::ContextRotation | Self::CrossedContext => {
                (4 + self.context_stages() * (cfg.width / 2)) * cfg.width
            }
            Self::AssociativeReflection => {
                Self::ReflectionPhase.parameters(cfg)
                    + (crate::associative::PROJECTED + crate::associative::VALUES) * cfg.width
            }
            Self::Original => 0,
            Self::Stream | Self::CompactStream => 4 * cfg.width,
            Self::PhasePrefix
            | Self::InputPhase
            | Self::AffinePhase
            | Self::BilinearPhase
            | Self::SharedBilinear
            | Self::ReflectionPhase
            | Self::SingleReflection => {
                (4 + self.phase_inputs() + 2 * MEMORY_WIDTH) * cfg.width
                    + self.phase_gates()
                    + self.frequency_parameters()
            }
            Self::Relations => pair_memory::parameters(cfg),
            Self::Transition | Self::Recurrent | Self::Bounded | Self::InputProduct => {
                4 * MEMORY_WIDTH * cfg.width
            }
        }
    }

    pub const fn context_stages(self) -> usize {
        match self {
            Self::ContextRotation => 1,
            Self::CrossedContext => 2,
            _ => 0,
        }
    }

    pub const fn activation_parameters(self) -> usize {
        if matches!(self, Self::PeriodicReflection) {
            2 * MEMORY_WIDTH
        } else {
            0
        }
    }

    pub fn initialize(self, cfg: &Config, seed: u32, parameters: &mut [f32]) {
        match self {
            Self::Original | Self::Stream | Self::CompactStream => {}
            Self::PrefixToeplitz => {
                let d = cfg.width;
                let mut offset = crate::architecture::memory_start(cfg);
                let scale = (d as f64).sqrt().recip();
                let mut rng = Random::new(seed.wrapping_add(30_000));
                for (count, stddev) in [
                    (2 * d * d, scale),
                    (2 * d * d, scale / 2f64.sqrt()),
                    ((2 * d - 1) * d, 1. / d as f64),
                    (2 * d * d, scale),
                    (2 * d * d, scale),
                ] {
                    for w in &mut parameters[offset..offset + count] {
                        *w = rng.gauss(0., stddev) as f32;
                    }
                    offset += count;
                }
            }
            Self::ScalarPhase => {
                let first = crate::architecture::memory_start(cfg);
                let output = first + self.phase_inputs() * cfg.width;
                let frequency = output + 2 * MEMORY_WIDTH * cfg.width;
                let context = frequency + MEMORY_WIDTH;
                let mut rng = Random::new(seed.wrapping_add(30_000));
                for (range, fan_in) in [
                    (first..output, cfg.width),
                    (output..frequency, 2 * MEMORY_WIDTH),
                    (frequency..context, 1),
                    (context..context + cfg.width, cfg.width),
                ] {
                    let scale = (fan_in as f64).sqrt().recip();
                    for w in &mut parameters[range] {
                        *w = rng.gauss(0., scale) as f32;
                    }
                }
            }
            Self::Householder => {
                let first = crate::architecture::memory_start(cfg);
                let output = first + crate::householder::INPUTS * cfg.width;
                let initial = output + crate::householder::FEATURES * cfg.width;
                let mut rng = Random::new(seed.wrapping_add(30_000));
                let scale = (cfg.width as f64).sqrt().recip();
                for w in &mut parameters[first..output] {
                    *w = rng.gauss(0., scale) as f32;
                }
                parameters[output..initial].fill(0.);
                for w in &mut parameters[initial..initial + crate::householder::FEATURES] {
                    *w = rng.gauss(0., 0.5) as f32;
                }
            }
            Self::PeriodicReflection => {
                Self::ReflectionPhase.initialize(cfg, seed, parameters);
                let start = crate::architecture::start(cfg) + Self::ReflectionPhase.parameters(cfg);
                parameters[start..start + MEMORY_WIDTH].fill(1.);
                parameters[start + MEMORY_WIDTH..start + 2 * MEMORY_WIDTH].fill(0.);
            }
            Self::AssociativeReflection => {
                Self::ReflectionPhase.initialize(cfg, seed, parameters);
                let start = crate::architecture::start(cfg) + Self::ReflectionPhase.parameters(cfg);
                let output = start + crate::associative::PROJECTED * cfg.width;
                let scale = (cfg.width as f64).sqrt().recip();
                let mut rng = Random::new(seed.wrapping_add(31_000));
                for w in &mut parameters[start..output] {
                    *w = rng.gauss(0., scale) as f32;
                }
                parameters[output..output + crate::associative::VALUES * cfg.width].fill(0.);
            }
            Self::ContextRotation | Self::CrossedContext => {
                parameters[crate::architecture::memory_start(cfg)
                    ..crate::architecture::start(cfg) + self.parameters(cfg)]
                    .fill(0.);
            }
            Self::Relations => pair_memory::initialize_gains(cfg, parameters),
            Self::Transition
            | Self::Recurrent
            | Self::Bounded
            | Self::InputProduct
            | Self::PhasePrefix
            | Self::InputPhase
            | Self::AffinePhase
            | Self::BilinearPhase
            | Self::SharedBilinear
            | Self::ReflectionPhase
            | Self::SingleReflection => {
                let mut rng = Random::new(seed.wrapping_add(30_000));
                let start = crate::architecture::memory_start(cfg);
                let end = crate::architecture::start(cfg) + self.parameters(cfg);
                let rows = if self.is_phase() {
                    self.phase_inputs()
                } else {
                    3 * MEMORY_WIDTH
                };
                let (input, output) = parameters[start..end].split_at_mut(rows * cfg.width);
                let mut initialize = |matrix: &mut [f32], fan_in: usize| {
                    let scale = (fan_in as f64).sqrt().recip();
                    for w in matrix {
                        *w = rng.gauss(0., scale) as f32;
                    }
                };
                let active_rows = match self {
                    Self::AffinePhase => MEMORY_WIDTH,
                    Self::BilinearPhase => 2 * MEMORY_WIDTH,
                    Self::SharedBilinear => 2 * SHARED_LATENTS,
                    _ => rows,
                };
                let (active, inactive) = input.split_at_mut(active_rows * cfg.width);
                initialize(active, cfg.width);
                inactive.fill(0.);
                if self.is_phase() || matches!(self, Self::Bounded | Self::InputProduct) {
                    output.fill(0.);
                    if self == Self::ReflectionPhase {
                        output[2 * MEMORY_WIDTH * cfg.width..].fill(0.5);
                    } else if self == Self::SharedBilinear {
                        initialize(&mut output[2 * MEMORY_WIDTH * cfg.width..], SHARED_LATENTS);
                    }
                } else {
                    initialize(output, MEMORY_WIDTH);
                }
            }
        }
    }
}
