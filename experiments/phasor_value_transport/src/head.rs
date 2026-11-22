//! Model variants share the tied vocabulary matrix and the pinned optimizer.
use crate::{Config, Result, pair_memory, random::Random};

pub const MEMORY_WIDTH: usize = 32;

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
    ReflectionPhase,
    SingleReflection,
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
            "reflection-phase" => Ok(Self::ReflectionPhase),
            "single-reflection" => Ok(Self::SingleReflection),
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
            Self::ReflectionPhase => "phasor-learned-reflection-phase-32-v1",
            Self::SingleReflection => "phasor-learned-single-reflection-32-v1",
        }
    }

    pub const fn is_phase(self) -> bool {
        matches!(
            self,
            Self::PhasePrefix
                | Self::InputPhase
                | Self::AffinePhase
                | Self::BilinearPhase
                | Self::ReflectionPhase
                | Self::SingleReflection
        )
    }

    pub const fn projects_embeddings(self) -> bool {
        matches!(
            self,
            Self::InputPhase
                | Self::AffinePhase
                | Self::BilinearPhase
                | Self::ReflectionPhase
                | Self::SingleReflection
        )
    }

    pub const fn phase_inputs(self) -> usize {
        match self {
            Self::AffinePhase => 2 * MEMORY_WIDTH,
            Self::BilinearPhase => 4 * MEMORY_WIDTH,
            _ => MEMORY_WIDTH,
        }
    }

    pub const fn phase_gates(self) -> usize {
        if matches!(self, Self::ReflectionPhase) {
            MEMORY_WIDTH
        } else {
            0
        }
    }

    pub fn parameters(self, cfg: &Config) -> usize {
        match self {
            Self::Original => 0,
            Self::Stream | Self::CompactStream => 4 * cfg.width,
            Self::PhasePrefix
            | Self::InputPhase
            | Self::AffinePhase
            | Self::BilinearPhase
            | Self::ReflectionPhase
            | Self::SingleReflection => {
                (4 + self.phase_inputs() + 2 * MEMORY_WIDTH) * cfg.width + self.phase_gates()
            }
            Self::Relations => pair_memory::parameters(cfg),
            Self::Transition | Self::Recurrent | Self::Bounded | Self::InputProduct => {
                4 * MEMORY_WIDTH * cfg.width
            }
        }
    }

    pub fn initialize(self, cfg: &Config, seed: u32, parameters: &mut [f32]) {
        match self {
            Self::Original | Self::Stream | Self::CompactStream => {}
            Self::Relations => pair_memory::initialize_gains(cfg, parameters),
            Self::Transition
            | Self::Recurrent
            | Self::Bounded
            | Self::InputProduct
            | Self::PhasePrefix
            | Self::InputPhase
            | Self::AffinePhase
            | Self::BilinearPhase
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
                    _ => rows,
                };
                let (active, inactive) = input.split_at_mut(active_rows * cfg.width);
                initialize(active, cfg.width);
                inactive.fill(0.);
                if self.is_phase() || matches!(self, Self::Bounded | Self::InputProduct) {
                    output.fill(0.);
                    if self == Self::ReflectionPhase {
                        output[2 * MEMORY_WIDTH * cfg.width..].fill(0.5);
                    }
                } else {
                    initialize(output, MEMORY_WIDTH);
                }
            }
        }
    }
}
