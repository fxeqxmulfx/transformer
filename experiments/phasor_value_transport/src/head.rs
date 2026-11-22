//! Output heads share the backbone, the tied vocabulary matrix and its optimizer.
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
            _ => Err(format!("unknown head {name}").into()),
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Original => "complex-phase-score-lerope",
            Self::Stream => "phasor-learned-stream-read-v1",
            Self::Relations => pair_memory::MODEL,
            Self::Transition => "phasor-hidden-transition-32-v1",
            Self::Recurrent => "phasor-hidden-recurrent-32-v1",
            Self::Bounded => "phasor-bounded-memory-32-v1",
        }
    }

    pub fn parameters(self, cfg: &Config) -> usize {
        match self {
            Self::Original => 0,
            Self::Stream => 4 * cfg.width,
            Self::Relations => pair_memory::parameters(cfg),
            Self::Transition | Self::Recurrent | Self::Bounded => 4 * MEMORY_WIDTH * cfg.width,
        }
    }

    pub fn initialize(self, cfg: &Config, seed: u32, parameters: &mut [f32]) {
        match self {
            Self::Original | Self::Stream => {}
            Self::Relations => pair_memory::initialize_gains(cfg, parameters),
            Self::Transition | Self::Recurrent | Self::Bounded => {
                let mut rng = Random::new(seed.wrapping_add(30_000));
                let start = crate::architecture::start(cfg);
                let end = start + self.parameters(cfg);
                let (input, output) =
                    parameters[start..end].split_at_mut(3 * MEMORY_WIDTH * cfg.width);
                let mut initialize = |matrix: &mut [f32], fan_in: usize| {
                    let scale = (fan_in as f64).sqrt().recip();
                    for w in matrix {
                        *w = rng.gauss(0., scale) as f32;
                    }
                };
                initialize(input, cfg.width);
                if self == Self::Bounded {
                    output.fill(0.);
                } else {
                    initialize(output, MEMORY_WIDTH);
                }
            }
        }
    }
}
