//! Model variants share the tied vocabulary matrix and the pinned optimizer.
use crate::{Config, Result, pair_memory, random::Random};

pub const MEMORY_WIDTH: usize = 32;
pub const SHARED_LATENTS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerSearch {
    Aligned,
    Witnesses,
    Iterated(Option<crate::slot_orbits::RoleEvidence>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadGates {
    WidthScaled,
    Unit,
    SharedPrevious,
    SharedCurrent,
    ResidualCurrent,
}

impl ReadGates {
    pub const fn shared(self) -> bool {
        matches!(
            self,
            Self::SharedPrevious | Self::SharedCurrent | Self::ResidualCurrent
        )
    }

    /// Number of affine gates stored in the parameter array.
    pub const fn count(self) -> usize {
        self.reads() - self.first_learned()
    }

    pub const fn reads(self) -> usize {
        if self.shared() { 3 } else { 4 }
    }

    /// A unit context residual precedes the two address-dependent gates.
    pub const fn first_learned(self) -> usize {
        matches!(self, Self::ResidualCurrent) as usize
    }

    pub fn position(self, row: usize, length: usize) -> Option<usize> {
        if matches!(self, Self::SharedCurrent | Self::ResidualCurrent) {
            Some(row)
        } else if row.is_multiple_of(length) {
            None
        } else {
            Some(row - 1)
        }
    }
}

/// Search and gate scales vary independently in fixed-weight controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerRead {
    pub search: Option<PeerSearch>,
    pub gates: ReadGates,
    pub pairs: Option<crate::slot_orbits::PairRead>,
    pub dense_induction: bool,
    pub fibers: Option<crate::slot_orbits::FiberRead>,
    pub path_passes: Option<usize>,
}

impl PeerRead {
    const fn new(search: PeerSearch) -> Self {
        Self {
            search: Some(search),
            gates: ReadGates::WidthScaled,
            pairs: None,
            dense_induction: false,
            fibers: None,
            path_passes: None,
        }
    }

    pub const UNIT_GRAPH: Self = Self {
        search: Some(PeerSearch::Iterated(Some(
            crate::slot_orbits::RoleEvidence::WITH_GRAPH,
        ))),
        gates: ReadGates::Unit,
        pairs: None,
        dense_induction: false,
        fibers: None,
        path_passes: None,
    };

    pub const SHARED_GRAPH: Self = Self {
        gates: ReadGates::SharedPrevious,
        ..Self::UNIT_GRAPH
    };
    pub const CURRENT_GRAPH: Self = Self {
        gates: ReadGates::SharedCurrent,
        ..Self::UNIT_GRAPH
    };

    pub const PAIR_GRAPH: Self = Self {
        pairs: Some(crate::slot_orbits::PairRead::FirstOfTwo),
        ..Self::CURRENT_GRAPH
    };

    pub const CONTEXT_RESIDUAL: Self = Self {
        gates: ReadGates::ResidualCurrent,
        ..Self::PAIR_GRAPH
    };

    pub const DENSE_INDUCTION: Self = Self {
        dense_induction: true,
        ..Self::PAIR_GRAPH
    };
    pub const CHART_INDUCTION: Self = Self {
        fibers: Some(crate::slot_orbits::FiberRead::Chart),
        ..Self::DENSE_INDUCTION
    };
    pub const TYPED_LOOKUP: Self = Self {
        path_passes: Some(5),
        ..Self::CHART_INDUCTION
    };

    /// Change only the retrieved value. All arms use the original parameter layout.
    pub fn induction_controls(self) -> Vec<(&'static str, Self)> {
        use crate::slot_orbits::FiberRead;
        let mut variants = Vec::new();
        if self.path_passes == Some(5) {
            for (name, passes) in [
                ("one_local_read", 1),
                ("two_local_reads", 2),
                ("three_local_reads", 3),
                ("ten_local_reads", 10),
                ("without_local_reads", 0),
            ] {
                variants.push((
                    name,
                    Self {
                        path_passes: Some(passes),
                        ..self
                    },
                ));
            }
        }
        if self.fibers == Some(FiberRead::Chart) {
            for (name, fibers) in [
                ("without_chart", None),
                ("partner_values_only", Some(FiberRead::PartnerValues)),
                ("with_partner_values", Some(FiberRead::PartnerThenChart)),
            ] {
                variants.push((name, Self { fibers, ..self }));
            }
        }
        if self.dense_induction {
            variants.push((
                "without_dense_induction",
                Self {
                    dense_induction: false,
                    ..self
                },
            ));
        }
        variants
    }

    const fn id(self, with_phase: bool) -> &'static str {
        use crate::slot_orbits::RoleEvidence;
        if self.pairs.is_some() {
            return match (self, with_phase) {
                (Self::CHART_INDUCTION, true) => "phasor-triple-chart-induction-32-v1",
                (Self::TYPED_LOOKUP, true) => "phasor-triple-typed-lookup-32-v1",
                (Self::TYPED_LOOKUP, false) => "phasor-triple-typed-lookup-trunk-v1",
                (Self::CHART_INDUCTION, false) => "phasor-triple-chart-induction-trunk-v1",
                (Self::DENSE_INDUCTION, true) => "phasor-triple-dense-induction-32-v1",
                (Self::DENSE_INDUCTION, false) => "phasor-triple-dense-induction-trunk-v1",
                (Self::PAIR_GRAPH, true) => "phasor-triple-pair-classes-current-32-v1",
                (Self::PAIR_GRAPH, false) => "phasor-triple-pair-classes-current-trunk-v1",
                (Self::CONTEXT_RESIDUAL, true) => "phasor-triple-context-residual-32-v1",
                (Self::CONTEXT_RESIDUAL, false) => "phasor-triple-context-residual-trunk-v1",
                (_, true) => "phasor-triple-pair-ablation-32",
                (_, false) => "phasor-triple-pair-ablation-trunk",
            };
        }
        let (trunk, phase) = match (self.gates, self.search) {
            (
                ReadGates::SharedCurrent,
                Some(PeerSearch::Iterated(Some(RoleEvidence::WITH_GRAPH))),
            ) => (
                "phasor-triple-graph-current-gate-trunk-v1",
                "phasor-triple-graph-current-gate-32-v1",
            ),
            (
                ReadGates::SharedPrevious,
                Some(PeerSearch::Iterated(Some(RoleEvidence::WITH_GRAPH))),
            ) => (
                "phasor-triple-graph-shared-gate-trunk-v1",
                "phasor-triple-graph-shared-gate-32-v1",
            ),
            (ReadGates::Unit, Some(PeerSearch::Iterated(Some(RoleEvidence::WITH_GRAPH)))) => (
                "phasor-triple-graph-unit-gates-trunk-v1",
                "phasor-triple-graph-unit-gates-32-v1",
            ),
            (ReadGates::WidthScaled, Some(PeerSearch::Aligned)) => (
                "phasor-triple-peer-trunk-v1",
                "phasor-triple-peer-phase-32-v1",
            ),
            (ReadGates::WidthScaled, Some(PeerSearch::Witnesses)) => (
                "phasor-triple-witness-trunk-v1",
                "phasor-triple-witness-phase-32-v1",
            ),
            (ReadGates::WidthScaled, Some(PeerSearch::Iterated(None))) => (
                "phasor-triple-iterated-witness-trunk-v1",
                "phasor-triple-iterated-witness-32-v1",
            ),
            (
                ReadGates::WidthScaled,
                Some(PeerSearch::Iterated(Some(RoleEvidence::WITH_GRAPH))),
            ) => (
                "phasor-triple-graph-relations-trunk-v1",
                "phasor-triple-graph-relations-32-v1",
            ),
            (
                ReadGates::WidthScaled,
                Some(PeerSearch::Iterated(Some(RoleEvidence::OBSERVED_ROLES))),
            ) => (
                "phasor-triple-role-overlap-trunk-v1",
                "phasor-triple-role-overlap-32-v1",
            ),
            _ => (
                "phasor-triple-peer-ablation-trunk",
                "phasor-triple-peer-ablation-phase",
            ),
        };
        if with_phase { phase } else { trunk }
    }
}

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
    TiedToeplitz,
    TwoFrame,
    TripleTransport,
    BipartiteTriple,
    LookupTripleTransport,
    TriplePhase,
    LookupTriplePhase,
    PeerTriple(PeerRead),
    PeerTriplePhase(PeerRead),
    FourierTransport,
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
            "tied-toeplitz" => Ok(Self::TiedToeplitz),
            "two-frame" => Ok(Self::TwoFrame),
            "triple-transport" => Ok(Self::TripleTransport),
            "triple-bipartite" => Ok(Self::BipartiteTriple),
            "triple-lookup-prototypes" => Ok(Self::LookupTripleTransport),
            "triple-phase-read" => Ok(Self::TriplePhase),
            "triple-lookup-phase" => Ok(Self::LookupTriplePhase),
            "triple-peer-phase" => Ok(Self::peer_phase(PeerSearch::Aligned)),
            "triple-witness-phase" => Ok(Self::peer_phase(PeerSearch::Witnesses)),
            "triple-iterated-witness" => Ok(Self::peer_phase(PeerSearch::Iterated(None))),
            "triple-unit-graph" => Ok(Self::PeerTriplePhase(PeerRead::UNIT_GRAPH)),
            "triple-shared-graph" => Ok(Self::PeerTriplePhase(PeerRead::SHARED_GRAPH)),
            "triple-current-graph" => Ok(Self::PeerTriplePhase(PeerRead::CURRENT_GRAPH)),
            "triple-pair-classes" => Ok(Self::PeerTriplePhase(PeerRead::PAIR_GRAPH)),
            "triple-dense-induction" => Ok(Self::PeerTriplePhase(PeerRead::DENSE_INDUCTION)),
            "triple-chart-induction" => Ok(Self::PeerTriplePhase(PeerRead::CHART_INDUCTION)),
            "triple-typed-lookup" => Ok(Self::PeerTriplePhase(PeerRead::TYPED_LOOKUP)),
            "triple-context-residual" => Ok(Self::PeerTriplePhase(PeerRead::CONTEXT_RESIDUAL)),
            "triple-graph-witness" => Ok(Self::peer_phase(PeerSearch::Iterated(Some(
                crate::slot_orbits::RoleEvidence::WITH_GRAPH,
            )))),
            "triple-role-witness" => Ok(Self::peer_phase(PeerSearch::Iterated(Some(
                crate::slot_orbits::RoleEvidence::OBSERVED_ROLES,
            )))),
            "fourier-transport" => Ok(Self::FourierTransport),
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
            Self::TiedToeplitz => "phasor-learned-tied-toeplitz-v1",
            Self::TwoFrame => "phasor-learned-two-sided-frame-v1",
            Self::TripleTransport => "phasor-learned-triple-transport-v1",
            Self::BipartiteTriple => "phasor-triple-bipartite-v1",
            Self::LookupTripleTransport => "phasor-triple-prototype-capacity-v1",
            Self::TriplePhase => "phasor-triple-phase-residual-32-v1",
            Self::LookupTriplePhase => "phasor-triple-lookup-phase-32-v1",
            Self::PeerTriple(read) => read.id(false),
            Self::PeerTriplePhase(read) => read.id(true),
            Self::FourierTransport => "phasor-learned-fourier-transport-v1",
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
        self.has_triple_phase()
            || matches!(
                self,
                Self::PrefixToeplitz
                    | Self::TiedToeplitz
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
        self.is_phase() || self.is_toeplitz() || matches!(self, Self::Householder)
    }

    pub const fn replaces_context(self) -> bool {
        self.is_toeplitz()
            || self.is_triple()
            || matches!(
                self,
                Self::ScalarPhase | Self::TwoFrame | Self::FourierTransport
            )
    }

    pub const fn transforms_context(self) -> bool {
        self.context_stages() > 0
            || self.is_triple()
            || matches!(self, Self::TwoFrame | Self::FourierTransport)
    }

    pub const fn is_toeplitz(self) -> bool {
        matches!(self, Self::PrefixToeplitz | Self::TiedToeplitz)
    }

    pub const fn is_triple(self) -> bool {
        matches!(
            self,
            Self::TripleTransport
                | Self::BipartiteTriple
                | Self::LookupTripleTransport
                | Self::PeerTriple(_)
        ) || self.has_triple_phase()
    }

    pub const fn separates_triple_roles(self) -> bool {
        matches!(self, Self::BipartiteTriple)
    }

    pub const fn has_triple_phase(self) -> bool {
        matches!(
            self,
            Self::TriplePhase | Self::LookupTriplePhase | Self::PeerTriplePhase(_)
        )
    }

    pub const fn uses_lookup_capacity(self) -> bool {
        matches!(self, Self::LookupTripleTransport | Self::LookupTriplePhase)
            || self.peer_read().is_some()
    }

    pub const fn peer_phase(search: PeerSearch) -> Self {
        Self::PeerTriplePhase(PeerRead::new(search))
    }

    pub const fn peer_read(self) -> Option<PeerRead> {
        match self {
            Self::PeerTriple(read) | Self::PeerTriplePhase(read) => Some(read),
            _ => None,
        }
    }

    pub const fn read_gates(self) -> ReadGates {
        match self.peer_read() {
            Some(read) => read.gates,
            None => ReadGates::WidthScaled,
        }
    }

    pub const fn unit_read_gates(self) -> bool {
        !matches!(self.read_gates(), ReadGates::WidthScaled)
    }

    pub fn peer_search(self) -> Option<PeerSearch> {
        self.peer_read().and_then(|read| read.search)
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
            Self::TwoFrame => (2 * cfg.width + 8) * cfg.width,
            Self::TripleTransport
            | Self::BipartiteTriple
            | Self::LookupTripleTransport
            | Self::TriplePhase
            | Self::PeerTriple(_)
            | Self::PeerTriplePhase(_)
            | Self::LookupTriplePhase => crate::slot_memory::parameter_count(
                self,
                cfg.width,
                crate::slot_memory::capacity(self, cfg.width, cfg.vocab),
            ),
            Self::FourierTransport => crate::fourier::parameters(cfg.width),
            Self::PrefixToeplitz | Self::TiedToeplitz => {
                crate::toeplitz::parameter_count(cfg.width, self == Self::TiedToeplitz)
            }
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
            Self::TripleTransport
            | Self::BipartiteTriple
            | Self::LookupTripleTransport
            | Self::TriplePhase
            | Self::PeerTriple(_)
            | Self::PeerTriplePhase(_)
            | Self::LookupTriplePhase => {
                crate::slot_memory::initialize(cfg, seed, parameters);
            }
            Self::FourierTransport => crate::fourier::initialize(cfg, seed, parameters),
            Self::TwoFrame => {
                let d = cfg.width;
                let first = crate::architecture::memory_start(cfg);
                let bias = first + 3 * d;
                let matrix = bias + d * d;
                let freq = matrix + d * d;
                let mut rng = Random::new(seed.wrapping_add(30_000));
                let scale = (d as f64).sqrt().recip();
                for (range, sd) in [
                    (first..first + 2 * d, scale),
                    (bias..matrix, 0.25 * scale),
                    (matrix..freq, scale),
                    (freq..freq + d, (d as f64).sqrt().sqrt()),
                ] {
                    for w in &mut parameters[range] {
                        *w = rng.gauss(0., sd) as f32;
                    }
                }
                parameters[first + 2 * d..bias].fill(0.);
            }
            Self::PrefixToeplitz | Self::TiedToeplitz => {
                let d = cfg.width;
                let mut offset = crate::architecture::memory_start(cfg);
                let scale = (d as f64).sqrt().recip();
                let mut rng = Random::new(seed.wrapping_add(30_000));
                for (i, (count, stddev)) in [
                    (2 * d * d, scale),
                    (2 * d * d, scale / 2f64.sqrt()),
                    ((2 * d - 1) * d, 1. / d as f64),
                    (2 * d * d, scale),
                    (2 * d * d, scale),
                ]
                .into_iter()
                .enumerate()
                {
                    if self == Self::TiedToeplitz && i == 1 {
                        // Preserve the untied model's random stream for all shared controls.
                        for _ in 0..count {
                            rng.gauss(0., stddev);
                        }
                        continue;
                    }
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
