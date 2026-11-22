// Included as a child of the pinned engine to use its affine and tied-readout adjoints.
use super::*;
use super::shared_bilinear_layer::SharedBilinearPhase;
use crate::{
    head::{Head, MEMORY_WIDTH as M},
    affine_phase, bilinear_phase, phase_prefix, recurrent, reflection_phase, transition,
};

enum Memory {
    Phase(phase_prefix::kernels::Kernels),
    AffinePhase(affine_phase::kernels::Kernels),
    BilinearPhase(bilinear_phase::kernels::Kernels),
    SharedBilinear(SharedBilinearPhase),
    Toeplitz(super::toeplitz_layer::ToeplitzLayer),
    Householder {
        scan: crate::householder::kernels::Kernels,
        reduction: reflection_phase::kernels::Kernels,
        previous: F,
        partial_initial: F,
        initial_weights: usize,
    },
    Reflection {
        kernels: reflection_phase::kernels::Kernels,
        angles: F,
        gate_partials: F,
        gate_offset: usize,
    },
    Transitions {
        kernels: transition::kernels::Kernels,
        probabilities: F,
        scores: F,
    },
    Recurrent {
        kernels: recurrent::kernels::Kernels,
        states: F,
    },
}

pub(super) struct HiddenLayer {
    memory: Memory,
    tied: Option<super::tied_projection::TiedProjection>,
    periodic: Option<PeriodicActivation>,
    associative: Option<super::associative_layer::AssociativeLayer>,
    projected: F,
    dprojected: F,
    read: F,
    dread: F,
    merged: F,
    dhidden: F,
    input_weights: usize,
    output_weights: usize,
    projected_channels: usize,
    read_channels: usize,
    forward_shape: std::cell::Cell<Option<(usize, usize)>>,
}

struct PeriodicActivation {
    kernels: crate::periodic_activation::kernels::Kernels,
    raw: F,
    weights: usize,
}

impl HiddenLayer {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        let cfg = &e.config;
        let rows = cfg.max_batch * cfg.max_length;
        let memory = match cfg.head {
            Head::PhasePrefix | Head::InputPhase | Head::SingleReflection => Memory::Phase(phase_prefix::kernels::Kernels::load(&e.cuda)?),
            Head::AffinePhase => Memory::AffinePhase(affine_phase::kernels::Kernels::load(&e.cuda)?),
            Head::BilinearPhase => Memory::BilinearPhase(bilinear_phase::kernels::Kernels::load(&e.cuda)?),
            Head::SharedBilinear | Head::ScalarPhase => Memory::SharedBilinear(SharedBilinearPhase::new(e, rows)?),
            Head::PrefixToeplitz | Head::TiedToeplitz => Memory::Toeplitz(super::toeplitz_layer::ToeplitzLayer::new(e, rows)?),
            Head::Householder => Memory::Householder {
                scan: crate::householder::kernels::Kernels::load(&e.cuda)?,
                reduction: reflection_phase::kernels::Kernels::load(&e.cuda)?,
                previous: e.cuda.alloc(rows * crate::householder::FEATURES)?,
                partial_initial: e.cuda.alloc(cfg.max_batch * crate::householder::FEATURES)?,
                initial_weights: crate::architecture::start(cfg) + cfg.head.parameters(cfg) - crate::householder::FEATURES,
            },
            Head::ReflectionPhase | Head::AssociativeReflection | Head::PeriodicReflection => Memory::Reflection {
                kernels: reflection_phase::kernels::Kernels::load(&e.cuda)?,
                angles: e.cuda.alloc(rows * M)?,
                gate_partials: e.cuda.alloc(cfg.max_batch * M)?,
                gate_offset: crate::architecture::start(cfg) + Head::ReflectionPhase.parameters(cfg) - M,
            },
            Head::Transition => Memory::Transitions {
                kernels: transition::kernels::Kernels::load(&e.cuda)?,
                probabilities: e.cuda.alloc(rows * cfg.max_length)?,
                scores: e.cuda.alloc(rows * cfg.max_length)?,
            },
            Head::Recurrent | Head::Bounded => Memory::Recurrent {
                kernels: recurrent::kernels::Kernels::load(&e.cuda)?,
                states: e.cuda.alloc(rows * M)?,
            },
            _ => return Err("expected a learned hidden memory".into()),
        };
        let periodic = if cfg.head == Head::PeriodicReflection {
            Some(PeriodicActivation {
                kernels: crate::periodic_activation::kernels::Kernels::load(&e.cuda)?,
                raw: e.cuda.alloc(rows * M)?,
                weights: crate::architecture::start(cfg) + Head::ReflectionPhase.parameters(cfg),
            })
        } else { None };
        let input_weights = crate::architecture::memory_start(cfg);
        let (projected_channels, read_channels) = if cfg.head.is_phase() {
            (cfg.head.phase_inputs(), 2 * M)
        } else if cfg.head.is_toeplitz() {
            (2 * cfg.width, 2 * cfg.width)
        } else if cfg.head == Head::Householder {
            (crate::householder::INPUTS, crate::householder::FEATURES)
        } else {
            (3 * M, M)
        };
        Ok(Self {
            memory,
            tied: (cfg.head == Head::TiedToeplitz).then(|| super::tied_projection::TiedProjection::new(e, input_weights, 2*cfg.width)).transpose()?,
            periodic,
            associative: (cfg.head == Head::AssociativeReflection)
                .then(|| super::associative_layer::AssociativeLayer::new(e)).transpose()?,
            projected: e.cuda.alloc(rows * projected_channels)?,
            dprojected: e.cuda.alloc(rows * projected_channels)?,
            read: e.cuda.alloc(rows * read_channels)?,
            dread: e.cuda.alloc(rows * read_channels)?,
            merged: e.cuda.alloc(rows * cfg.width)?,
            dhidden: e.cuda.alloc(rows * cfg.width)?,
            input_weights,
            output_weights: input_weights + projected_channels * cfg.width,
            projected_channels,
            read_channels,
            forward_shape: std::cell::Cell::new(None),
        })
    }

    fn projection_input(e: &Engine) -> P {
        if e.config.head.projects_embeddings() {
            e.embed_norm.ptr()
        } else {
            e.last.ptr()
        }
    }

    pub(super) fn features(&self) -> &F {
        self.associative.as_ref().map_or(&self.merged, |a| a.features())
    }

    pub(super) fn forward_features(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        self.forward_shape.set(None);
        let d = e.config.width;
        e.linear(
            self.projected.ptr(),
            Self::projection_input(e),
            self.input_weights,
            rows,
            self.projected_channels,
            d,
        )?;
        // Each launch owns its sequence slice, with length and rows checked by Engine::forward.
        unsafe {
            if let Some(a) = &self.periodic {
                a.kernels.forward(Grid::new(rows as u32,M as u32), self.projected.ptr(),
                    a.raw.ptr(), e.parameters.at(a.weights).ok_or("missing periodic weights")?)?;
            }
            match &self.memory {
                Memory::Householder { scan, previous, initial_weights, .. } => scan.forward(
                    Grid::new((rows / length) as u32, crate::householder::FEATURES as u32),
                    self.read.ptr(), previous.ptr(), self.projected.ptr(),
                    e.parameters.at(*initial_weights).ok_or("missing initial reflection states")?, length as u32)?,
                Memory::Reflection { kernels, angles, gate_offset, .. } => kernels.forward(
                    Grid::new((rows / length) as u32, M as u32),
                    self.read.ptr(), self.projected.ptr(), angles.ptr(),
                    e.parameters.at(*gate_offset).ok_or("reflection gates outside parameters")?,
                    length as u32,
                )?,
                Memory::SharedBilinear(memory) => memory.forward(e, self.read.ptr(),
                    self.projected.ptr(), rows, length)?,
                Memory::Toeplitz(memory) => memory.forward(e, self.read.ptr(), self.projected.ptr(), rows, length)?,
                Memory::BilinearPhase(kernels) => kernels.forward::<false>(
                    Grid::new((rows / length) as u32, M as u32),
                    self.read.ptr(), self.projected.ptr(), length as u32,
                )?,
                Memory::AffinePhase(kernels) => kernels.forward(
                    Grid::new((rows / length) as u32, M as u32),
                    self.read.ptr(),
                    self.projected.ptr(),
                    length as u32,
                )?,
                Memory::Phase(kernels) => {
                    let forward = if e.config.head == Head::SingleReflection {
                        phase_prefix::kernels::Kernels::forward::<true>
                    } else {
                        phase_prefix::kernels::Kernels::forward::<false>
                    };
                    forward(kernels, Grid::new((rows / length) as u32, M as u32),
                        self.read.ptr(), self.projected.ptr(), length as u32)?;
                }
                Memory::Transitions {
                    kernels,
                    probabilities,
                    scores,
                } => kernels.forward(
                    Grid::new(rows as u32, 32),
                    self.read.ptr(),
                    probabilities.ptr(),
                    scores.ptr(),
                    self.projected.ptr(),
                    length as u32,
                )?,
                Memory::Recurrent { kernels, states } => {
                    let forward = if e.config.head == Head::Bounded {
                        recurrent::kernels::Kernels::forward::<true>
                    } else {
                        recurrent::kernels::Kernels::forward::<false>
                    };
                    forward(
                        kernels,
                        Grid::new((rows / length) as u32, 32),
                        self.read.ptr(),
                        states.ptr(),
                        self.projected.ptr(),
                        length as u32,
                    )?;
                }
            }
        }
        if let Some(tied) = &self.tied {
            tied.forward(e, self.merged.ptr(), self.read.ptr(), rows)?;
        } else { e.linear(
            self.merged.ptr(),
            self.read.ptr(),
            self.output_weights,
            rows,
            d,
            self.read_channels,
        )?; }
        if !e.config.head.replaces_context() { unsafe {
            e.ops.add(
                grid(rows * d),
                self.merged.ptr(),
                self.merged.ptr(),
                e.last.ptr(),
                (rows * d) as u32,
            )?;
        } }
        if let Some(a) = &self.associative {
            a.forward(e, self.merged.ptr(), rows, length)?;
        }
        self.forward_shape.set(Some((rows, length)));
        Ok(())
    }

    pub(super) fn backward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        if self.forward_shape.get() != Some((rows, length)) {
            return Err("hidden memory backward needs a forward pass with the same shape".into());
        }
        let d = e.config.width;
        if let Some(a) = &self.associative {
            a.backward(e, rows, length)?;
        }
        if let Some(tied) = &self.tied {
            tied.backward(e, self.dread.ptr(), self.read.ptr(), rows)?;
        } else { e.linear_back(
            self.dread.ptr(),
            e.dlast.ptr(),
            self.read.ptr(),
            self.output_weights,
            rows,
            d,
            self.read_channels,
        )?; }
        unsafe {
            match &self.memory {
                Memory::Householder { scan, reduction, previous, partial_initial, initial_weights } => {
                    let batch = (rows / length) as u32;
                    let features = crate::householder::FEATURES as u32;
                    scan.backward(Grid::new(batch, features), self.dprojected.ptr(), partial_initial.ptr(),
                        previous.ptr(), self.projected.ptr(), self.dread.ptr(), length as u32)?;
                    reduction.sum_partials::<true>(Grid::new(1,features),
                        e.gradients.at(*initial_weights).ok_or("missing initial reflection gradients")?,
                        partial_initial.ptr(), batch)?;
                }
                Memory::Reflection { kernels, angles, gate_partials, gate_offset } => {
                    let batch = (rows / length) as u32;
                    kernels.backward(
                        Grid::new(batch,M as u32), self.dprojected.ptr(), gate_partials.ptr(),
                        self.projected.ptr(), angles.ptr(), self.dread.ptr(),
                        e.parameters.at(*gate_offset).ok_or("reflection gates outside parameters")?, length as u32,
                    )?;
                    kernels.sum_partials::<false>(Grid::new(1,M as u32),
                        e.gradients.at(*gate_offset).ok_or("reflection gates outside gradients")?,
                        gate_partials.ptr(), batch)?;
                }
                Memory::SharedBilinear(memory) => memory.backward(e, self.dprojected.ptr(),
                    self.projected.ptr(), self.dread.ptr(), rows, length)?,
                Memory::Toeplitz(memory) => memory.backward(e, self.dprojected.ptr(), self.projected.ptr(), self.dread.ptr(), rows, length)?,
                Memory::BilinearPhase(kernels) => kernels.backward::<false>(
                    Grid::new((rows / length) as u32, M as u32),
                    self.dprojected.ptr(), self.projected.ptr(), self.dread.ptr(), length as u32,
                )?,
                Memory::AffinePhase(kernels) => kernels.backward(
                    Grid::new((rows / length) as u32, M as u32),
                    self.dprojected.ptr(),
                    self.projected.ptr(),
                    self.read.ptr(),
                    self.dread.ptr(),
                    length as u32,
                )?,
                Memory::Phase(kernels) => {
                    let backward = if e.config.head == Head::SingleReflection {
                        phase_prefix::kernels::Kernels::backward::<true>
                    } else {
                        phase_prefix::kernels::Kernels::backward::<false>
                    };
                    backward(kernels, Grid::new((rows / length) as u32, M as u32),
                        self.dprojected.ptr(), self.projected.ptr(), self.dread.ptr(), length as u32)?;
                }
                Memory::Transitions {
                    kernels,
                    probabilities,
                    scores,
                } => {
                    let launch = Grid::new(rows as u32, 32);
                    kernels.queries_backward(
                        launch,
                        self.dprojected.ptr(),
                        scores.ptr(),
                        self.projected.ptr(),
                        probabilities.ptr(),
                        self.read.ptr(),
                        self.dread.ptr(),
                        length as u32,
                    )?;
                    kernels.sources_backward(
                        launch,
                        self.dprojected.ptr(),
                        self.projected.ptr(),
                        probabilities.ptr(),
                        scores.ptr(),
                        self.dread.ptr(),
                        length as u32,
                    )?;
                }
                Memory::Recurrent { kernels, states } => {
                    let backward = if e.config.head == Head::Bounded {
                        recurrent::kernels::Kernels::backward::<true>
                    } else {
                        recurrent::kernels::Kernels::backward::<false>
                    };
                    backward(
                        kernels,
                        Grid::new((rows / length) as u32, 32),
                        self.dprojected.ptr(),
                        self.projected.ptr(),
                        states.ptr(),
                        self.dread.ptr(),
                        length as u32,
                    )?;
                }
            }
        }
        if let Some(a) = &self.periodic {
            unsafe {
                a.kernels.backward(Grid::new(M as u32,32), self.dprojected.ptr(),
                    e.gradients.at(a.weights).ok_or("missing periodic gradient")?,
                    a.raw.ptr(), e.parameters.at(a.weights).ok_or("missing periodic weights")?, rows as u32)?;
            }
        }
        e.linear_back(
            self.dhidden.ptr(),
            self.dprojected.ptr(),
            Self::projection_input(e),
            self.input_weights,
            rows,
            self.projected_channels,
            d,
        )?;
        if !e.config.head.projects_embeddings() {
            self.add_projection_gradient(e, e.dlast.ptr(), rows)?;
        }
        if let Some(tied) = &self.tied { tied.accumulate(e)?; }
        Ok(())
    }

    // Add the VJP at the same activation that supplied the forward projection.
    pub(super) fn add_projection_gradient(&self, e: &Engine, target: P, rows: usize) -> Result<()> {
        let d = e.config.width;
        if let Some(a) = &self.associative {
            a.add_input_gradient(e, target, rows)?;
        }
        unsafe {
            e.ops.add(
                grid(rows * d),
                target,
                target,
                self.dhidden.ptr(),
                (rows * d) as u32,
            )?;
        }
        Ok(())
    }
}
