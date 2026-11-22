// Included as a child of the pinned engine to use its affine and tied-readout adjoints.
use super::*;
use crate::{
    head::{Head, MEMORY_WIDTH as M},
    affine_phase, bilinear_phase, phase_prefix, recurrent, reflection_phase, transition,
};

enum Memory {
    Phase(phase_prefix::kernels::Kernels),
    AffinePhase(affine_phase::kernels::Kernels),
    BilinearPhase(bilinear_phase::kernels::Kernels),
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

impl HiddenLayer {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        let cfg = &e.config;
        let rows = cfg.max_batch * cfg.max_length;
        let memory = match cfg.head {
            Head::PhasePrefix | Head::InputPhase | Head::SingleReflection => Memory::Phase(phase_prefix::kernels::Kernels::load(&e.cuda)?),
            Head::AffinePhase => Memory::AffinePhase(affine_phase::kernels::Kernels::load(&e.cuda)?),
            Head::BilinearPhase => Memory::BilinearPhase(bilinear_phase::kernels::Kernels::load(&e.cuda)?),
            Head::ReflectionPhase | Head::AssociativeReflection => Memory::Reflection {
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
        let input_weights = crate::architecture::memory_start(cfg);
        let (projected_channels, read_channels) = if cfg.head.is_phase() {
            (cfg.head.phase_inputs(), 2 * M)
        } else {
            (3 * M, M)
        };
        Ok(Self {
            memory,
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
            match &self.memory {
                Memory::Reflection { kernels, angles, gate_offset, .. } => kernels.forward(
                    Grid::new((rows / length) as u32, M as u32),
                    self.read.ptr(), self.projected.ptr(), angles.ptr(),
                    e.parameters.at(*gate_offset).ok_or("reflection gates outside parameters")?,
                    length as u32,
                )?,
                Memory::BilinearPhase(kernels) => kernels.forward(
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
        e.linear(
            self.merged.ptr(),
            self.read.ptr(),
            self.output_weights,
            rows,
            d,
            self.read_channels,
        )?;
        unsafe {
            e.ops.add(
                grid(rows * d),
                self.merged.ptr(),
                self.merged.ptr(),
                e.last.ptr(),
                (rows * d) as u32,
            )?;
        }
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
        e.linear_back(
            self.dread.ptr(),
            e.dlast.ptr(),
            self.read.ptr(),
            self.output_weights,
            rows,
            d,
            self.read_channels,
        )?;
        unsafe {
            match &self.memory {
                Memory::Reflection { kernels, angles, gate_partials, gate_offset } => {
                    let batch = (rows / length) as u32;
                    kernels.backward(
                        Grid::new(batch,M as u32), self.dprojected.ptr(), gate_partials.ptr(),
                        self.projected.ptr(), angles.ptr(), self.dread.ptr(),
                        e.parameters.at(*gate_offset).ok_or("reflection gates outside parameters")?, length as u32,
                    )?;
                    kernels.gate_sum(Grid::new(1,M as u32),
                        e.gradients.at(*gate_offset).ok_or("reflection gates outside gradients")?,
                        gate_partials.ptr(), batch)?;
                }
                Memory::BilinearPhase(kernels) => kernels.backward(
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
