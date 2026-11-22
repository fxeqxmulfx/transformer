// Included as a child of the pinned engine to use its affine and tied-readout adjoints.
use super::*;
use crate::{
    head::{Head, MEMORY_WIDTH as M},
    recurrent, transition,
};

enum Memory {
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
    projected: F,
    dprojected: F,
    read: F,
    dread: F,
    merged: F,
    dhidden: F,
    input_weights: usize,
    output_weights: usize,
    forward_shape: std::cell::Cell<Option<(usize, usize)>>,
}

impl HiddenLayer {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        let cfg = &e.config;
        let rows = cfg.max_batch * cfg.max_length;
        let memory = match cfg.head {
            Head::Transition => Memory::Transitions {
                kernels: transition::kernels::Kernels::load(&e.cuda)?,
                probabilities: e.cuda.alloc(rows * cfg.max_length)?,
                scores: e.cuda.alloc(rows * cfg.max_length)?,
            },
            Head::Recurrent => Memory::Recurrent {
                kernels: recurrent::kernels::Kernels::load(&e.cuda)?,
                states: e.cuda.alloc(rows * M)?,
            },
            _ => return Err("expected a learned hidden memory".into()),
        };
        let input_weights = crate::architecture::start(cfg);
        Ok(Self {
            memory,
            projected: e.cuda.alloc(rows * 3 * M)?,
            dprojected: e.cuda.alloc(rows * 3 * M)?,
            read: e.cuda.alloc(rows * M)?,
            dread: e.cuda.alloc(rows * M)?,
            merged: e.cuda.alloc(rows * cfg.width)?,
            dhidden: e.cuda.alloc(rows * cfg.width)?,
            input_weights,
            output_weights: input_weights + 3 * M * cfg.width,
            forward_shape: std::cell::Cell::new(None),
        })
    }

    // The only input activations are the backbone's final RMS-normalized stream means.
    pub(super) fn features(&self) -> &F {
        &self.merged
    }

    pub(super) fn forward_features(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        self.forward_shape.set(None);
        let d = e.config.width;
        e.linear(
            self.projected.ptr(),
            e.last.ptr(),
            self.input_weights,
            rows,
            3 * M,
            d,
        )?;
        // Each launch owns its sequence slice, with length and rows checked by Engine::forward.
        unsafe {
            match &self.memory {
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
                Memory::Recurrent { kernels, states } => kernels.forward(
                    Grid::new((rows / length) as u32, 32),
                    self.read.ptr(),
                    states.ptr(),
                    self.projected.ptr(),
                    length as u32,
                )?,
            }
        }
        e.linear(
            self.merged.ptr(),
            self.read.ptr(),
            self.output_weights,
            rows,
            d,
            M,
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
        self.forward_shape.set(Some((rows, length)));
        Ok(())
    }

    pub(super) fn backward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        if self.forward_shape.get() != Some((rows, length)) {
            return Err("hidden memory backward needs a forward pass with the same shape".into());
        }
        let d = e.config.width;
        // This writes the tied output embedding gradient and the direct residual path dh.
        e.linear_back(
            e.dlast.ptr(),
            e.dlogits.ptr(),
            self.merged.ptr(),
            0,
            rows,
            e.config.vocab,
            d,
        )?;
        e.linear_back(
            self.dread.ptr(),
            e.dlast.ptr(),
            self.read.ptr(),
            self.output_weights,
            rows,
            d,
            M,
        )?;
        unsafe {
            match &self.memory {
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
                Memory::Recurrent { kernels, states } => kernels.backward(
                    Grid::new((rows / length) as u32, 32),
                    self.dprojected.ptr(),
                    self.projected.ptr(),
                    states.ptr(),
                    self.dread.ptr(),
                    length as u32,
                )?,
            }
        }
        e.linear_back(
            self.dhidden.ptr(),
            self.dprojected.ptr(),
            e.last.ptr(),
            self.input_weights,
            rows,
            3 * M,
            d,
        )?;
        unsafe {
            e.ops.add(
                grid(rows * d),
                e.dlast.ptr(),
                e.dlast.ptr(),
                self.dhidden.ptr(),
                (rows * d) as u32,
            )?;
        }
        Ok(())
    }
}
