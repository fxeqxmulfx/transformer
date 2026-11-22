//! Included in the pinned engine. Both inputs are dense feature sequences.
use super::*;
use crate::context_rotation::kernels::Kernels;

pub(super) struct ContextRotation {
    kernels: Kernels,
    angles: F,
    angle_gradient: F,
    rotated: F,
    input_gradient: F,
    projection: usize,
    shape: std::cell::Cell<Option<(usize, usize)>>,
}

impl ContextRotation {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        let cfg = &e.config;
        let size = cfg.max_batch * cfg.max_length * cfg.width;
        Ok(Self {
            kernels: Kernels::load(&e.cuda)?,
            angles: e.cuda.alloc(size / 2)?,
            angle_gradient: e.cuda.alloc(size / 2)?,
            rotated: e.cuda.alloc(size)?,
            input_gradient: e.cuda.alloc(size)?,
            projection: crate::architecture::memory_start(cfg),
            shape: std::cell::Cell::new(None),
        })
    }

    pub(super) fn features(&self) -> &F { &self.rotated }

    pub(super) fn forward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        self.shape.set(None);
        let width = e.config.width;
        let pairs = width / 2;
        e.linear(self.angles.ptr(), e.embed_norm.ptr(), self.projection, rows, pairs, width)?;
        // The engine has validated the even model width, batch and sequence length.
        unsafe {
            self.kernels.forward(Grid::new((rows / length) as u32, pairs as u32),
                self.rotated.ptr(), self.angles.ptr(), e.last.ptr(), width as u32, length as u32)?;
        }
        self.shape.set(Some((rows, length)));
        Ok(())
    }

    pub(super) fn backward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        if self.shape.get() != Some((rows, length)) {
            return Err("contextual rotation backward requires a matching forward".into());
        }
        let width = e.config.width;
        let pairs = width / 2;
        unsafe {
            self.kernels.backward(Grid::new((rows / length) as u32, pairs as u32),
                e.dlast.ptr(), self.angle_gradient.ptr(), self.rotated.ptr(), self.angles.ptr(),
                width as u32, length as u32)?;
        }
        e.linear_back(self.input_gradient.ptr(), self.angle_gradient.ptr(), e.embed_norm.ptr(),
            self.projection, rows, pairs, width)
    }

    pub(super) fn add_input_gradient(&self, e: &Engine, target: P, rows: usize) -> Result<()> {
        let n = rows * e.config.width;
        unsafe {
            e.ops.add(grid(n), target, target, self.input_gradient.ptr(), n as u32)?;
        }
        Ok(())
    }
}
