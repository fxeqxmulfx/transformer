//! Associative residual on the mixed reflection backbone.
use super::*;
use crate::associative::{KEYS, PROJECTED, VALUES, kernels::Kernels};
use crate::head::Head;

pub(super) struct AssociativeLayer {
    kernels: Kernels,
    projected: F,
    inverse: F,
    states: F,
    read: F,
    merged: F,
    read_grad: F,
    projected_grad: F,
    input_grad: F,
    input_weights: usize,
    output_weights: usize,
}

impl AssociativeLayer {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        let cfg = &e.config;
        let rows = cfg.max_batch * cfg.max_length;
        let input_weights = crate::architecture::start(cfg) + Head::ReflectionPhase.parameters(cfg);
        Ok(Self {
            kernels: Kernels::load(&e.cuda)?,
            projected: e.cuda.alloc(rows * PROJECTED)?,
            inverse: e.cuda.alloc(rows * 2)?,
            states: e.cuda.alloc(rows * VALUES * KEYS)?,
            read: e.cuda.alloc(rows * VALUES)?,
            merged: e.cuda.alloc(rows * cfg.width)?,
            read_grad: e.cuda.alloc(rows * VALUES)?,
            projected_grad: e.cuda.alloc(rows * PROJECTED)?,
            input_grad: e.cuda.alloc(rows * cfg.width)?,
            input_weights,
            output_weights: input_weights + PROJECTED * cfg.width,
        })
    }

    pub(super) fn features(&self) -> &F {
        &self.merged
    }

    pub(super) fn forward(&self, e: &Engine, mixed: P, rows: usize, length: usize) -> Result<()> {
        let d = e.config.width;
        e.linear(
            self.projected.ptr(),
            e.embed_norm.ptr(),
            self.input_weights,
            rows,
            PROJECTED,
            d,
        )?;
        // Each write carries state within exactly one validated sequence.
        unsafe {
            self.kernels.bound::<false>(
                Grid::new(rows as u32, 32),
                self.projected.ptr(),
                self.projected_grad.ptr(),
                self.inverse.ptr(),
            )?;
            self.kernels.forward(
                Grid::new((rows / length) as u32, 32),
                self.read.ptr(),
                self.states.ptr(),
                self.projected.ptr(),
                length as u32,
            )?;
        }
        e.linear(
            self.merged.ptr(),
            self.read.ptr(),
            self.output_weights,
            rows,
            d,
            VALUES,
        )?;
        unsafe {
            e.ops.add(
                grid(rows * d),
                self.merged.ptr(),
                self.merged.ptr(),
                mixed,
                (rows * d) as u32,
            )?;
        }
        Ok(())
    }

    pub(super) fn backward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        let d = e.config.width;
        e.linear_back(
            self.read_grad.ptr(),
            e.dlast.ptr(),
            self.read.ptr(),
            self.output_weights,
            rows,
            d,
            VALUES,
        )?;
        unsafe {
            self.kernels.backward(
                Grid::new((rows / length) as u32, 32),
                self.projected_grad.ptr(),
                self.read_grad.ptr(),
                self.states.ptr(),
                self.projected.ptr(),
                length as u32,
            )?;
            self.kernels.bound::<true>(
                Grid::new(rows as u32, 32),
                self.projected.ptr(),
                self.projected_grad.ptr(),
                self.inverse.ptr(),
            )?;
        }
        e.linear_back(
            self.input_grad.ptr(),
            self.projected_grad.ptr(),
            e.embed_norm.ptr(),
            self.input_weights,
            rows,
            PROJECTED,
            d,
        )
    }

    pub(super) fn add_input_gradient(&self, e: &Engine, target: P, rows: usize) -> Result<()> {
        let size = rows * e.config.width;
        unsafe {
            e.ops.add(
                grid(size),
                target,
                target,
                self.input_grad.ptr(),
                size as u32,
            )?;
        }
        Ok(())
    }
}
