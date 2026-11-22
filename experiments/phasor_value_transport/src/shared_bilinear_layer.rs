//! A shared recurrent input state observed through a learned frequency matrix.
use super::*;
use crate::{
    bilinear_phase,
    head::{MEMORY_WIDTH as M, SHARED_LATENTS},
    phase_prefix,
};

pub(super) struct SharedBilinearPhase {
    scan: bilinear_phase::kernels::Kernels,
    observations: phase_prefix::kernels::Kernels,
    latent: F,
    latent_grad: F,
    angles: F,
    angle_grad: F,
    frequencies: usize,
}

impl SharedBilinearPhase {
    pub(super) fn new(e: &Engine, rows: usize) -> Result<Self> {
        let cfg = &e.config;
        let r = SHARED_LATENTS;
        Ok(Self {
            scan: bilinear_phase::kernels::Kernels::load(&e.cuda)?,
            observations: phase_prefix::kernels::Kernels::load(&e.cuda)?,
            latent: e.cuda.alloc(rows * r)?,
            latent_grad: e.cuda.alloc(rows * r)?,
            angles: e.cuda.alloc(rows * M)?,
            angle_grad: e.cuda.alloc(rows * M)?,
            frequencies: crate::architecture::memory_start(cfg)
                + (cfg.head.phase_inputs() + 2 * M) * cfg.width,
        })
    }

    pub(super) fn forward(
        &self,
        e: &Engine,
        read: P,
        projected: P,
        rows: usize,
        length: usize,
    ) -> Result<()> {
        let r = SHARED_LATENTS;
        unsafe {
            self.scan.forward::<true>(
                Grid::new((rows / length) as u32, r as u32),
                self.latent.ptr(),
                projected,
                length as u32,
            )?;
        }
        e.linear(
            self.angles.ptr(),
            self.latent.ptr(),
            self.frequencies,
            rows,
            M,
            r,
        )?;
        // A length-one scan observes each mixed angle independently at its own row.
        unsafe {
            self.observations.forward::<false>(
                Grid::new(rows as u32, M as u32),
                read,
                self.angles.ptr(),
                1,
            )?;
        }
        Ok(())
    }

    pub(super) fn backward(
        &self,
        e: &Engine,
        gradient: P,
        projected: P,
        dread: P,
        rows: usize,
        length: usize,
    ) -> Result<()> {
        let r = SHARED_LATENTS;
        unsafe {
            self.observations.backward::<false>(
                Grid::new(rows as u32, M as u32),
                self.angle_grad.ptr(),
                self.angles.ptr(),
                dread,
                1,
            )?;
        }
        e.linear_back(
            self.latent_grad.ptr(),
            self.angle_grad.ptr(),
            self.latent.ptr(),
            self.frequencies,
            rows,
            M,
            r,
        )?;
        unsafe {
            self.scan.backward::<true>(
                Grid::new((rows / length) as u32, r as u32),
                gradient,
                projected,
                self.latent_grad.ptr(),
                length as u32,
            )?;
        }
        Ok(())
    }
}
