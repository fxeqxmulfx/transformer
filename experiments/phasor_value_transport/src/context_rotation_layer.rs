//! Included in the pinned engine. Every rotation stage reads only dense vectors.
use super::*;
use crate::context_rotation::kernels::Kernels;

struct Stage {
    angles: F,
    angle_gradient: F,
    rotated: F,
    input_gradient: F,
    projection: usize,
}

impl Stage {
    fn new(e: &Engine, projection: usize) -> Result<Self> {
        let cfg = &e.config;
        let size = cfg.max_batch * cfg.max_length * cfg.width;
        Ok(Self {
            angles: e.cuda.alloc(size / 2)?,
            angle_gradient: e.cuda.alloc(size / 2)?,
            rotated: e.cuda.alloc(size)?,
            input_gradient: e.cuda.alloc(size)?,
            projection,
        })
    }

    fn forward<const ADJACENT: bool>(
        &self,
        e: &Engine,
        kernels: &Kernels,
        context: P,
        rows: usize,
        length: usize,
    ) -> Result<()> {
        let width = e.config.width;
        let pairs = width / 2;
        e.linear(
            self.angles.ptr(),
            e.embed_norm.ptr(),
            self.projection,
            rows,
            pairs,
            width,
        )?;
        // Each sequence and physical coordinate pair has a single owner.
        unsafe {
            kernels.forward::<ADJACENT>(
                Grid::new((rows / length) as u32, pairs as u32),
                self.rotated.ptr(),
                self.angles.ptr(),
                context,
                width as u32,
                length as u32,
            )?;
        }
        Ok(())
    }

    fn backward<const ADJACENT: bool>(
        &self,
        e: &Engine,
        kernels: &Kernels,
        rows: usize,
        length: usize,
    ) -> Result<()> {
        let width = e.config.width;
        let pairs = width / 2;
        unsafe {
            kernels.backward::<ADJACENT>(
                Grid::new((rows / length) as u32, pairs as u32),
                e.dlast.ptr(),
                self.angle_gradient.ptr(),
                self.rotated.ptr(),
                self.angles.ptr(),
                width as u32,
                length as u32,
            )?;
        }
        e.linear_back(
            self.input_gradient.ptr(),
            self.angle_gradient.ptr(),
            e.embed_norm.ptr(),
            self.projection,
            rows,
            pairs,
            width,
        )
    }
}

pub(super) struct ContextRotation {
    kernels: Kernels,
    half_split: Stage,
    adjacent: Option<Stage>,
    shape: std::cell::Cell<Option<(usize, usize)>>,
}

impl ContextRotation {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        let cfg = &e.config;
        let start = crate::architecture::memory_start(cfg);
        Ok(Self {
            kernels: Kernels::load(&e.cuda)?,
            half_split: Stage::new(e, start)?,
            adjacent: (cfg.head.context_stages() == 2)
                .then(|| Stage::new(e, start + (cfg.width / 2) * cfg.width))
                .transpose()?,
            shape: std::cell::Cell::new(None),
        })
    }

    pub(super) fn features(&self) -> &F {
        self.adjacent
            .as_ref()
            .map_or(&self.half_split.rotated, |layer| &layer.rotated)
    }

    pub(super) fn forward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        self.shape.set(None);
        self.half_split
            .forward::<false>(e, &self.kernels, e.last.ptr(), rows, length)?;
        if let Some(layer) = &self.adjacent {
            layer.forward::<true>(
                e,
                &self.kernels,
                self.half_split.rotated.ptr(),
                rows,
                length,
            )?;
        }
        self.shape.set(Some((rows, length)));
        Ok(())
    }

    pub(super) fn backward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        if self.shape.get() != Some((rows, length)) {
            return Err("contextual rotation backward requires a matching forward".into());
        }
        // Each transpose replaces the hidden cotangent in place. Apply the
        // adjacent-pair transpose before computing the first angular gradients.
        if let Some(layer) = &self.adjacent {
            layer.backward::<true>(e, &self.kernels, rows, length)?;
        }
        self.half_split
            .backward::<false>(e, &self.kernels, rows, length)
    }

    pub(super) fn add_input_gradient(&self, e: &Engine, target: P, rows: usize) -> Result<()> {
        let n = rows * e.config.width;
        for stage in std::iter::once(&self.half_split).chain(&self.adjacent) {
            unsafe {
                e.ops.add(
                    grid(n),
                    target,
                    target,
                    stage.input_gradient.ptr(),
                    n as u32,
                )?;
            }
        }
        Ok(())
    }
}
