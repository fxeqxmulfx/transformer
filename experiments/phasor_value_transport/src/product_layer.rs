// Included inside the pinned engine. This layer accepts normalized input
// embeddings and precedes every attention and feed-forward layer.
use super::*;
use crate::{causal_product::kernels::Kernels, head::MEMORY_WIDTH as M};

pub(super) struct ProductLayer {
    kernels: Kernels,
    projected: F,
    dprojected: F,
    product: F,
    dproduct: F,
    features: F,
    dnormalized: F,
    input_weights: usize,
    output_weights: usize,
}

impl ProductLayer {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        let rows = e.config.max_batch * e.config.max_length;
        let width = e.config.width;
        let input_weights = crate::architecture::start(&e.config);
        Ok(Self {
            kernels: Kernels::load(&e.cuda)?,
            projected: e.cuda.alloc(rows * 3 * M)?,
            dprojected: e.cuda.alloc(rows * 3 * M)?,
            product: e.cuda.alloc(rows * M)?,
            dproduct: e.cuda.alloc(rows * M)?,
            features: e.cuda.alloc(rows * width)?,
            dnormalized: e.cuda.alloc(rows * width)?,
            input_weights,
            output_weights: input_weights + 3 * M * width,
        })
    }

    pub(super) fn forward(&self, e: &Engine, rows: usize, length: usize) -> Result<P> {
        let width = e.config.width;
        e.linear(
            self.projected.ptr(),
            e.embed_norm.ptr(),
            self.input_weights,
            rows,
            3 * M,
            width,
        )?;
        // Each block owns one complete row with one lane per product coordinate.
        unsafe {
            self.kernels.forward(
                Grid::new(rows as u32, M as u32),
                self.product.ptr(),
                self.projected.ptr(),
                length as u32,
            )?;
        }
        e.linear(
            self.features.ptr(),
            self.product.ptr(),
            self.output_weights,
            rows,
            width,
            M,
        )?;
        unsafe {
            e.ops.add(
                grid(rows * width),
                self.features.ptr(),
                self.features.ptr(),
                e.embed_norm.ptr(),
                (rows * width) as u32,
            )?;
        }
        Ok(self.features.ptr())
    }

    pub(super) fn backward(&self, e: &Engine, rows: usize, length: usize) -> Result<P> {
        let width = e.config.width;
        e.linear_back(
            self.dproduct.ptr(),
            e.dx.ptr(),
            self.product.ptr(),
            self.output_weights,
            rows,
            width,
            M,
        )?;
        unsafe {
            self.kernels.backward(
                Grid::new(rows as u32, M as u32),
                self.dprojected.ptr(),
                self.projected.ptr(),
                self.dproduct.ptr(),
                length as u32,
            )?;
        }
        e.linear_back(
            self.dnormalized.ptr(),
            self.dprojected.ptr(),
            e.embed_norm.ptr(),
            self.input_weights,
            rows,
            3 * M,
            width,
        )?;
        unsafe {
            e.ops.add(
                grid(rows * width),
                self.dnormalized.ptr(),
                self.dnormalized.ptr(),
                e.dx.ptr(),
                (rows * width) as u32,
            )?;
        }
        Ok(self.dnormalized.ptr())
    }
}

impl Engine {
    pub(crate) fn input_product_snapshot(&self, rows: usize) -> Result<Vec<f32>> {
        let layer = self
            .input_product
            .as_ref()
            .ok_or("expected a causal input product")?;
        let mut features = layer.features.read()?;
        features.truncate(rows * self.config.width);
        Ok(features)
    }
}
