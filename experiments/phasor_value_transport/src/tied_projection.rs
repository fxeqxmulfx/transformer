//! Decode with the transpose of the learned value projection and add both parameter gradients.
use super::*;

pub(super) struct TiedProjection {
    weights: usize,
    output_gradient: F,
}

impl TiedProjection {
    pub(super) fn new(e: &Engine, weights: usize) -> Result<Self> {
        Ok(Self {
            weights,
            output_gradient: e.cuda.alloc(2 * e.config.width * e.config.width)?,
        })
    }

    pub(super) fn forward(&self, e: &Engine, out: P, read: P, rows: usize) -> Result<()> {
        let d = e.config.width;
        let weights = e
            .parameters
            .at(self.weights)
            .ok_or("missing tied value matrix")?;
        if crate::precision::FP32_AFFINE_REFERENCE {
            e.mm_backward(out, read, weights, rows, d, 2 * d)
        } else {
            e.pipeline
                .as_ref()
                .ok_or("missing mixed-precision forward pipeline")?
                .multiply::<{ crate::precision::BF16 }>(out, read, weights, rows, d, 2 * d)
        }
    }

    pub(super) fn backward(&self, e: &Engine, dread: P, read: P, rows: usize) -> Result<()> {
        let d = e.config.width;
        let weights = e
            .parameters
            .at(self.weights)
            .ok_or("missing tied value matrix")?;
        e.transpose(weights, 2 * d, d)?;
        e.mm_backward(dread, e.dlast.ptr(), e.scratch.ptr(), rows, 2 * d, d)?;
        e.transpose(read, rows, 2 * d)?;
        e.mm_backward(
            self.output_gradient.ptr(),
            e.scratch.ptr(),
            e.dlast.ptr(),
            2 * d,
            d,
            rows,
        )
    }

    pub(super) fn accumulate(&self, e: &Engine) -> Result<()> {
        let elements = self.output_gradient.len();
        let grad = e
            .gradients
            .at(self.weights)
            .ok_or("missing tied value gradient")?;
        unsafe {
            e.ops.add(
                grid(elements),
                grad,
                grad,
                self.output_gradient.ptr(),
                elements as u32,
            )?;
        }
        Ok(())
    }
}
