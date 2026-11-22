//! Decode with the transpose of the learned value projection and add both parameter gradients.
use super::*;

pub(super) struct TiedProjection {
    weights: usize,
    coordinates: usize,
    output_gradient: F,
}

impl TiedProjection {
    pub(super) fn new(e: &Engine, weights: usize, coordinates: usize) -> Result<Self> {
        Ok(Self {
            weights,
            coordinates,
            output_gradient: e.cuda.alloc(coordinates * e.config.width)?,
        })
    }

    pub(super) fn forward(&self, e: &Engine, out: P, read: P, rows: usize) -> Result<()> {
        let d = e.config.width;
        let weights = e
            .parameters
            .at(self.weights)
            .ok_or("missing tied value matrix")?;
        if crate::precision::FP32_AFFINE_REFERENCE {
            e.mm_backward(out, read, weights, rows, d, self.coordinates)
        } else {
            e.pipeline
                .as_ref()
                .ok_or("missing mixed-precision forward pipeline")?
                .multiply::<{ crate::precision::BF16 }>(
                    out,
                    read,
                    weights,
                    rows,
                    d,
                    self.coordinates,
                )
        }
    }

    pub(super) fn backward(&self, e: &Engine, dread: P, read: P, rows: usize) -> Result<()> {
        let d = e.config.width;
        let weights = e
            .parameters
            .at(self.weights)
            .ok_or("missing tied value matrix")?;
        e.transpose(weights, self.coordinates, d)?;
        e.mm_backward(
            dread,
            e.dlast.ptr(),
            e.scratch.ptr(),
            rows,
            self.coordinates,
            d,
        )?;
        e.transpose(read, rows, self.coordinates)?;
        e.mm_backward(
            self.output_gradient.ptr(),
            e.scratch.ptr(),
            e.dlast.ptr(),
            self.coordinates,
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
