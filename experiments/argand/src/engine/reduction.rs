//! Fixed-order sums for the gradients shared across tokens and attention heads.
use super::{F, P, Result, kernels::device::Kernels};
use kernite_cuda::{Cuda, Grid};

pub(super) struct OrderedGradients {
    pub terms: F,
    partials: F,
}

fn partitions(rows: usize) -> usize {
    rows.div_ceil(256).clamp(1, 128)
}

impl OrderedGradients {
    pub fn new(cuda: &Cuda, rows: usize, width: usize) -> Result<Self> {
        let count = rows.checked_mul(width).ok_or("oversized gradient buffer")?;
        u32::try_from(count)?;
        Ok(Self {
            terms: cuda.alloc(count)?,
            partials: cuda.alloc(partitions(rows) * width)?,
        })
    }

    /// Adds the column totals to an existing adjoint. Both launches follow
    /// the producer on the training stream and each output has a single writer.
    pub fn accumulate(&self, k: &Kernels, dst: P, rows: u32, width: u32) -> Result<()> {
        let parts = partitions(rows as usize) as u32;
        assert!(rows as usize * width as usize <= self.terms.len());
        assert!(parts as usize * width as usize <= self.partials.len());
        let launch = |parts| Grid {
            blocks: [width, parts, 1],
            threads: [32, 1, 1],
            shared: 0,
        };
        // SAFETY: both input shapes fit the corresponding scratch buffer;
        // dst is the real parameter array with one entry per column.
        unsafe {
            if parts == 1 {
                k.sum_columns::<true>(launch(1), dst, self.terms.ptr(), rows, width)?;
            } else {
                k.sum_columns::<false>(
                    launch(parts),
                    self.partials.ptr(),
                    self.terms.ptr(),
                    rows,
                    width,
                )?;
                k.sum_columns::<true>(launch(1), dst, self.partials.ptr(), parts, width)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::Engine, suite::Shape};

    #[test]
    #[ignore = "requires CUDA"]
    fn column_sums_preserve_existing_gradients_with_short_and_split_reductions() -> Result<()> {
        let cuda = Cuda::new()?;
        let kernel = Kernels::load(&cuda)?;
        for rows in [19, 1029, 40_017] {
            let width = 3;
            let mut sums = OrderedGradients::new(&cuda, rows, width)?;
            let input: Vec<_> = (0..rows * width).map(|i| (i % 37) as f32 - 18.0).collect();
            sums.terms.write(&input)?;
            let mut expected = [0.5, -1.5, 3.0];
            let dst = cuda.upload(&expected)?;
            for _ in 0..2 {
                sums.accumulate(&kernel, dst.ptr(), rows as u32, width as u32)?;
                for row in input.chunks_exact(width) {
                    for (total, value) in expected.iter_mut().zip(row) {
                        *total += value;
                    }
                }
                assert_eq!(dst.read()?, expected);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn full_model_backward_matches_atomic_gradients_and_repeats_exactly() -> Result<()> {
        let shape = Shape {
            width: 16,
            heads: 2,
            layers: 2,
            vocab: 7,
            max_batch: 48,
            max_length: 5,
        };
        let mut engine = Engine::new(shape, 2, 113)?;
        let tokens: Vec<_> = (0..240).map(|i| (i * 3 + i / 5) % 7).collect();
        let labels: Vec<_> = (0..240).map(|i| (i + i / 3) % 7).collect();
        let gradient = |e: &mut Engine| -> Result<Vec<f32>> {
            e.forward(&tokens, &labels, 48, 5)?;
            e.backward()?;
            Ok(e.parameters.grad.read()?)
        };
        let atomic = gradient(&mut engine)?;
        engine.order_shared_gradients()?;
        let ordered = gradient(&mut engine)?;
        assert_eq!(ordered, gradient(&mut engine)?);
        let start = engine.parameters.embedding().len;
        let ranges = std::iter::once(start..start + 8).chain(
            engine
                .parameters
                .layers
                .iter()
                .map(|layer| layer.score..layer.score + 2),
        );
        let mut shared = vec![false; atomic.len()];
        for range in ranges {
            let largest = atomic[range.clone()]
                .iter()
                .map(|x| x.abs())
                .fold(0.0, f32::max);
            assert!(largest > 0.0);
            for i in range {
                shared[i] = true;
                assert!((atomic[i] - ordered[i]).abs() <= largest * 1e-4);
            }
        }
        for (i, is_shared) in shared.iter().enumerate() {
            if !is_shared {
                assert_eq!(atomic[i], ordered[i]);
            }
        }
        Ok(())
    }
}
