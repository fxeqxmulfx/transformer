// Reuse the exact local objectives; append only the new A update and the
// source optimizer's E/B range launches. A moments are neither read nor written.
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../forward_training/src/instrumentation.rs"
));
impl Engine {
    pub fn normalized_update(
        &self,
        step: usize,
        normalizer: &crate::normalized::Normalizer,
        kernels: &crate::normalized_kernel::kernels::Kernels,
    ) -> Result<()> {
        if step == 0 {
            return Err("updates start at one".into());
        }
        self.normalized_a_only(step, normalizer, kernels)?;
        let lr = (0.001 * (step as f64 / 10.).min(1.)) as f32;
        let bc1 = (1. - 0.9f64.powf(step as f64)) as f32;
        let bc2 = (1. - 0.98f64.powf(step as f64)).sqrt() as f32;
        // SAFETY: E/B are disjoint from A. The exact source kernels receive
        // unchanged pre-update gradients and the original clock/constants.
        unsafe {
            if self.arm == Arm::Swiglu {
                let b = E + crate::normalized::A_COUNT;
                for (offset, count) in [(0, E), (b, self.parameters.len() - b)] {
                    self.ops.adamw(
                        grid(count),
                        self.parameters.at(offset),
                        self.first.at(offset),
                        self.second.at(offset),
                        self.gradients.at(offset),
                        count as u32,
                        lr,
                        1.,
                        bc1,
                        bc2,
                    )?;
                }
            } else {
                let b = (E + crate::normalized::A_COUNT) / 2;
                for (offset, count) in [(0, E / 2), (b, self.pairs.len() / 5 - b)] {
                    self.product.adamw::<false>(
                        grid(count),
                        self.parameters.ptr(),
                        self.first.ptr(),
                        self.second.ptr(),
                        self.gradients.ptr(),
                        self.pairs.at(offset * 5),
                        count as u32,
                        lr,
                        1.,
                        bc1,
                        bc2,
                    )?;
                }
            }
        }
        Ok(())
    }
    pub fn normalized_a_only(
        &self,
        step: usize,
        normalizer: &crate::normalized::Normalizer,
        kernels: &crate::normalized_kernel::kernels::Kernels,
    ) -> Result<()> {
        // Shared implementation; edge-case audits omit the E/B update.
        let threads = Grid::new(normalizer.count.div_ceil(128) as u32, 128);
        let mode = u32::from(self.arm != Arm::Swiglu);
        let map = self.pairs.at(E / 2 * 5);
        // SAFETY: fixed A sizes are whole warps; the original map covers A.
        // Scratch holds two FP64 energies per warp and four scalar statistics.
        unsafe {
            kernels.energies(
                threads,
                normalizer.partial.ptr(),
                self.parameters.ptr(),
                self.gradients.ptr(),
                map,
                normalizer.count as u32,
                mode,
            )?;
            kernels.factor(
                Grid::new(1, 32),
                normalizer.stats.ptr(),
                normalizer.partial.ptr(),
                (normalizer.count / 32) as u32,
                crate::normalized::alpha(step),
            )?;
            kernels.apply(
                threads,
                self.parameters.ptr(),
                self.gradients.ptr(),
                map,
                normalizer.stats.ptr(),
                normalizer.count as u32,
                mode,
            )?;
        }
        Ok(())
    }
}
