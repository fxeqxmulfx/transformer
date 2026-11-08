// Buffer accounting adapted from forward_training/instrumentation.rs at
// c9995ba5. The source CE, backward and AdamW functions are unchanged.
impl Engine {
    pub fn allocated_bytes(&self) -> usize {
        let floats = [
            &self.parameters,
            &self.gradients,
            &self.first,
            &self.second,
            &self.embed,
            &self.x0,
            &self.n1,
            &self.gv,
            &self.hidden,
            &self.branch,
            &self.branch_norm,
            &self.x1,
            &self.last,
            &self.logits,
            &self.dlogits,
            &self.dlast,
            &self.dx1,
            &self.dbranch,
            &self.dhidden,
            &self.dgv,
            &self.dn1,
            &self.dx0,
            &self.dembed,
            &self.losses,
            &self.scratch,
        ]
        .iter()
        .map(|b| b.len())
        .sum::<usize>();
        let words = [
            &self.pairs,
            &self.train,
            &self.validation,
            &self.order,
            &self.identity,
            &self.tokens,
            &self.labels,
            &self.hits,
        ]
        .iter()
        .map(|b| b.len())
        .sum::<usize>();
        4 * (floats + words)
    }
    pub fn moment_bytes(&self) -> usize {
        4 * (self.first.len() + self.second.len())
    }

    pub fn compact_moments(&mut self) -> Result<()> {
        if self.arm != Arm::Swiglu {
            let cuda = Cuda::new()?;
            // Temporary source allocation is replaced during charged setup.
            self.first = cuda.alloc(crate::layout::COUNT)?;
            self.second = cuda.alloc(crate::layout::COUNT)?;
        }
        Ok(())
    }
    pub fn restore_compact(&mut self, state: &State) -> Result<()> {
        self.parameters.write(&state.parameters)?;
        self.first
            .write(&crate::layout::pack(self.arm, &state.first))?;
        self.second
            .write(&crate::layout::pack(self.arm, &state.second))?;
        Ok(())
    }
    pub fn compact_update(
        &self,
        step: usize,
        kernels: &crate::fast_kernel::kernels::Kernels,
    ) -> Result<()> {
        if step == 0 {
            return Err("updates start at one".into());
        }
        let lr = (0.001 * (step as f64 / 10.).min(1.)) as f32;
        let bc1 = (1. - 0.9f64.powf(step as f64)) as f32;
        let bc2 = (1. - 0.98f64.powf(step as f64)).sqrt() as f32;
        let ratio = lr / bc1;
        let decay = 1. - lr * 1.;
        // SAFETY: dense moments match source storage; complex moments hold
        // exactly two scalars per map owner. Original gradients and expanded
        // parameters keep their source sizes; every parameter alias is unique.
        unsafe {
            if self.arm == Arm::Swiglu {
                kernels.dense(
                    grid(self.parameters.len()),
                    self.parameters.ptr(),
                    self.first.ptr(),
                    self.second.ptr(),
                    self.gradients.ptr(),
                    self.parameters.len() as u32,
                    ratio,
                    bc2,
                    decay,
                )?;
            } else {
                kernels.complex(
                    grid(self.pairs.len() / 5),
                    self.parameters.ptr(),
                    self.first.ptr(),
                    self.second.ptr(),
                    self.gradients.ptr(),
                    self.pairs.ptr(),
                    (self.pairs.len() / 5) as u32,
                    ratio,
                    bc2,
                    decay,
                )?;
            }
        }
        Ok(())
    }
}
