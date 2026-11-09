// Operand-only source prefix through logits, copied from trainer 14cff3b5.
// Same GEMM/activation/RMS operations; cached transposes change no arithmetic.
impl Engine {
    pub fn weight_cache(&self, version: &str) -> Result<crate::device::Weights> {
        let cuda = Cuda::new()?;
        let h = self.arm.hidden();
        let out = crate::device::Weights {
            version: version.into(),
            first: cuda.alloc(2 * h * W)?,
            second: cuda.alloc(W * h)?,
            readout: cuda.alloc(E)?,
        };
        // SAFETY: exact transpose shapes, disjoint buffers, original stream.
        unsafe {
            self.ops.transpose(
                grid(2 * h * W),
                out.first.ptr(),
                self.parameters.at(E),
                (2 * h) as u32,
                W as u32,
            )?;
            self.ops.transpose(
                grid(W * h),
                out.second.ptr(),
                self.parameters.at(E + 2 * h * W),
                W as u32,
                h as u32,
            )?;
            self.ops.transpose(
                grid(E),
                out.readout.ptr(),
                self.parameters.ptr(),
                97,
                W as u32,
            )?;
        }
        Ok(out)
    }
    pub fn inference(
        &mut self,
        pairs: &[[u32; 2]],
        cache: Option<&crate::device::Weights>,
        version: &str,
        io: &crate::io::Transfers<'_>,
    ) -> Result<Vec<f32>> {
        crate::device::valid_pairs(pairs)?;
        if let Some(c) = cache {
            c.validate(version)?;
        }
        io.write(&mut self.tokens, pairs.as_flattened())?;
        let batch = pairs.len();
        let h = self.arm.hidden();
        let n = batch * W;
        unsafe {
            self.rotate.embed(
                grid(n),
                self.embed.ptr(),
                self.parameters.ptr(),
                self.tokens.ptr(),
                n as u32,
            )?;
        }
        self.rms(self.x0.ptr(), self.embed.ptr(), batch)?;
        self.rms(self.n1.ptr(), self.x0.ptr(), batch)?;
        if let Some(c) = cache {
            self.mm(self.gv.ptr(), self.n1.ptr(), c.first.ptr(), batch, 2 * h, W)?;
        } else {
            self.linear(self.gv.ptr(), self.n1.ptr(), E, batch, 2 * h, W)?;
        }
        unsafe {
            match self.arm {
                Arm::Swiglu => self.ops.swiglu(
                    grid(batch * h),
                    self.hidden.ptr(),
                    self.gv.ptr(),
                    h as u32,
                    (batch * h) as u32,
                )?,
                Arm::Diagonal => self.product.product::<0>(
                    grid(batch * h / 2),
                    self.hidden.ptr(),
                    self.gv.ptr(),
                    (h / 2) as u32,
                    (batch * h / 2) as u32,
                )?,
                Arm::Product => self.product.product::<1>(
                    grid(batch * h / 2),
                    self.hidden.ptr(),
                    self.gv.ptr(),
                    (h / 2) as u32,
                    (batch * h / 2) as u32,
                )?,
                Arm::Plain => self.product.product::<2>(
                    grid(batch * h / 2),
                    self.hidden.ptr(),
                    self.gv.ptr(),
                    (h / 2) as u32,
                    (batch * h / 2) as u32,
                )?,
            }
        }
        if let Some(c) = cache {
            self.mm(
                self.branch.ptr(),
                self.hidden.ptr(),
                c.second.ptr(),
                batch,
                W,
                h,
            )?;
        } else {
            self.linear(
                self.branch.ptr(),
                self.hidden.ptr(),
                E + 2 * h * W,
                batch,
                W,
                h,
            )?;
        }
        self.rms(self.branch_norm.ptr(), self.branch.ptr(), batch)?;
        unsafe {
            self.ops.add(
                grid(n),
                self.x1.ptr(),
                self.x0.ptr(),
                self.branch_norm.ptr(),
                n as u32,
            )?;
        }
        self.rms(self.last.ptr(), self.x1.ptr(), batch)?;
        if let Some(c) = cache {
            self.mm(
                self.logits.ptr(),
                self.last.ptr(),
                c.readout.ptr(),
                batch,
                97,
                W,
            )?;
        } else {
            self.linear(self.logits.ptr(), self.last.ptr(), 0, batch, 97, W)?;
        }
        let logits = io.read(&self.logits, batch * 97)?;
        crate::inputs::require(
            logits.iter().all(|x| x.is_finite()),
            "nonfinite inference logits",
        )?;
        Ok(logits)
    }
    pub fn inference_features(
        &self,
        batch: usize,
        io: &crate::io::Transfers<'_>,
    ) -> Result<Vec<f32>> {
        crate::inputs::require(batch > 0 && batch <= 512, "invalid feature batch")?;
        io.read(&self.last, batch * W)
    }
    pub fn reused_logits(
        &mut self,
        pairs: &[[u32; 2]],
        features: &F,
        cache: &crate::device::Weights,
        version: &str,
        kernels: &crate::gather_gpu::kernels::Kernels,
        io: &crate::io::Transfers<'_>,
    ) -> Result<Vec<f32>> {
        crate::device::valid_pairs(pairs)?;
        cache.validate(version)?;
        crate::inputs::require(features.len() == 9409 * W, "invalid old feature table")?;
        io.write(&mut self.tokens, pairs.as_flattened())?;
        let batch = pairs.len();
        unsafe {
            kernels.gather(
                grid(batch * W),
                self.last.ptr(),
                features.ptr(),
                self.tokens.ptr(),
                (batch * W) as u32,
            )?;
        }
        self.mm(
            self.logits.ptr(),
            self.last.ptr(),
            cache.readout.ptr(),
            batch,
            97,
            W,
        )?;
        let logits = io.read(&self.logits, batch * 97)?;
        crate::inputs::require(
            logits.iter().all(|x| x.is_finite()),
            "nonfinite reused logits",
        )?;
        Ok(logits)
    }
}
