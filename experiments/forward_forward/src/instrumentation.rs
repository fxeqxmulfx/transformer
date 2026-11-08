// Appended to the hash-pinned engine; source forward/optimizer stay identical.
impl Engine {
    pub fn local_features(&self, batch: usize) -> Result<Vec<f32>> {
        if batch == 0 || batch > 512 {
            return Err("invalid local batch".into());
        }
        Ok(self.last.read()?[..batch * W].to_vec())
    }

    /// Replace the loss/gradient buffers, with no propagation into the encoder.
    pub fn local_loss(
        &self,
        kernel: &crate::kernel::kernels::Kernels,
        batch: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if batch == 0 || batch > 512 {
            return Err("invalid local batch".into());
        }
        // SAFETY: source forward populated valid logits/labels; disjoint buffers
        // hold batch*97 gradients and batch losses. The same CUDA stream is used.
        unsafe {
            kernel.pairwise(
                rows_grid(batch),
                self.dlogits.ptr(),
                self.losses.ptr(),
                self.logits.ptr(),
                self.labels.ptr(),
                batch as u32,
            )?;
        }
        Ok((
            self.losses.read()?[..batch].to_vec(),
            self.dlogits.read()?[..batch * 97].to_vec(),
        ))
    }

    /// Partial derivative with respect to readout E, with `last` held fixed.
    /// Omits E's input-embedding contribution and never calls source backward.
    pub fn local_readout_gradient(&self, batch: usize) -> Result<Vec<f32>> {
        if batch == 0 || batch > 512 {
            return Err("invalid local batch".into());
        }
        self.transpose(self.dlogits.ptr(), batch, 97)?;
        self.mm(
            self.gradients.ptr(),
            self.scratch.ptr(),
            self.last.ptr(),
            97,
            W,
            batch,
        )?;
        Ok(self.gradients.read()?[..E].to_vec())
    }
}
