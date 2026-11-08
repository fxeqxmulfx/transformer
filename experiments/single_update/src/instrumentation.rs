// Appended to the pinned engine module by build.rs. No optimizer equations change.
impl Engine {
    /// Capture readout gradient before the source backward adds the encoder path.
    /// Call immediately after forward; the final buffers equal ordinary backward.
    pub fn audit_gradient(&self, batch: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        if batch == 0 || batch > 512 {
            return Err("invalid audit batch".into());
        }
        self.linear_back(
            self.dlast.ptr(),
            self.dlogits.ptr(),
            self.last.ptr(),
            0,
            batch,
            97,
            W,
        )?;
        let readout = self.gradients.read()?[..E]
            .iter()
            .map(|&x| x as f64)
            .collect();
        self.backward(batch)?;
        let full = self.gradients.read()?[..E]
            .iter()
            .map(|&x| x as f64)
            .collect();
        Ok((readout, full))
    }

    /// Raw pre-final-RMS states and the normalized features used by source backward.
    pub fn audit_features(&self, batch: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        if batch == 0 || batch > 512 {
            return Err("invalid audit batch".into());
        }
        let raw = self.x1.read()?[..batch * W]
            .iter()
            .map(|&x| x as f64)
            .collect();
        let normalized = self.last.read()?[..batch * W]
            .iter()
            .map(|&x| x as f64)
            .collect();
        Ok((raw, normalized))
    }
}
