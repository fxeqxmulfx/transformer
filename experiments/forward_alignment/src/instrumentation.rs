// Reuse c9995ba5 exactly. Only diagnostic accessors are added here.
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../forward_training/src/instrumentation.rs"
));

impl Engine {
    pub fn pair_scores(&self, kernels: &Kernels, batch: usize) -> Result<Metrics> {
        self.pair_error(kernels, batch)?;
        self.metrics(batch)
    }
    /// Call immediately after terminal backward. Read only its input-E path;
    /// addition from zero has a different rounding order from the full path.
    pub fn input_component(&mut self, batch: usize) -> Result<Vec<f32>> {
        self.gradients.zero()?;
        // SAFETY: backward supplied dembed and matching tokens, batch <=512.
        unsafe {
            self.rotate.embed_bwd(
                grid(E),
                self.gradients.ptr(),
                self.dembed.ptr(),
                self.tokens.ptr(),
                batch as u32,
            )?;
        }
        Ok(self.gradients.read()?)
    }
    /// Isolate vocabulary E without subtracting two rounded gradient sums.
    pub fn vocabulary_component(&mut self, aux: &Aux, kernels: &Kernels) -> Result<Vec<f32>> {
        self.gradients.zero()?;
        self.vocabulary_objective(aux, kernels)?;
        Ok(self.gradients.read()?)
    }
}
