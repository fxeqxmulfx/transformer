// Included in the pinned engine. The backbone sees dense features and sequence
// dimensions only. The tied class projection and its loss are external operations.
pub struct EmbeddingOutput {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

impl Engine {
    fn uses_embedding_backbone(&self) -> bool {
        self.mhc.is_some() && self.config.head != crate::head::Head::Relations
    }

    fn encode_embedding_sequence(&mut self, input: P, batch: usize, length: usize) -> Result<()> {
        self.supervised_shape = None;
        let rows = batch * length;
        self.prepare_positions(length)?;
        self.rms(self.embed_norm.ptr(), input, rows)?;
        let features = match &self.input_product {
            Some(layer) => layer.forward(self, rows, length)?,
            None => self.embed_norm.ptr(),
        };
        self.forward_mhc(features, batch, length)?;
        self.rms(self.last.ptr(), self.shared.ptr(), rows)?;
        if let Some(memory) = &self.hidden_layer {
            memory.forward_features(self, rows, length)?;
        }
        Ok(())
    }

    fn backward_input_rms(&self, input: P, rows: usize, length: usize) -> Result<()> {
        let gradient = match &self.input_product {
            Some(layer) => layer.backward(self, rows, length)?,
            None => self.dx.ptr(),
        };
        if self.config.head.projects_embeddings() {
            self.hidden_layer.as_ref().ok_or("missing input phase read")?
                .add_projection_gradient(self, gradient, rows)?;
        }
        self.rms_back(self.dembed.ptr(), input, gradient, rows)
    }

    fn encoded_features(&self) -> &F {
        self.hidden_layer
            .as_ref()
            .map_or(&self.last, |m| m.features())
    }

    fn classify_features(&self, rows: usize) -> Result<()> {
        self.linear(
            self.logits.ptr(),
            self.encoded_features().ptr(),
            0,
            rows,
            self.config.vocab,
            self.config.width,
        )
    }

    fn forward_vector_batch(&mut self, input: &[f32], batch: usize, length: usize) -> Result<F> {
        if !self.uses_embedding_backbone() {
            return Err("the selected historical head requires token-addressed features".into());
        }
        let elements = batch
            .checked_mul(length)
            .and_then(|r| r.checked_mul(self.config.width));
        if batch == 0
            || length == 0
            || batch > self.config.max_batch
            || length > self.config.max_length
            || elements != Some(input.len())
            || !input.iter().all(|v| v.is_finite())
        {
            return Err("embedding input must contain a finite vector of the configured width at each position".into());
        }
        let rows = batch * length;
        let device_input = self.cuda.upload(input)?;
        self.encode_embedding_sequence(device_input.ptr(), batch, length)?;
        self.classify_features(rows)?;
        Ok(device_input)
    }

    /// Run the same embedding-input backbone used in training on arbitrary real vectors.
    pub fn infer_embeddings(
        &mut self,
        input: &[f32],
        batch: usize,
        length: usize,
    ) -> Result<EmbeddingOutput> {
        let _input = self.forward_vector_batch(input, batch, length)?;
        let mut hidden = self.encoded_features().read()?;
        hidden.truncate(input.len());
        let mut logits = self.logits.read()?;
        logits.truncate(batch * length * self.config.vocab);
        Ok(EmbeddingOutput { hidden, logits })
    }

    /// Differentiate a caller-supplied scalar pairing with the logits with respect
    /// to the input vectors. This uses the ordinary reverse pass up to the vectors.
    pub(crate) fn embedding_pullback(
        &mut self,
        input: &[f32],
        batch: usize,
        length: usize,
        cotangent: &[f32],
    ) -> Result<Vec<f32>> {
        let device_input = self.forward_vector_batch(input, batch, length)?;
        let rows = batch * length;
        if cotangent.len() != rows * self.config.vocab || !cotangent.iter().all(|v| v.is_finite()) {
            return Err("logit cotangent has the wrong shape or a nonfinite value".into());
        }
        let mut padded = vec![0.; self.dlogits.len()];
        padded[..cotangent.len()].copy_from_slice(cotangent);
        self.dlogits.write(&padded)?;
        self.output_backward(rows, length)?;
        self.rms_back(
            self.dshared.ptr(),
            self.shared.ptr(),
            self.dlast.ptr(),
            rows,
        )?;
        self.backward_embedding_layers(device_input.ptr(), batch, length)?;
        let mut gradient = self.dembed.read()?;
        gradient.truncate(input.len());
        Ok(gradient)
    }
}
