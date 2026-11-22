//! Share the pinned positional tables and cross-entropy with the embedding-input forward.
pub fn extract(engine: &mut String) -> String {
    let positions_start = engine
        .find("        if self.rope_length != length {")
        .unwrap();
    let positions_end = positions_start
        + engine[positions_start..]
            .find("        unsafe {\n            self.seq.prefix(")
            .unwrap();
    let positions = engine[positions_start..positions_end].to_owned();
    engine.replace_range(
        positions_start..positions_end,
        "        if !self.uses_embedding_backbone() { self.prepare_positions(length)?; }\n",
    );

    let targets = "        self.host_labels[..r].copy_from_slice(labels);\n        self.labels.write(&self.host_labels)?;\n";
    assert_eq!(engine.matches(targets).count(), 1);
    *engine = engine.replace(targets, "");

    let loss_start = engine.find("        let normalizer = labels").unwrap();
    let loss_end = loss_start
        + engine[loss_start..]
            .find("\n    }\n\n    pub fn backward(")
            .unwrap();
    let loss = engine[loss_start..loss_end].to_owned();
    engine.replace_range(
        loss_start..loss_end,
        "        self.loss_for_logits(labels, r)",
    );

    let begin = "        let mut x = self.embed.ptr();";
    assert_eq!(engine.matches(begin).count(), 1);
    *engine = engine.replace(begin, &format!(
        "        if self.uses_embedding_backbone() {{\n            self.encode_embedding_sequence(self.embed.ptr(), batch, length)?;\n            self.classify_features(r)?;\n            self.loss_for_logits(labels, r)?;\n            self.supervised_shape = Some((batch, length));\n            return Ok(());\n        }}\n{begin}"
    ));

    format!(
        "\nimpl Engine {{\n    fn prepare_positions(&mut self, length: usize) -> Result<()> {{\n        let w = self.config.width;\n{positions}        Ok(())\n    }}\n    fn loss_for_logits(&mut self, labels: &[u32], r: usize) -> Result<()> {{\n{targets}{loss}\n    }}\n}}\n"
    )
}

pub fn separate_input_gradient(engine: &mut String) {
    let old = "fn backward_mhc(&self, batch: usize, length: usize) -> Result<()> {";
    assert_eq!(engine.matches(old).count(), 1);
    *engine = engine.replace(
        old,
        "fn backward_embedding_layers(&self, input: P, batch: usize, length: usize) -> Result<()> {",
    );
    let final_rms =
        "        self.rms_back(self.dembed.ptr(), self.embed.ptr(), self.dx.ptr(), r)?;";
    assert_eq!(engine.matches(final_rms).count(), 1);
    *engine = engine.replace(final_rms, "        self.backward_input_rms(input, r, length)\n    }\n    fn backward_mhc(&self, batch: usize, length: usize) -> Result<()> {\n        self.backward_embedding_layers(self.embed.ptr(), batch, length)?;\n        let (r, w) = (batch * length, self.config.width);");
}
