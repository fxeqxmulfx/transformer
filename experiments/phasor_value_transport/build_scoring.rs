//! Keep the pinned forward kernels and restrict reusable scores to frozen scoring.
pub fn prepare(engine: &mut String) {
    let buffers = r#"
#[derive(Clone)]
struct AttentionScores {
    probability: std::rc::Rc<F>,
    imaginary: std::rc::Rc<F>,
}

impl AttentionScores {
    fn new(cuda: &Cuda, cfg: &Config) -> Result<Self> {
        let count = cfg.max_batch * cfg.heads * cfg.max_length * cfg.max_length;
        Ok(Self {
            probability: std::rc::Rc::new(cuda.alloc(count)?),
            imaginary: std::rc::Rc::new(cuda.alloc(if layout::phase_score(cfg) { count } else { 1 })?),
        })
    }
}

"#;
    let constructors = r#"pub fn new(config: Config, parameters: &[f32]) -> Result<Self> {
        Self::with_score_storage(config, parameters, false)
    }

    pub(crate) fn for_scoring(config: Config, parameters: &[f32]) -> Result<Self> {
        Self::with_score_storage(config, parameters, true)
    }

    #[cfg(test)]
    pub(crate) fn attention_score_storage(&self) -> usize {
        let mut seen = std::collections::BTreeSet::new();
        let mut total = 0;
        for layer in &self.layers {
            total += layer.ds.len();
            for buffer in [&layer.prob, &layer.score_imaginary] {
                if seen.insert(std::rc::Rc::as_ptr(buffer)) {
                    total += buffer.len();
                }
            }
        }
        total
    }

    fn require_training_tape(&self) -> Result<()> {
        if self.scoring_only {
            return Err("this operation requires retained per-layer attention scores".into());
        }
        Ok(())
    }

    fn with_score_storage(config: Config, parameters: &[f32], scoring_only: bool) -> Result<Self> {"#;
    for (from, to) in [
        (
            "    score_imaginary: F,",
            "    score_imaginary: std::rc::Rc<F>,",
        ),
        ("    prob: F,", "    prob: std::rc::Rc<F>,"),
        (
            "hidden: usize, offset: usize, index: usize) -> Result<Self> {",
            "hidden: usize, offset: usize, index: usize, shared: Option<&AttentionScores>) -> Result<Self> {",
        ),
        (
            "let scores = cfg.max_batch * cfg.heads * cfg.max_length * cfg.max_length;",
            "let scores = cfg.max_batch * cfg.heads * cfg.max_length * cfg.max_length;\n        let forward_scores = match shared { Some(s) => s.clone(), None => AttentionScores::new(cuda, cfg)?, };",
        ),
        (
            "score_imaginary: cuda.alloc(if layout::phase_score(cfg) { scores } else { 1 })?,",
            "score_imaginary: forward_scores.imaginary,",
        ),
        (
            "prob: cuda.alloc(scores)?,",
            "prob: forward_scores.probability,",
        ),
        (
            "ds: cuda.alloc(scores)?,",
            "ds: cuda.alloc(if shared.is_some() { 1 } else { scores })?,",
        ),
        (
            "pub struct Engine {",
            "pub struct Engine {\n    scoring_only: bool,",
        ),
        (
            "fn backward_original(&self, batch: usize, length: usize) -> Result<()> {",
            "fn backward_original(&self, batch: usize, length: usize) -> Result<()> {\n        self.require_training_tape()?;",
        ),
        (
            "pub fn update(&self, completed: usize) -> Result<()> {",
            "pub fn update(&self, completed: usize) -> Result<()> {\n        self.require_training_tape()?;",
        ),
        (
            "pub fn new(config: Config, parameters: &[f32]) -> Result<Self> {",
            constructors,
        ),
        (
            "let mut layers = Vec::new();",
            "let shared_scores = scoring_only.then(|| AttentionScores::new(&cuda, &config)).transpose()?;\n        let mut layers = Vec::new();",
        ),
        (
            "                l,\n            )?);",
            "                l,\n                shared_scores.as_ref(),\n            )?);",
        ),
        (
            "let mut engine = Self {",
            "let mut engine = Self {\n            scoring_only,",
        ),
        (
            "let mut out = std::collections::BTreeMap::new();",
            "self.require_training_tape()?;\n        let mut out = std::collections::BTreeMap::new();",
        ),
    ] {
        assert_eq!(
            engine.matches(from).count(),
            1,
            "pinned score allocation changed: {from}"
        );
        *engine = engine.replace(from, to);
    }
    engine.insert_str(engine.find("struct Layer {").unwrap(), buffers);
}
