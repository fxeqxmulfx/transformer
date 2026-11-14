//! A training-learned affine slice of CM53's joint tensor domain. Decoder
//! templates are fixed before CE updates; gamma[j,s,c]=w[j]*template[j,s,c].
//! This does not optimize two independently learned factors or claim the full
//! domain is restricted to deterministic completions.
use crate::{
    head::Head,
    program::{Program, Value},
    *,
};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub head: Head,
    pub context: usize,
    pub alphabet: Vec<usize>,
    pub programs: Vec<Program>,
    pub weights: Vec<f64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Encoded {
    pub states: Vec<usize>,
    pub label: usize,
}
impl Model {
    pub fn new(report: &discovery::Report) -> Result<Self> {
        if report.error.is_some() || report.retained.is_empty() {
            return Err("refused latent filter has no complete model".into());
        }
        let out = Self {
            head: Head::new(
                report.input.vocabulary,
                report.retained.iter().map(|p| p.decoder.clone()).collect(),
            )?,
            context: report.input.context,
            alphabet: report.alphabet.clone(),
            programs: report.retained.iter().map(|p| p.program.clone()).collect(),
            weights: vec![1. / report.retained.len() as f64; report.retained.len()],
        };
        out.validate()?;
        Ok(out)
    }
    pub fn validate(&self) -> Result<()> {
        self.head.validate()?;
        self.check_weights()?;
        let order = program::catalogue(&self.alphabet, self.head.vocabulary)?;
        let indices = self
            .programs
            .iter()
            .map(|p| order.iter().position(|q| q == p))
            .collect::<Option<Vec<_>>>()
            .ok_or("program outside fixed latent bank")?;
        if indices.windows(2).any(|w| w[0] >= w[1])
            || self.programs.len() != self.head.decoders.len()
        {
            return Err("latent program order or dimensions changed".into());
        }
        Ok(())
    }
    pub fn check_weights(&self) -> Result<()> {
        if self.programs.is_empty()
            || self.programs.len() > 128
            || self.weights.len() != self.programs.len()
            || self.weights.iter().any(|w| !w.is_finite() || *w < 0.)
            || (self.weights.iter().sum::<f64>() - 1.).abs() > 1e-10
            || self.head.max_column_l1(&self.weights) > self.head.radius + 1e-8
        {
            return Err("invalid latent joint masses/budget".into());
        }
        Ok(())
    }
    pub fn states(&self, prefix: &[usize]) -> Vec<Value> {
        assert!(
            !prefix.is_empty()
                && prefix.len() <= 4096
                && prefix.iter().all(|&t| t < self.head.vocabulary)
        );
        self.programs.iter().map(|p| p.run(prefix)).collect()
    }
    pub fn labels(&self, states: &[usize]) -> Vec<Option<usize>> {
        self.head
            .decoders
            .iter()
            .zip(states)
            .map(|(d, &s)| d[s])
            .collect()
    }
    pub fn logits_states(&self, states: &[usize]) -> Vec<f64> {
        self.head.logits(states, &self.weights)
    }
    pub fn encode(&self, input: &Input) -> Result<Vec<Encoded>> {
        self.validate()?;
        if input.vocabulary != self.head.vocabulary
            || input.context != self.context
            || program::alphabet(input) != self.alphabet
        {
            return Err("latent input dimensions changed".into());
        }
        input
            .observations
            .iter()
            .map(|row| {
                let states: Vec<_> = self.states(&row.tokens).iter().map(|v| v.state).collect();
                if self.labels(&states).iter().any(|&y| y != Some(row.label)) {
                    return Err("latent dictionary disagrees with original training labels".into());
                }
                Ok(Encoded {
                    states,
                    label: row.label,
                })
            })
            .collect()
    }
}
impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.head.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        assert!(prefix.iter().all(|&v| v >= 0));
        self.logits_states(
            &self
                .states(&prefix.iter().map(|&t| t as usize).collect::<Vec<_>>())
                .iter()
                .map(|v| v.state)
                .collect::<Vec<_>>(),
        )
    }
}
