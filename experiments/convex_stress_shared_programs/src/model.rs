//! Actual mixed code rows followed by the complete fixed complex head (CM52).
//! Numerical argmax and program consensus are separate reported quantities.
use crate::{
    head::Head,
    program::{Program, Value},
    *,
};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub head: Head,
    pub context: usize,
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
            return Err("refused program filter has no complete model".into());
        }
        let out = Self {
            head: Head::new(report.input.vocabulary)?,
            context: report.input.context,
            programs: report.retained.clone(),
            weights: vec![1. / report.retained.len() as f64; report.retained.len()],
        };
        out.validate()?;
        Ok(out)
    }
    pub fn validate(&self) -> Result<()> {
        self.head.validate()?;
        self.check_weights()?;
        let catalogue = program::catalogue(self.head.vocabulary, self.context)?;
        let indices = self
            .programs
            .iter()
            .map(|p| catalogue.iter().position(|c| c == p))
            .collect::<Option<Vec<_>>>()
            .ok_or("program outside fixed input catalogue")?;
        if indices.windows(2).any(|w| w[0] >= w[1]) {
            return Err("program order or uniqueness changed".into());
        }
        Ok(())
    }
    pub fn check_weights(&self) -> Result<()> {
        if self.programs.is_empty()
            || self.programs.len() > 1024
            || self.weights.len() != self.programs.len()
            || self.weights.iter().any(|w| !w.is_finite() || *w < 0.)
            || (self.weights.iter().sum::<f64>() - 1.).abs() > 1e-10
        {
            return Err("invalid compatible simplex weights".into());
        }
        Ok(())
    }
    pub fn states(&self, prefix: &[usize]) -> Vec<Value> {
        assert!(
            !prefix.is_empty()
                && prefix.len() <= 4096
                && prefix.iter().all(|&x| x < self.head.vocabulary)
        );
        self.programs.iter().map(|p| p.run(prefix)).collect()
    }
    pub fn features(&self, states: &[usize], weights: &[f64]) -> Vec<C> {
        assert_eq!(states.len(), self.programs.len());
        assert_eq!(weights.len(), states.len());
        let n = self.head.vocabulary;
        let mut phi = vec![C::new(0., 0.); n];
        for (&state, &w) in states.iter().zip(weights) {
            for (f, value) in phi.iter_mut().enumerate() {
                *value += self.head.code[state * n + f] * w;
            }
        }
        phi
    }
    pub fn logits_states(&self, states: &[usize]) -> Vec<f64> {
        self.head.logits(&self.features(states, &self.weights))
    }
    pub fn logits_tokens(&self, tokens: &[usize]) -> Vec<f64> {
        self.logits_states(
            &self
                .states(tokens)
                .iter()
                .map(|v| v.state)
                .collect::<Vec<_>>(),
        )
    }
    pub fn encode(&self, input: &Input) -> Result<Vec<Encoded>> {
        self.validate()?;
        if input.vocabulary != self.head.vocabulary || input.context != self.context {
            return Err("original input dimensions changed".into());
        }
        input
            .observations
            .iter()
            .map(|s| {
                let values = self.states(&s.tokens);
                if values.iter().any(|v| v.state != s.label) {
                    return Err(
                        "model contains a program inconsistent with original training labels"
                            .into(),
                    );
                }
                Ok(Encoded {
                    states: values.iter().map(|v| v.state).collect(),
                    label: s.label,
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
        self.logits_tokens(&prefix.iter().map(|&v| v as usize).collect::<Vec<_>>())
    }
}
