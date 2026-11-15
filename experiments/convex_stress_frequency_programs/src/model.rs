//! Fixed training-selected latent program and decoder, followed by CM56's
//! actual affine complex CE head. No task identity enters prediction.
use crate::{
    coordinates::{Fit, Record},
    discovery::{Learned, Report},
    head::Head,
    training::{Outcome, Policy},
    *,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Prior {
    Coordinates,
    Categorical,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub learned: Learned,
    pub prior: Prior,
    pub context: usize,
    pub head: Head,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trained {
    pub model: Model,
    pub training: Outcome,
}

pub fn records(
    learned: &Learned,
    coordinates: &[f64],
    input: &Input,
    prior: Prior,
) -> Result<Vec<Record>> {
    input
        .observations
        .iter()
        .enumerate()
        .map(|(id, row)| {
            let atom = learned.program.run(&row.tokens).atom;
            let (feature, identified) =
                learned
                    .decoder
                    .center(coordinates, atom, prior == Prior::Coordinates);
            if !identified {
                return Err("selected decoder refuses an observed training state".into());
            }
            Ok(Record {
                id,
                feature,
                label: row.label,
            })
        })
        .collect()
}

pub fn train(report: &Report, prior: Prior, policy: Policy) -> Result<Trained> {
    let index = match prior {
        Prior::Coordinates => report.geometry_selection,
        Prior::Categorical => report.categorical_selection,
    }
    .ok_or("no training-compatible candidate for requested prior")?;
    let learned = report.retained[index].clone();
    let rows = records(&learned, &report.coordinates, &report.input, prior)?;
    let fit = Fit::new(report.coordinates.clone(), &rows)?;
    let training = training::train(fit, &rows, policy)?;
    let model = Model {
        learned,
        prior,
        context: report.input.context,
        head: training.head.clone(),
    };
    Ok(Trained { model, training })
}

impl Model {
    pub fn feature(&self, tokens: &[usize]) -> (f64, bool) {
        let atom = self.learned.program.run(tokens).atom;
        self.learned.decoder.center(
            &self.head.fit.coordinates,
            atom,
            self.prior == Prior::Coordinates,
        )
    }
    pub fn independent_logits(&self, tokens: &[usize]) -> Vec<f64> {
        let atom = self.learned.program.interpret(tokens);
        let (x, _) = self.learned.decoder.center(
            &self.head.fit.coordinates,
            atom,
            self.prior == Prior::Coordinates,
        );
        self.head.independent_logits(x)
    }
    pub fn logits(&self, tokens: &[usize]) -> Vec<f64> {
        self.head.logits(self.feature(tokens).0)
    }
}

impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.head.fit.coordinates.len()
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        assert!(prefix.iter().all(|&x| x >= 0));
        self.logits(&prefix.iter().map(|&x| x as usize).collect::<Vec<_>>())
    }
}
