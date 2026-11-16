//! Actual CM70 complete complex head with all retained programs. Unknown rows
//! have every raw class coordinate; uniform initialization is not a constraint.
//! Prediction receives only the visible prefix. Coverage is a diagnostic, not
//! a substitute for the complex contraction or a claim about the true answer.
use crate::{Result, Work, features, input::Prepared};
use joint::{
    Shape,
    admission::Budget,
    domain::Domain,
    records::Packed,
    training::{Config, Mode, Trained},
};
use probes::boundary::Predictor;

pub fn policy(mode: Mode) -> Result<Config> {
    if !matches!(mode, Mode::Forward | Mode::Mini) {
        return Err("this comparison requires forward or minibatch only".into());
    }
    Ok(Config {
        mode,
        epochs: usize::from(mode == Mode::Mini),
        batch_size: 127,
        learning_rate: 1e6,
        initial_mass: 8.,
    })
}
pub struct Compiled {
    pub domain: Domain,
    pub records: Packed,
    pub budget: Budget,
}
pub fn compile(input: &Prepared<'_>, work: &mut Work) -> Result<Option<Compiled>> {
    if input.partition.retained.is_empty() {
        return Ok(None);
    }
    let shape = Shape {
        programs: input.partition.retained.len(),
        states: features::STATES,
        classes: input.alphabet.vocabulary,
    };
    let budget = input.budget()?;
    let records = budget.pack(
        shape,
        input.labels(),
        |row, p, work| {
            features::state(
                input.programs[input.partition.retained[p]],
                &input.alphabet,
                input.prefix(row)?,
                work,
            )
        },
        work,
    )?;
    let domain = budget.dictionary(&records, 1., 64., work)?;
    Ok(Some(Compiled {
        domain,
        records,
        budget,
    }))
}
impl Compiled {
    pub fn train(&self, mode: Mode, work: &mut Work) -> Result<Trained> {
        self.budget
            .train(&self.domain, &self.records, policy(mode)?, work)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    Unseen,
    Disagreement,
    Agreement(usize),
}
pub struct Forward<'a, 'b> {
    input: &'a Prepared<'b>,
    domain: &'a Domain,
    head: &'a joint::head::Head,
    work: &'a mut Work,
}
impl<'a, 'b> Forward<'a, 'b> {
    pub fn new(
        input: &'a Prepared<'b>,
        domain: &'a Domain,
        head: &'a joint::head::Head,
        work: &'a mut Work,
    ) -> Result<Self> {
        if head.shape() != domain.shape
            || domain.shape.programs != input.partition.retained.len()
            || domain.shape.states != features::STATES
            || domain.shape.classes != input.alphabet.vocabulary
        {
            return Err("model endpoint differs from complete input family".into());
        }
        Ok(Self {
            input,
            domain,
            head,
            work,
        })
    }
    fn states(&mut self, prefix: &[i64]) -> Result<Vec<usize>> {
        features::validate_prefix(prefix, self.input.alphabet.vocabulary)?;
        self.input
            .partition
            .retained
            .iter()
            .map(|&p| {
                features::state(
                    self.input.programs[p],
                    &self.input.alphabet,
                    prefix,
                    self.work,
                )
            })
            .collect()
    }
    pub fn coverage(&mut self, prefix: &[i64]) -> Result<Coverage> {
        let states = self.states(prefix)?;
        let mut agreed = None;
        let mut unknown = false;
        for (p, s) in states.into_iter().enumerate() {
            if let Some(y) = self.domain.labels[p * features::STATES + s] {
                if agreed.is_some_and(|a| a != y) {
                    return Ok(Coverage::Disagreement);
                }
                agreed = Some(y);
            } else {
                unknown = true;
            }
        }
        Ok(if unknown {
            Coverage::Unseen
        } else {
            Coverage::Agreement(agreed.ok_or("empty model")?)
        })
    }
}
impl Predictor for Forward<'_, '_> {
    fn logits(&mut self, prefix: &[i64]) -> probes::Result<Vec<f64>> {
        let states = self.states(prefix).map_err(|error| error.to_string())?;
        self.head
            .logits(&states, self.work)
            .map_err(|error| error.to_string().into())
    }
}
