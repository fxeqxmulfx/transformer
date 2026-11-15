//! CM63 ExactScoring at18308305: evaluate the actual saved CM57 tensor.
//! Outward IEEE754 enclosures certify separated winners; ambiguous products
//! are recomputed as exact rationals by independent real/complex contractions.
//! Lean proves the real identities, not a refinement of this Rust program.
pub use convex_stress_streaming_counts as previous;
use joint::Features;
pub use previous::{Result, joint};
use serde::{Deserialize, Serialize};
mod bounded;
mod exact;
mod interval;
pub mod model;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Forecast {
    pub programs: usize,
    pub classes: usize,
    pub complete_saved_coefficients: usize,
    pub maximum_coefficient_multiplications_per_query: usize,
    pub exact_integer_bit_cap: u64,
    pub buffer_bytes: u64,
}

pub fn forecast(programs: usize, classes: usize) -> Result<Forecast> {
    if !(1..=64).contains(&programs) || !(2..=1024).contains(&classes) {
        return Err("certified complete tensor dimensions refused".into());
    }
    let products = programs * 16 * classes;
    // Two exact coefficient contractions, temporaries, intervals, serialization
    // and allocator slack. Finite bounded dyadics have <=2200 product bits;
    // rational normalization/conversion is additionally checked at8192 bits.
    let bytes = 256 * 1024u128.pow(2)
        + products as u128 * 8192
        + classes as u128 * 65536
        + programs as u128 * 32768;
    if bytes > joint::MAX_BYTES {
        return Err("certified complete tensor memory forecast refused".into());
    }
    Ok(Forecast {
        programs,
        classes,
        complete_saved_coefficients: products,
        // Floating real/complex each once; exact real once; exact complex
        // distributes each coefficient over four source coordinates.
        maximum_coefficient_multiplications_per_query: 7 * products,
        exact_integer_bit_cap: 8192,
        buffer_bytes: bytes as u64,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    SeparatedEnclosures,
    ExactRationals,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    /// A common class shift puts the first exact maximum at zero. Ordinary CE
    /// is invariant under this shift (CM63 gauge_crossEntropy).
    pub logits: Vec<f64>,
    pub predicted: usize,
    pub maximizers: Vec<usize>,
    pub method: Method,
    pub independent_residual: f64,
    /// Maximum outward enclosure width for the returned, shifted logits.
    pub logit_enclosure_width: f64,
    pub target_realizability_certified: bool,
}

pub struct Evaluator<'a> {
    pub(crate) coefficients: &'a [f64],
    pub forecast: Forecast,
}

impl<'a> Evaluator<'a> {
    pub fn from_head(head: &'a joint::head::Head) -> Result<Self> {
        forecast(head.theta.len(), head.coordinates.len())?;
        head.validate()?;
        Self::from_coefficients(&head.coefficients, head.theta.len(), head.coordinates.len())
    }

    /// Arbitrary complete tensors are admitted for component controls. The
    /// model adapter uses from_head, which also verifies saved affine parameters.
    pub fn from_coefficients(coefficients: &'a [f64], p: usize, v: usize) -> Result<Self> {
        let forecast = forecast(p, v)?;
        if coefficients.len() != forecast.complete_saved_coefficients
            || coefficients
                .iter()
                .any(|x| !x.is_finite() || x.abs() > 1e14)
        {
            return Err("invalid finite complete saved coefficient tensor".into());
        }
        Ok(Self {
            coefficients,
            forecast,
        })
    }

    pub fn evaluate(&self, rows: &[Features]) -> Result<Scores> {
        joint::data::features(rows, self.forecast.programs)?;
        if let Some(scores) = bounded::evaluate(self, rows) {
            return Ok(scores);
        }
        exact::evaluate(self, rows)
    }
}

fn row(h: &Features, block: usize) -> [f64; 4] {
    if block == 0 {
        [h[0], h[1], h[2], h[3]]
    } else {
        [h[4], 1., 0., 0.]
    }
}

/// Exact realification of the four-point complex code; no trigonometric calls.
fn source(k: usize, s: usize) -> f64 {
    let phase = (s * (k / 2)) % 4;
    let units = if k.is_multiple_of(2) {
        [1., 0., -1., 0.][phase]
    } else {
        [0., -1., 0., 1.][phase]
    };
    units * 0.5
}
