//! CM97 Pair and CM98 Rotary: full complex matrices under the product maximum norm.
//! CM99 adds an explicit fixed-frequency control; all content coordinates remain learned.
use crate::{C, Result};
use serde::{Deserialize, Serialize};
pub type State = [C; 2];
pub type Matrix = [State; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    Plain,
    Fixed,
    Rotary,
}
impl Family {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "plain" => Ok(Self::Plain),
            "fixed" => Ok(Self::Fixed),
            "rotary" => Ok(Self::Rotary),
            _ => Err("unknown recurrent family".into()),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Fixed => "fixed",
            Self::Rotary => "rotary",
        }
    }
    pub fn coordinates(self, objects: usize) -> usize {
        4 + 12 * objects + if self == Self::Plain { 0 } else { 2 }
    }
    pub fn trainable_coordinates(self, objects: usize) -> usize {
        4 + 12 * objects + if self == Self::Rotary { 2 } else { 0 }
    }
    pub fn initial_frequency(self) -> Option<[f64; 2]> {
        match self {
            Self::Plain => None,
            Self::Fixed => Some([1., 0.01]),
            Self::Rotary => Some([0.; 2]),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Atom {
    pub frequency: Option<[f64; 2]>,
    pub initial: State,
    pub transition: Vec<Matrix>,
    pub output: Vec<State>,
}

pub fn row(u: State, v: State) -> C {
    u[0] * v[0] + u[1] * v[1]
}
pub fn operator(m: Matrix, v: State) -> State {
    [row(m[0], v), row(m[1], v)]
}

impl Atom {
    pub fn zero(objects: usize, family: Family) -> Self {
        Self {
            frequency: family.initial_frequency(),
            initial: [C::default(); 2],
            transition: vec![[[C::default(); 2]; 2]; objects],
            output: vec![[C::default(); 2]; objects],
        }
    }
    pub fn identity_start(objects: usize, family: Family) -> Self {
        let mut a = Self::zero(objects, family);
        a.initial = [C::new(1., 0.); 2];
        a.transition.fill([
            [C::new(1., 0.), C::default()],
            [C::default(), C::new(1., 0.)],
        ]);
        a
    }
    pub fn bias(objects: usize, family: Family, class: usize, sign: f64) -> Result<Self> {
        if class >= objects || !matches!(sign, -1. | 1.) {
            return Err("invalid signed original-class query".into());
        }
        let mut a = Self::identity_start(objects, family);
        a.initial[1] = C::default();
        a.output[class][0] = C::new(sign, 0.);
        Ok(a)
    }
    pub fn validate(&self, objects: usize, family: Family) -> Result<f64> {
        if objects == 0
            || self.transition.len() != objects
            || self.output.len() != objects
            || self.frequency.is_some() != (family != Family::Plain)
            || (family == Family::Fixed && self.frequency != family.initial_frequency())
            || self
                .frequency
                .is_some_and(|f| f.iter().any(|x| !x.is_finite()))
        {
            return Err("incomplete or mismatched learned recurrent family".into());
        }
        let mut excess = 0f64;
        for z in &self.initial {
            if !z.re.is_finite() || !z.im.is_finite() || z.norm() > 1. + 2e-12 {
                return Err("unbounded initial state".into());
            }
            excess = excess.max(z.norm() - 1.);
        }
        for u in self.transition.iter().flatten().chain(&self.output) {
            let n = u[0].norm() + u[1].norm();
            if u.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) || n > 1. + 2e-12 {
                return Err("unbounded complete matrix/readout row".into());
            }
            excess = excess.max(n - 1.);
        }
        Ok(excess)
    }
    /// Ephemeral evaluation cache derived only from current frequencies and input length.
    /// No cached value is a learned parameter or fitted to an observed label.
    pub(crate) fn phases(&self, length: usize) -> Result<Option<Vec<State>>> {
        self.frequency
            .map(|f| {
                (0..length)
                    .map(|t| {
                        let mut r = [C::default(); 2];
                        for j in 0..2 {
                            let x = t as f64 * f[j];
                            if !x.is_finite() {
                                return Err("nonfinite rotary argument".into());
                            }
                            let (s, c) = x.sin_cos();
                            r[j] = C::new(c, s);
                        }
                        Ok(r)
                    })
                    .collect()
            })
            .transpose()
    }
    pub(crate) fn state_with(&self, prefix: &[i64], phases: Option<&[State]>) -> Result<State> {
        let mut z = self.initial;
        for (t, &o) in prefix.iter().enumerate() {
            let m = *self
                .transition
                .get(usize::try_from(o)?)
                .ok_or("object outside vocabulary")?;
            z = if let Some(phases) = phases {
                let r = phases.get(t).ok_or("incomplete rotary evaluation cache")?;
                let v = operator(m, [r[0].conj() * z[0], r[1].conj() * z[1]]);
                [r[0] * v[0], r[1] * v[1]]
            } else {
                operator(m, z)
            };
        }
        if z.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
            return Err("nonfinite executed recurrent state".into());
        }
        Ok(z)
    }
    pub fn state(&self, prefix: &[i64]) -> Result<State> {
        let phases = self.phases(prefix.len())?;
        self.state_with(prefix, phases.as_deref())
    }
    pub fn complex(&self, prefix: &[i64]) -> Result<Vec<C>> {
        let state = self.state(prefix)?;
        Ok(self.output.iter().map(|&u| row(u, state)).collect())
    }
}
