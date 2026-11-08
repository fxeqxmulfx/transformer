//! Same-update decomposition of QDM; every class retains its own coordinate D.
use crate::{application::State, domain::model::E, geometry, gradient::energy};
use serde_json::{Value, json};

pub const NAMES: [&str; 5] = [
    "readout",
    "commutator",
    "encoder",
    "minibatch",
    "moment_memory",
];
pub fn subtract(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a - b).collect()
}
pub fn multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a * b).collect()
}
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

pub struct Direction {
    pub mean: Vec<f64>,
    pub diagonal: Vec<f64>,
    pub forcing: Vec<f64>,
    pub components: [Vec<f64>; 4],
    pub rounding: Vec<f64>,
    pub eta: f64,
    pub decay: f64,
}
impl Direction {
    pub fn new(
        old: &State,
        new: &State,
        readout: &[f64],
        full: &[f64],
        batch: &[f64],
        step: usize,
    ) -> Self {
        assert!(step > 0);
        let b1 = 1. - 0.9f64.powf(step as f64);
        let b2 = (1. - 0.98f64.powf(step as f64)).sqrt();
        let mean: Vec<_> = new.first[..E].iter().map(|&v| v as f64 / b1).collect();
        let diagonal: Vec<_> = new.second[..E]
            .iter()
            .map(|&v| 1. / ((v as f64).sqrt() / b2 + 1e-8))
            .collect();
        let forcing = multiply(&diagonal, &mean);
        let eta = 0.001 * (step as f64 / 10.).min(1.);
        let decay = 1. - eta;
        let rounding = (0..E)
            .map(|k| new.parameters[k] as f64 - decay * old.parameters[k] as f64 + eta * forcing[k])
            .collect();
        Self {
            components: [
                readout.to_vec(),
                subtract(full, readout),
                subtract(batch, full),
                subtract(&mean, batch),
            ],
            mean,
            diagonal,
            forcing,
            rounding,
            eta,
            decay,
        }
    }
    pub fn summary(&self) -> Value {
        let max_error = (0..self.mean.len())
            .map(|k| (self.mean[k] - self.components.iter().map(|x| x[k]).sum::<f64>()).abs())
            .fold(0., f64::max);
        json!({"gradient_components":["full_train_readout","full_train_encoder","sampling_difference","corrected_moment_minus_batch"],
            "raw_component_energies":self.components.each_ref().map(|x|energy(x)),
            "raw_closure_max_error":max_error,"update_rounding_energy":energy(&self.rounding),
            "update_rounding_max":self.rounding.iter().map(|x|x.abs()).fold(0.,f64::max),
            "real_valued_forcing_energy":energy(&self.forcing),"learning_rate":self.eta,"decay":self.decay,
            "dmax":self.diagonal.iter().copied().fold(0.,f64::max)})
    }
    pub fn project(&self, q: &[f64], width: usize) -> Parts {
        let a = multiply(
            &self.diagonal,
            &geometry::project(&self.components[0], q, width),
        );
        let b = subtract(
            &geometry::project(&multiply(&self.diagonal, &self.components[0]), q, width),
            &a,
        );
        let c = std::array::from_fn::<_, 3, _>(|i| {
            geometry::project(&multiply(&self.diagonal, &self.components[i + 1]), q, width)
        });
        let [encoder, batch, memory] = c;
        Parts {
            terms: [a, b, encoder, batch, memory],
            actual: geometry::project(&self.forcing, q, width),
        }
    }
}

pub struct Parts {
    pub terms: [Vec<f64>; 5],
    pub actual: Vec<f64>,
}
#[derive(Default)]
pub struct Energy {
    n: usize,
    actual: f64,
    terms: [f64; 5],
    cross: [[f64; 5]; 5],
    closure: f64,
    closure_relative: f64,
    energy_error: f64,
}
impl Energy {
    pub fn add(&mut self, p: &Parts) {
        self.n += 1;
        let actual = energy(&p.actual);
        let es = p.terms.each_ref().map(|x| energy(x));
        self.actual += actual;
        for (i, e) in es.iter().enumerate() {
            self.terms[i] += e;
            for j in i + 1..5 {
                self.cross[i][j] += 2. * dot(&p.terms[i], &p.terms[j]);
            }
        }
        let error = (0..p.actual.len())
            .map(|k| (p.actual[k] - p.terms.iter().map(|x| x[k]).sum::<f64>()).abs())
            .fold(0., f64::max);
        let scale = p
            .terms
            .iter()
            .flatten()
            .chain(&p.actual)
            .map(|x| x.abs())
            .fold(0., f64::max);
        self.closure = self.closure.max(error);
        self.closure_relative = self.closure_relative.max(error / (1. + scale));
        let mut sum = es.iter().sum::<f64>();
        for i in 0..5 {
            for j in i + 1..5 {
                sum += 2. * dot(&p.terms[i], &p.terms[j]);
            }
        }
        self.energy_error = self
            .energy_error
            .max((actual - sum).abs() / (1. + actual + es.iter().sum::<f64>()));
    }
    pub fn valid(&self) -> bool {
        self.closure_relative <= 1e-8 && self.energy_error <= 1e-8
    }
    pub fn json(&self) -> Value {
        json!({"bases":self.n,"actual_energy":self.actual,"term_names":NAMES,"term_energies":self.terms,
            "twice_pairwise_inner_products_upper_triangle":self.cross,
            "closure_max_error":self.closure,"closure_scaled_error":self.closure_relative,
            "energy_scaled_error":self.energy_error,"passed":self.valid()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn noncommuting_class_scales_preserve_all_cross_terms() {
        let d = Direction {
            mean: vec![2., -1., 0.3, 4.],
            diagonal: vec![1., 3., 2., 0.2],
            forcing: vec![2., -3., 0.6, 0.8],
            components: [
                vec![1., 2., -2., 3.],
                vec![0.5, -1., 1., 2.],
                vec![-1., 0.2, 0.3, -2.],
                vec![1.5, -2.2, 1., 1.],
            ],
            rounding: vec![0.; 4],
            eta: 0.1,
            decay: 0.9,
        };
        let parts = d.project(&[0.8, -0.4, -0.4, 0.2], 2);
        let mut e = Energy::default();
        e.add(&parts);
        assert!(e.valid());
        assert!(energy(&parts.terms[1]) > 0.);
        assert!((e.actual - e.terms.iter().sum::<f64>()).abs() > 0.1);
        let contrast = geometry::contrasts(&parts.actual, 0, 2);
        let independent = geometry::project(&[0.6 - 2., 0.8 + 3.], &[0.8, -0.4, -0.4, 0.2], 2);
        assert!(
            subtract(&contrast, &independent)
                .iter()
                .all(|x| x.abs() < 1e-14)
        );
    }
    #[test]
    fn fp32_parameter_rounding_is_explicit() {
        let old = State {
            parameters: vec![1.1; E],
            first: vec![0.; E],
            second: vec![0.; E],
        };
        let new = State {
            parameters: vec![1.0989; E],
            first: vec![0.; E],
            second: vec![0.; E],
        };
        let d = Direction::new(&old, &new, &vec![0.; E], &vec![0.; E], &vec![0.; E], 11);
        assert!(d.rounding[0] != 0.);
        assert_eq!(
            new.parameters[0] as f64,
            d.decay * old.parameters[0] as f64 + d.rounding[0]
        );
    }
}
