//! Two signed addresses of the same dense relation tuple share one
//! parameter budget. AgreementGain gives the diagonal and collision bounds.
use super::paths::{BINS, Witness};

pub(super) struct Agreement {
    floor: f64,
    denominator: f64,
    maximum: f32,
    ties: usize,
    power: i32,
}
impl Agreement {
    pub(super) fn new(raw: &[f32], power: u8) -> Self {
        assert_eq!(raw.len(), BINS);
        let maximum = raw.iter().map(|w| w.abs()).fold(0., f32::max);
        let ties = raw.iter().filter(|w| w.abs() == maximum).count();
        let floor = 1. / BINS as f64;
        Self {
            floor,
            denominator: f64::from(maximum).powi(2) + floor,
            maximum,
            ties,
            power: power.into(),
        }
    }
    fn values(&self, raw: &[f32], positions: [(usize, f64); 2]) -> (f64, f64, f64, f64) {
        let [(i, s), (j, t)] = positions;
        let a = s * f64::from(raw[i]);
        let b = t * f64::from(raw[j]);
        let mean = (a + b) / 2.;
        let ratio = (a * b + self.floor) / self.denominator;
        (a, b, mean, ratio)
    }
    pub(super) fn weight(&self, raw: &[f32], path: Witness) -> f32 {
        let (_, _, mean, ratio) = self.values(raw, path.addresses());
        (mean * ratio.powi(self.power)) as f32
    }
    pub(super) fn add_gradient(
        &self,
        raw: &[f32],
        path: Witness,
        up: f32,
        dw: &mut [f32],
        radial: &mut f64,
    ) {
        let addresses = path.addresses();
        let (a, b, mean, ratio) = self.values(raw, addresses);
        let power = ratio.powi(self.power);
        let slope = if self.power == 0 {
            0.
        } else {
            f64::from(self.power) * mean * ratio.powi(self.power - 1) / self.denominator
        };
        let [(i, s), (j, t)] = addresses;
        let g = f64::from(up);
        dw[i] += (g * s * (power / 2. + slope * b)) as f32;
        dw[j] += (g * t * (power / 2. + slope * a)) as f32;
        *radial += g * mean * power;
    }
    pub(super) fn maximum_gradient(&self, raw: &[f32], radial: f64, dw: &mut [f32]) {
        let factor = -2. * f64::from(self.power) * radial / (self.denominator * self.ties as f64);
        for (&w, g) in raw.iter().zip(dw) {
            if w.abs() == self.maximum {
                *g += (factor * f64::from(w)) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_coordinates_and_maximum_share_the_pullback() {
        let path = Witness::at_code(0x9017_cc24);
        let [(i, s), (j, t)] = path.addresses();
        let elsewhere = 7;
        for (a, b, m) in [
            (0., 0., 0.),
            (0.75, -0.65, 1.),
            (-0.8, -0.8, 0.8),
            (1., 1., 0.7),
        ] {
            let mut w = vec![0f32; BINS];
            w[i] = (a * s) as f32;
            w[j] = (b * t) as f32;
            w[elsewhere] = m;
            let agreement = Agreement::new(&w, 4);
            let mut dw = vec![0.; BINS];
            let mut radial = 0.;
            agreement.add_gradient(&w, path, 1., &mut dw, &mut radial);
            agreement.maximum_gradient(&w, radial, &mut dw);
            if a == 0. && b == 0. {
                assert_eq!((dw[i], dw[j]), (s as f32 / 2., t as f32 / 2.));
                continue;
            }
            for idx in [i, j, elsewhere] {
                let value = |delta: f32| {
                    let mut x = w.clone();
                    x[idx] += delta;
                    f64::from(Agreement::new(&x, 4).weight(&x, path))
                };
                let step = 0.0005f32;
                let slope = |h: f32| (value(h) - value(-h)) / (2. * f64::from(h));
                // Two coincident maxima admit the symmetric one-sided derivative.
                // The convention for three or more is the equal subgradient.
                let tied = w.iter().filter(|x| x.abs() == agreement.maximum).count();
                if tied <= 2 || w[idx].abs() != agreement.maximum {
                    let expected = 2. * slope(step) - slope(2. * step);
                    assert!(
                        (f64::from(dw[idx]) - expected).abs() < 0.003,
                        "at {idx}, numeric {expected}, adjoint {}",
                        dw[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn disabling_concentration_keeps_the_mean_and_its_gradient_at_zero_product_ratio() {
        let path = Witness::at_code(0x9017_cc24);
        let [(i, s), (j, t)] = path.addresses();
        let mut w = vec![0f32; BINS];
        w[i] = s as f32;
        w[j] = (-t / BINS as f64) as f32;
        let rule = Agreement::new(&w, 0);
        assert_eq!(rule.weight(&w, path), (1. - 1. / BINS as f32) / 2.);
        let mut dw = vec![0.; BINS];
        let mut radial = 0.;
        rule.add_gradient(&w, path, 1., &mut dw, &mut radial);
        rule.maximum_gradient(&w, radial, &mut dw);
        assert_eq!((dw[i], dw[j]), (s as f32 / 2., t as f32 / 2.));
    }
}
