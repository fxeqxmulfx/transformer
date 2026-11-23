//! Magnitude contrast on the shared, initially zero coefficient table.
//! The small positive floor is the mean square coordinate of a unit vector.
pub(crate) fn cubic_weight(value: f32, floor: f64, denominator: f64) -> f32 {
    let x = f64::from(value);
    (x * ((x * x + floor) / denominator)) as f32
}

pub(super) struct NormalizedGain {
    pub effective: Vec<f32>,
    floor: f64,
    denominator: f64,
    maximum: f32,
    ties: usize,
}

impl NormalizedGain {
    pub fn new(raw: &[f32]) -> Self {
        let maximum = raw.iter().copied().map(f32::abs).fold(0., f32::max);
        let ties = raw.iter().filter(|x| x.abs() == maximum).count();
        let floor = 1.0 / raw.len().max(1) as f64;
        let denominator = f64::from(maximum).powi(2) + floor;
        let effective = raw
            .iter()
            .map(|&x| cubic_weight(x, floor, denominator))
            .collect();
        Self {
            effective,
            floor,
            denominator,
            maximum,
            ties,
        }
    }

    pub fn backward(&self, raw: &[f32], incoming: &[f32], out: &mut [f32]) {
        assert_eq!(raw.len(), self.effective.len());
        assert_eq!(out.len(), raw.len());
        assert_eq!(incoming.len(), raw.len());
        let weighted: f64 = incoming
            .iter()
            .zip(&self.effective)
            .map(|(&g, &v)| f64::from(g) * f64::from(v))
            .sum();
        let shared = -2.0 * weighted / (self.denominator * self.ties.max(1) as f64);
        for ((dst, &x), &g) in out.iter_mut().zip(raw).zip(incoming) {
            let v = f64::from(x);
            let direct = f64::from(g) * ((3.0 * v * v + self.floor) / self.denominator);
            let correction = if x.abs() == self.maximum {
                shared * v
            } else {
                0.0
            };
            *dst += (direct + correction) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_symmetric_ties_have_the_specified_pullback() {
        let up = [1.0, -2.0, 0.3, 0.7];
        let mut dx = [0.0; 4];
        let zero = [0.0; 4];
        let g = NormalizedGain::new(&zero);
        assert_eq!(g.effective, zero);
        g.backward(&zero, &up, &mut dx);
        assert_eq!(dx, up);

        let w = [1.0, -1.0, 0.5, -0.2];
        let g = NormalizedGain::new(&w);
        assert_eq!(g.effective[..2], w[..2]);
        assert!(g.effective[2] > 0. && g.effective[2] < w[2]);
        assert!(g.effective[3] < 0. && g.effective[3] > w[3]);
        dx.fill(0.);
        g.backward(&w, &up, &mut dx);
        let step = 0.0005;
        for i in 0..4 {
            let eval = |delta| {
                let mut raw = w;
                raw[i] += delta;
                NormalizedGain::new(&raw)
                    .effective
                    .iter()
                    .zip(&up)
                    .map(|(&x, &g)| f64::from(x) * f64::from(g))
                    .sum::<f64>()
            };
            let numerical = (eval(step) - eval(-step)) / (2.0 * f64::from(step));
            assert!((numerical - f64::from(dx[i])).abs() < 0.002);
        }
    }
}
