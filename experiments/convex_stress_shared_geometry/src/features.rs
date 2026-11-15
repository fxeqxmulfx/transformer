//! Generic ordinal features; the same complete Fourier row is used at every
//! observed or unobserved query. Flatness bounds the code, not the empirical rows.
use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bank {
    Quadratic,
    Hinges16,
    Hinges32,
}
pub const BANKS: [Bank; 3] = [Bank::Quadratic, Bank::Hinges16, Bank::Hinges32];
impl Bank {
    pub fn grid(self) -> usize {
        match self {
            Self::Quadratic => 0,
            Self::Hinges16 => 16,
            Self::Hinges32 => 32,
        }
    }
    pub fn dimension(self) -> usize {
        6 + 2 * self.grid().saturating_sub(1)
    }
    pub fn states(self) -> usize {
        self.dimension().next_power_of_two()
    }
}
pub fn raw(bank: Bank, n: usize, cell: usize) -> Vec<f64> {
    let [u, v] = convex_stress_shared_head::features::point(n, cell);
    let mut row = vec![1., u, v, u * u, u * v, v * v];
    for x in [u, v] {
        for k in 1..bank.grid() {
            row.push((x - (2. * k as f64 / bank.grid() as f64 - 1.)).max(0.));
        }
    }
    row
}
pub fn phase(m: usize, i: usize, f: usize) -> C {
    C::from_polar(
        1. / (m as f64).sqrt(),
        std::f64::consts::TAU * ((i * f) % m) as f64 / m as f64,
    )
}
#[derive(Clone, Debug)]
pub struct Row {
    pub block: usize,
    pub label: usize,
    pub scale: f64,
    pub phi: Vec<C>,
}
pub struct Encoder {
    pub bank: Bank,
    phases: Vec<C>,
}
impl Encoder {
    pub fn new(bank: Bank) -> Self {
        let m = bank.states();
        Self {
            bank,
            phases: (0..m)
                .flat_map(|i| (0..m).map(move |f| phase(m, i, f)))
                .collect(),
        }
    }
    pub fn row(&self, input: &Input, cell: usize, label: usize) -> Row {
        let raw = raw(self.bank, input.n, cell);
        let scale = raw.iter().map(|x| x.abs()).sum::<f64>().max(1.);
        let m = self.bank.states();
        let phi = (0..m)
            .map(|f| {
                raw.iter()
                    .enumerate()
                    .map(|(i, &x)| self.phases[i * m + f] * (x / scale))
                    .sum()
            })
            .collect();
        Row {
            block: input.route[cell],
            label,
            scale,
            phi,
        }
    }
    pub fn records(&self, input: &Input) -> Vec<Row> {
        input
            .records
            .iter()
            .map(|r| self.row(input, r.cell, r.label))
            .collect()
    }
}
pub fn conjugate(bank: Bank, spatial: &[f64]) -> Vec<f64> {
    let m = bank.states();
    let mut theta = vec![0.; 2 * m];
    for f in 0..m {
        for (i, &x) in spatial.iter().enumerate() {
            let p = phase(m, i, f);
            theta[2 * f] += p.re * x;
            theta[2 * f + 1] -= p.im * x;
        }
    }
    theta
}
