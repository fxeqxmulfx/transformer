//! Outer prediction diagnostics for the class translations in CM121.
use serde_json::{Value, json};

#[derive(Default)]
struct PairCount {
    consistent: u64,
    total: u64,
}

impl PairCount {
    fn count<const P: usize>(&mut self, codes: impl Iterator<Item = usize>) {
        let mut bins = [0_u64; P];
        for (seen, code) in codes.enumerate() {
            self.consistent += bins[code];
            self.total += seen as u64;
            bins[code] += 1;
        }
    }

    fn json(&self) -> Value {
        json!({"consistent": self.consistent, "total": self.total})
    }
}

/// Only observed cells belong to this split. A missing corner supplies no vote.
pub struct Residuals<const P: usize> {
    cells: [[Option<usize>; P]; P],
}

impl<const P: usize> Residuals<P> {
    pub fn new() -> Self {
        Self {
            cells: [[None; P]; P],
        }
    }

    pub fn record(
        &mut self,
        a: u32,
        b: u32,
        prediction: u32,
        target: u32,
    ) -> Result<(), &'static str> {
        let [a, b, prediction, target] = [a, b, prediction, target].map(|x| x as usize);
        if [a, b, prediction, target].into_iter().any(|x| x >= P) {
            return Err("class or coordinate outside the NV11 shift table");
        }
        if self.cells[a][b].is_some() {
            return Err("duplicate coordinate in NV11 shift diagnostics");
        }
        self.cells[a][b] = Some((prediction + P - target) % P);
        Ok(())
    }

    pub fn report(&self) -> Value {
        let mut rows = PairCount::default();
        let mut columns = PairCount::default();
        let mut rectangles = PairCount::default();
        for (i, row) in self.cells.iter().enumerate() {
            rows.count::<P>(row.iter().flatten().copied());
            columns.count::<P>(self.cells.iter().filter_map(|r| r[i]));
            for other in &self.cells[..i] {
                // For these rows, each pair of equal column differences
                // contributes exactly one zero alternating rectangle.
                rectangles.count::<P>(
                    row.iter()
                        .zip(other)
                        .filter_map(|(x, y)| Some(((*x)? + P - (*y)?) % P)),
                );
            }
        }
        json!({"row_pairs": rows.json(), "column_pairs": columns.json(),
               "rectangles": rectangles.json()})
    }
}
