//! The outer temporal scorer from DiffGPT's RecurrenceTask.accumulate.
//! The center-token decode and reciprocal scale come entirely from the past.
#[derive(Clone, Copy, Debug)]
pub struct Decoder {
    pub center: i64,
    pub base: f64,
    pub step: f64,
    pub future: i64,
}

impl Decoder {
    pub fn accepts(self, token: i64) -> bool {
        (self.base + (token - self.center) as f64 * self.step).round_ties_even()
            == self.future as f64
    }
}
