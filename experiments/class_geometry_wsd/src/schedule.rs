//! GPT main-model schedule at 6bb9d03a, wsd_suite_native/src/schedule.rs.
//! Admit the fixed 10000-update budget; task #201 changes only the frozen peak.
use crate::Result;
#[cfg(all(feature = "lr003", feature = "lr01"))]
compile_error!("Choose only one frozen learning-rate arm");
#[cfg(not(any(feature = "lr003", feature = "lr01")))]
pub const PEAK: f64 = 0.001;
#[cfg(feature = "lr003")]
pub const PEAK: f64 = 0.003;
#[cfg(all(feature = "lr01", not(feature = "lr003")))]
pub const PEAK: f64 = 0.01;
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub budget: usize,
    pub warmup: usize,
    pub cooldown_start: usize,
}
impl Schedule {
    pub fn new(budget: usize) -> Result<Self> {
        if ![10_000, 20_000, 30_000, 100_000].contains(&budget) {
            return Err("undeclared task update budget".into());
        }
        Ok(Self {
            budget,
            warmup: budget * 2 / 100,
            cooldown_start: budget * 80 / 100,
        })
    }
    pub fn rate(self, completed: usize) -> f32 {
        let t = completed.min(self.budget - 1);
        if t < self.warmup {
            return (PEAK * t as f64 / self.warmup as f64) as f32;
        }
        let decay = if t < self.cooldown_start {
            0.
        } else {
            let u =
                (t - self.cooldown_start) as f64 / (self.budget - 1 - self.cooldown_start) as f64;
            0.5 * (1. - (std::f64::consts::PI * u).cos())
        };
        (PEAK + (1e-8 - PEAK) * decay) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn main_schedule_keeps_zero_based_boundaries_and_fixed_floor() {
        let s = Schedule::new(10000).unwrap();
        assert_eq!(s.rate(0), 0.);
        assert_eq!(s.rate(1), (PEAK / 200.) as f32);
        assert!(s.rate(199) < s.rate(200));
        assert_eq!(s.rate(200), PEAK as f32);
        assert_eq!(s.rate(7999), PEAK as f32);
        assert_eq!(s.rate(8000), PEAK as f32);
        assert!(s.rate(8001) < PEAK as f32);
        assert_eq!(s.rate(9999), 1e-8);
        assert_eq!(s.rate(10000), 1e-8);
        assert!(Schedule::new(9999).is_err());
    }
}
