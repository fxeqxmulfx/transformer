//! GPT main-model schedule at 6bb9d03a, wsd_suite_native/src/schedule.rs.
//! Admit the fixed 10000-update budget; task #201 changes only the frozen peak.
//! Task #202 deliberately adds an early cosine reduction and nonzero hold.
use crate::Result;
#[cfg(any(
    all(feature = "lr003", feature = "lr01"),
    all(feature = "settle", any(feature = "lr003", feature = "lr01"))
))]
compile_error!("Choose only one frozen learning-rate arm");
#[cfg(not(any(feature = "lr003", feature = "lr01", feature = "settle")))]
pub const PEAK: f64 = 0.001;
#[cfg(feature = "lr003")]
pub const PEAK: f64 = 0.003;
#[cfg(any(feature = "lr01", feature = "settle"))]
pub const PEAK: f64 = 0.01;
#[cfg(feature = "settle")]
pub const NAME: &str = "early_settle_1000_4000_hold_1e-4";
#[cfg(not(feature = "settle"))]
pub const NAME: &str = "main_wsd";
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub budget: usize,
    pub warmup: usize,
    pub cooldown_start: usize,
}
impl Schedule {
    pub fn new(budget: usize) -> Result<Self> {
        #[cfg(feature = "settle")]
        if budget != 10_000 {
            return Err("early settling has only the frozen 10000-update budget".into());
        }
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
        #[cfg(feature = "settle")]
        {
            let (start, end, high, low) = if t <= 1000 {
                return PEAK as f32;
            } else if t < 4000 {
                (1000, 4000, PEAK, 1e-4)
            } else if t <= self.cooldown_start {
                return 1e-4;
            } else {
                (self.cooldown_start, self.budget - 1, 1e-4, 1e-8)
            };
            let u = (t - start) as f64 / (end - start) as f64;
            (high + (low - high) * 0.5 * (1. - (std::f64::consts::PI * u).cos())) as f32
        }
        #[cfg(not(feature = "settle"))]
        {
            let decay = if t < self.cooldown_start {
                0.
            } else {
                let u = (t - self.cooldown_start) as f64
                    / (self.budget - 1 - self.cooldown_start) as f64;
                0.5 * (1. - (std::f64::consts::PI * u).cos())
            };
            (PEAK + (1e-8 - PEAK) * decay) as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg(not(feature = "settle"))]
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
    #[test]
    #[cfg(feature = "settle")]
    fn early_settling_has_the_frozen_endpoints_and_nonzero_hold() {
        let s = Schedule::new(10000).unwrap();
        assert_eq!(s.rate(0), 0.);
        assert_eq!(s.rate(1), 0.00005);
        assert!(s.rate(199) < s.rate(200));
        assert_eq!(s.rate(200), 0.01);
        assert_eq!(s.rate(1000), 0.01);
        assert!((s.rate(2500) - 0.00505).abs() < 1e-9);
        assert_eq!(s.rate(4000), 0.0001);
        assert!((4000..=8000).all(|t| s.rate(t) == 0.0001));
        assert!((1000..4000).all(|t| s.rate(t + 1) <= s.rate(t)));
        assert!((8000..9999).all(|t| s.rate(t + 1) <= s.rate(t)));
        assert_eq!(s.rate(9999), 1e-8);
        assert_eq!(s.rate(10000), 1e-8);
        assert!(Schedule::new(20000).is_err());
    }
}
