//! Outward finite binary64 enclosures, adapted from certified_head at
//! d1b10e2e. This source needs only addition and signed-half products.
#[derive(Clone, Copy, Debug)]
pub(super) struct Interval {
    pub lo: f64,
    pub hi: f64,
}
impl Interval {
    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    pub fn add(self, b: Self) -> Self {
        if b.lo == 0. && b.hi == 0. {
            return self;
        }
        if self.lo == 0. && self.hi == 0. {
            return b;
        }
        Self {
            lo: (self.lo + b.lo).next_down(),
            hi: (self.hi + b.hi).next_up(),
        }
    }
    pub fn scale(self, x: f64) -> Self {
        if x == 0. || (self.lo == 0. && self.hi == 0.) {
            return Self::point(0.);
        }
        let (a, b) = if x > 0. {
            (self.lo, self.hi)
        } else {
            (self.hi, self.lo)
        };
        Self {
            lo: (a * x).next_down(),
            hi: (b * x).next_up(),
        }
    }
    pub fn finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }
}
