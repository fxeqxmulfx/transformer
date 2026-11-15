//! Outward enclosures for finite IEEE754 binary64 operations without fast-math.
//! next_down/next_up include rounding and subnormal underflow. Nonfinite
//! endpoints are never accepted as a separation certificate.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Interval {
    pub lo: f64,
    pub hi: f64,
}
impl Interval {
    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    pub fn finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }
    pub fn contains(self, x: f64) -> bool {
        self.finite() && x.is_finite() && self.lo <= x && x <= self.hi
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
    pub fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    pub fn sub(self, b: Self) -> Self {
        self.add(b.neg())
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
    pub fn divide_positive(self, n: Self) -> Self {
        if !n.finite() || n.lo <= 0. {
            return Self {
                lo: f64::NAN,
                hi: f64::NAN,
            };
        }
        let corners = [
            self.lo / n.lo,
            self.lo / n.hi,
            self.hi / n.lo,
            self.hi / n.hi,
        ];
        Self {
            lo: corners
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            hi: corners
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
    }
}
