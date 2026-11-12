//! Enclosures of the logits of the decoder in exact reals: for each logit two doubles that the
//! real number lies between, whatever the rounding of the computation was.
//!
//! The decoder is that of `src/Transformer/Grokking/Protocol/Model.lean`, the function of real
//! parameters that the trained floats approximate: `position`, `layerNorm`, `score`, `weight`,
//! `attention`, `attended`, `preactivation`, `block`, `decodeTo` and `logits` there, with the
//! activation `relu`. A checkpoint's parameters are `f32` numbers and so are reals exactly. The
//! model in doubles of `reference.rs` is not this function: it rounds every operation, it takes
//! the sinusoids of the positions rounded to `f32` as the source does, and its `1e-5` is a
//! double. Here the sinusoids are those of the reals and the floor of the variance is `1/10^5`.
//!
//! The pass is written once, over an [`Enclosure`]: a set of reals with operations that return
//! a set holding the result for every choice of reals in the arguments. By induction over the
//! operations, every number of the forward pass lies in its set. Two arithmetics compute it.
//!
//! An [`Iv`] holds a lower and an upper end. Every operation computes its ends in the doubles'
//! arithmetic and then moves the lower one down and the upper one up to the next double. A sum,
//! a difference, a product, a quotient and a square root of doubles are rounded to a nearest
//! double by IEEE 754, so the real result is at most half a gap from the computed one and the
//! moved ends enclose it; this holds when the result is subnormal and when it overflows. Its
//! width is the rounding of a double times what the layers make of an error, which is `1e-4` at
//! the checkpoints trained with weight decay and more than 1 at those trained without.
//!
//! A [`Wide`] holds a double and an interval: the reals `head + t` for `t` in `tail`. The sum of
//! two doubles is a double and an error that is a double too, and so is their product; both are
//! computed exactly (`two_sum`, `product`). The error goes to the tail, so the rounding of a
//! `Wide` is that of a number `2^-53` times smaller: it holds a hundred bits and costs three
//! times an `Iv`.
//!
//! No library function that is not correctly rounded is used: the exponential, the sine and the
//! cosine are sums of their series with the remainder added, and `10000^(i/d)` is a power of a
//! repeated square root of 10. An operation panics rather than return what is not a set: an end
//! that is not a number, a lower end above the upper one, a product of 0 and an unbounded end.
//!
//! This is an argument about this file and the processor's arithmetic: sums, differences,
//! products, quotients, square roots and fused products rounded to nearest as IEEE 754 says,
//! which Rust promises for `f64`. A pass begins by checking, in its thread, that sums round to
//! nearest with ties to even and that subnormal numbers are not flushed to zero
//! (`arithmetic_holds`). Lean does not check any of it.

use std::ops::{Add, Div, Mul, Sub};

use crate::layout::Dims;

/// A product of two doubles below this is not split into a double and its error, which would be
/// rounded where it is subnormal.
const TINY: f64 = 1e-250;

/// Whether this thread's arithmetic is the one that the enclosures rest on: a sum is rounded to
/// a nearest double, to the even one of two, and a subnormal number is neither flushed to zero
/// as a result nor read as zero as an operand. Of the four rounding modes, upward fails the
/// first comparison, downward and toward zero the second, and the last two tell those two
/// apart from the mode to nearest once more.
fn arithmetic_holds() -> bool {
    use std::hint::black_box;
    let (one, half) = (black_box(1.0f64), black_box(f64::EPSILON / 2.0));
    let nearest = one + half == 1.0
        && one + 1.5 * half == 1.0 + f64::EPSILON
        && -one - half == -1.0
        && (one + f64::EPSILON) + half == 1.0 + 2.0 * f64::EPSILON;
    let subnormal = black_box(f64::MIN_POSITIVE) * black_box(0.5);
    nearest && subnormal > 0.0 && subnormal * black_box(2.0) == f64::MIN_POSITIVE
}

/// A set of reals with the operations of the pass: each returns a set that holds the result for
/// every choice of reals in its arguments.
pub trait Enclosure: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> {
    /// The real that the double `value` is.
    fn point(value: f64) -> Self;
    /// The reals of an interval.
    fn from_iv(interval: Iv) -> Self;
    /// An interval of doubles that holds the set.
    fn iv(self) -> Iv;
    /// The products with the real that the double `w` is.
    fn scale(self, w: f64) -> Self;
    /// The squares.
    fn sqr(self) -> Self;
    /// The square roots; the set has no negative real.
    fn sqrt(self) -> Self;
    /// `max 0`.
    fn relu(self) -> Self;
    /// The reals within `radius` of the set.
    fn widened(self, radius: f64) -> Self;
    /// The exponentials.
    fn exp(self) -> Self;
    /// `y = x W` for `x` of `n` sets and `W`, `n` rows of `m` doubles: `y_o = sum_i x_i W_io`.
    fn apply(x: &[Self], w: &[f64], m: usize) -> Vec<Self>;
}

/// The reals from `lo` to `hi`. An end that is infinite bounds nothing on its side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Iv {
    /// The lower end.
    pub lo: f64,
    /// The upper end.
    pub hi: f64,
}

impl Iv {
    /// The interval of two ends: numbers, the lower at most the upper, the lower not `+inf` and
    /// the upper not `-inf`.
    fn new(lo: f64, hi: f64) -> Iv {
        assert!(lo <= hi && lo < f64::INFINITY && hi > f64::NEG_INFINITY, "not an interval: {lo} to {hi}");
        Iv { lo, hi }
    }

    /// The ends as computed, moved outward to the next doubles.
    fn outward(lo: f64, hi: f64) -> Iv {
        Iv::new(lo.next_down(), hi.next_up())
    }

    /// The interval of the least and the largest of four computed numbers, moved outward.
    fn hull(c: [f64; 4]) -> Iv {
        assert!(c.iter().all(|v| !v.is_nan()), "a product or a quotient of ends is not a number: {c:?}");
        Iv::outward(c[0].min(c[1]).min(c[2]).min(c[3]), c[0].max(c[1]).max(c[2]).max(c[3]))
    }

    /// The largest absolute value in the interval.
    pub fn mag(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }
}

impl Add for Iv {
    type Output = Iv;
    fn add(self, other: Iv) -> Iv {
        Iv::outward(self.lo + other.lo, self.hi + other.hi)
    }
}

impl Sub for Iv {
    type Output = Iv;
    fn sub(self, other: Iv) -> Iv {
        Iv::outward(self.lo - other.hi, self.hi - other.lo)
    }
}

impl Mul for Iv {
    type Output = Iv;
    fn mul(self, other: Iv) -> Iv {
        Iv::hull([self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi])
    }
}

impl Div for Iv {
    type Output = Iv;
    /// The quotients by an interval of positive reals.
    fn div(self, other: Iv) -> Iv {
        assert!(other.lo > 0.0, "a quotient by an interval that is not positive");
        Iv::hull([self.lo / other.lo, self.lo / other.hi, self.hi / other.lo, self.hi / other.hi])
    }
}

impl Enclosure for Iv {
    fn point(value: f64) -> Iv {
        assert!(value.is_finite(), "a point that is not a real: {value}");
        Iv { lo: value, hi: value }
    }

    fn from_iv(interval: Iv) -> Iv {
        interval
    }

    fn iv(self) -> Iv {
        self
    }

    fn scale(self, w: f64) -> Iv {
        assert!(w.is_finite(), "a factor that is not a real: {w}");
        if w == 0.0 || (self.lo == 0.0 && self.hi == 0.0) {
            Iv::point(0.0)
        } else if w > 0.0 {
            Iv::outward(w * self.lo, w * self.hi)
        } else {
            Iv::outward(w * self.hi, w * self.lo)
        }
    }

    fn sqr(self) -> Iv {
        let (a, b) = (self.lo * self.lo, self.hi * self.hi);
        if self.lo >= 0.0 {
            Iv::outward(a, b)
        } else if self.hi <= 0.0 {
            Iv::outward(b, a)
        } else {
            Iv::new(0.0, a.max(b).next_up())
        }
    }

    fn sqrt(self) -> Iv {
        assert!(self.lo >= 0.0, "the square root of an interval with negative reals");
        Iv::new(self.lo.sqrt().next_down().max(0.0), self.hi.sqrt().next_up())
    }

    /// `max 0` is exact at the ends.
    fn relu(self) -> Iv {
        Iv::new(self.lo.max(0.0), self.hi.max(0.0))
    }

    fn widened(self, radius: f64) -> Iv {
        assert!(radius >= 0.0, "a radius that is not a nonnegative number: {radius}");
        Iv::outward(self.lo - radius, self.hi + radius)
    }

    /// The exponential increases, so the ends are those of the exponentials of the ends. Beyond
    /// `±700` an end is a bound on one side: `e^700 > 10^300`.
    fn exp(self) -> Iv {
        let end = |x: f64| {
            if x > 700.0 {
                Iv::new(1e300, f64::INFINITY)
            } else if x < -700.0 {
                Iv::new(0.0, 1e-300)
            } else {
                exp_point::<Iv>(x)
            }
        };
        Iv::new(end(self.lo).lo.max(0.0), end(self.hi).hi)
    }

    fn apply(x: &[Iv], w: &[f64], m: usize) -> Vec<Iv> {
        assert_eq!(w.len(), x.len() * m, "a matrix of another shape than its argument");
        assert!(w.iter().all(|v| v.is_finite()), "a parameter is not a real");
        let (mut lo, mut hi) = (vec![0.0f64; m], vec![0.0f64; m]);
        for (xi, row) in x.iter().zip(w.chunks_exact(m)) {
            // A term that is the real 0 adds nothing.
            if xi.lo == 0.0 && xi.hi == 0.0 {
                continue;
            }
            for o in 0..m {
                if row[o] == 0.0 {
                    continue;
                }
                let (a, b) =
                    if row[o] > 0.0 { (row[o] * xi.lo, row[o] * xi.hi) } else { (row[o] * xi.hi, row[o] * xi.lo) };
                lo[o] = (lo[o] + a.next_down()).next_down();
                hi[o] = (hi[o] + b.next_up()).next_up();
            }
        }
        lo.into_iter().zip(hi).map(|(lo, hi)| Iv::new(lo, hi)).collect()
    }
}

/// `a + b = s + e` in the reals for the rounded sum `s` (Knuth's TwoSum): the error of a sum of
/// two doubles is a double, and these operations compute it. If one of them overflows, `e` is
/// not finite, which the callers refuse.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let part = s - a;
    (s, (a - (s - part)) + (b - part))
}

/// `a * b = p + t` in the reals for some `t` of the interval returned with `p`. Where the
/// rounded product is at least `TINY` in absolute value its error is a double, which the fused
/// product `a * b - p` computes without rounding, and the interval is that point. A product that
/// overflows is refused.
fn product(a: f64, b: f64) -> (f64, Iv) {
    if a == 0.0 || b == 0.0 {
        assert!(a.is_finite() && b.is_finite(), "a factor that is not a real: {a} {b}");
        return (0.0, Iv::point(0.0));
    }
    let p = a * b;
    if p.abs() >= TINY { (p, Iv::point(a.mul_add(b, -p))) } else { (0.0, Iv::point(a) * Iv::point(b)) }
}

/// The reals `head + t` for `t` in `tail`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wide {
    head: f64,
    tail: Iv,
}

impl Wide {
    /// The reals `head + t` for `t` in `tail`, with the head moved toward the middle `m` of a
    /// bounded tail: `head + m = s + e` exactly, so `head + t = s + (e + (t - m))`.
    fn new(head: f64, tail: Iv) -> Wide {
        assert!(head.is_finite(), "the head of a set is not finite");
        let middle = 0.5 * (tail.lo + tail.hi);
        if !middle.is_finite() {
            return Wide { head, tail };
        }
        let (s, e) = two_sum(head, middle);
        if s == head {
            return Wide { head, tail };
        }
        assert!(s.is_finite(), "the head of a set is not finite");
        Wide { head: s, tail: tail - Iv::point(middle) + Iv::point(e) }
    }

    /// The negatives.
    fn neg(self) -> Wide {
        Wide { head: -self.head, tail: Iv::new(-self.tail.hi, -self.tail.lo) }
    }
}

impl Add for Wide {
    type Output = Wide;
    fn add(self, other: Wide) -> Wide {
        let (s, e) = two_sum(self.head, other.head);
        Wide::new(s, self.tail + other.tail + Iv::point(e))
    }
}

impl Sub for Wide {
    type Output = Wide;
    fn sub(self, other: Wide) -> Wide {
        self + other.neg()
    }
}

impl Mul for Wide {
    type Output = Wide;
    /// `(a + s)(b + t) = ab + a t + b s + s t`.
    fn mul(self, other: Wide) -> Wide {
        let (p, e) = product(self.head, other.head);
        Wide::new(p, e + other.tail.scale(self.head) + self.tail.scale(other.head) + self.tail * other.tail)
    }
}

impl Div for Wide {
    type Output = Wide;
    /// The quotients by a set of positive reals: `x / y = q + (x - q y) / y` for any double `q`,
    /// and with `x = a + s`, `y = b + t`, `q b = p + e` and `a - p = r + e'` the numerator is
    /// `r + e' - e + s - q t`.
    fn div(self, other: Wide) -> Wide {
        let divisor = other.iv();
        assert!(divisor.lo > 0.0, "a quotient by a set that is not positive");
        let q = self.head / other.head;
        if !q.is_finite() {
            return Wide::from_iv(self.iv() / divisor);
        }
        let (p, e) = product(q, other.head);
        let (r, rest) = two_sum(self.head, -p);
        let numerator = Iv::point(r) + Iv::point(rest) - e + self.tail - other.tail.scale(q);
        Wide::new(q, numerator / divisor)
    }
}

impl Enclosure for Wide {
    fn point(value: f64) -> Wide {
        assert!(value.is_finite(), "a point that is not a real: {value}");
        Wide { head: value, tail: Iv::point(0.0) }
    }

    fn from_iv(interval: Iv) -> Wide {
        Wide::new(0.0, interval)
    }

    fn iv(self) -> Iv {
        Iv::outward(self.head + self.tail.lo, self.head + self.tail.hi)
    }

    fn scale(self, w: f64) -> Wide {
        let (p, e) = product(w, self.head);
        Wide::new(p, e + self.tail.scale(w))
    }

    fn sqr(self) -> Wide {
        self * self
    }

    /// `sqrt x = r + (x - r^2) / (sqrt x + r)` for any positive double `r`, and with `x = a + s`,
    /// `r^2 = p + e` and `a - p = d + e'` the numerator is `d + e' - e + s`.
    fn sqrt(self) -> Wide {
        let coarse = self.iv();
        assert!(coarse.lo >= 0.0, "the square root of a set with negative reals");
        let root = self.head.sqrt();
        if !(root > 0.0 && root.is_finite()) {
            return Wide::from_iv(coarse.sqrt());
        }
        let (p, e) = product(root, root);
        let (d, rest) = two_sum(self.head, -p);
        let numerator = Iv::point(d) + Iv::point(rest) - e + self.tail;
        Wide::new(root, numerator / (coarse.sqrt() + Iv::point(root)))
    }

    /// With a head `a >= 0`, `max(0, a + t) = a + max(-a, t)`, which is exact at the ends of the
    /// tail. With a negative head the result is from 0 to the largest of 0 and `a + t`.
    fn relu(self) -> Wide {
        if self.head >= 0.0 {
            Wide { head: self.head, tail: Iv::new(self.tail.lo.max(-self.head), self.tail.hi.max(-self.head)) }
        } else {
            Wide { head: 0.0, tail: Iv::new(0.0, (self.head + self.tail.hi).next_up().max(0.0)) }
        }
    }

    fn widened(self, radius: f64) -> Wide {
        Wide::new(self.head, self.tail.widened(radius))
    }

    /// `e^(a + t) = e^a e^t`, and `e^t - 1 - t` is from 0 to `t^2` for `|t| <= 1/2`: it is
    /// `t^2 (1/2 + t/6 + ...)`, at most `t^2 e^(1/2) / 2`. Beyond that, and beyond `±700`, the
    /// exponentials are those of an interval of doubles.
    fn exp(self) -> Wide {
        let coarse = self.iv();
        if self.head.abs() > 700.0 || coarse.lo < -700.0 || coarse.hi > 700.0 || self.tail.mag() > 0.5 {
            return Wide::from_iv(coarse.exp());
        }
        let t = self.tail;
        let rest = Wide::new(1.0, Iv::new(t.lo, (t.hi + t.sqr().hi).next_up()));
        exp_point::<Wide>(self.head) * rest
    }

    /// The heads' products are summed with the errors of the sums and of the products sent to
    /// the tail: `w x = p + e`, `head + p = s + e'`, and the tail gains `e + e' + w t`.
    fn apply(x: &[Wide], w: &[f64], m: usize) -> Vec<Wide> {
        assert_eq!(w.len(), x.len() * m, "a matrix of another shape than its argument");
        assert!(w.iter().all(|v| v.is_finite()), "a parameter is not a real");
        let (mut head, mut lo, mut hi) = (vec![0.0f64; m], vec![0.0f64; m], vec![0.0f64; m]);
        for (xi, row) in x.iter().zip(w.chunks_exact(m)) {
            // A term that is the real 0 adds nothing.
            if xi.head == 0.0 && xi.tail.lo == 0.0 && xi.tail.hi == 0.0 {
                continue;
            }
            for o in 0..m {
                let c = row[o];
                if c == 0.0 {
                    continue;
                }
                let p = c * xi.head;
                let (p, e) = if p.abs() >= TINY {
                    let e = c.mul_add(xi.head, -p);
                    (p, Iv { lo: e, hi: e })
                } else {
                    (0.0, product(c, xi.head).1)
                };
                let (s, rest) = two_sum(head[o], p);
                head[o] = s;
                let (a, b) = if c > 0.0 { (c * xi.tail.lo, c * xi.tail.hi) } else { (c * xi.tail.hi, c * xi.tail.lo) };
                lo[o] = (lo[o] + ((e.lo + rest).next_down() + a.next_down()).next_down()).next_down();
                hi[o] = (hi[o] + ((e.hi + rest).next_up() + b.next_up()).next_up()).next_up();
            }
        }
        (0..m).map(|o| Wide::new(head[o], Iv::new(lo[o], hi[o]))).collect()
    }
}

/// A set that holds `e^x` for the real that the double `x` is, `|x| <= 700`.
fn exp_point<T: Enclosure>(x: f64) -> T {
    assert!(x.abs() <= 700.0, "an exponent beyond 700: {x}");
    // `x = 2^halvings r` with `|r| <= 1/2`: a product by 1/2 above 1/4 is exact.
    let (mut r, mut halvings) = (x, 0);
    while r.abs() > 0.5 {
        r *= 0.5;
        halvings += 1;
    }
    // The series to the term `r^30 / 30!`. The remainder is `e^s r^31 / 31!` for some `s`
    // between 0 and `r`, at most twice the next term: `e^(1/2) < 2`.
    let (mut sum, mut term) = (T::point(1.0), T::point(1.0));
    for n in 1..=30 {
        term = term.scale(r) / T::point(n as f64);
        sum = sum + term;
    }
    let next = term.scale(r) / T::point(31.0);
    let mut e = sum.widened((2.0 * next.iv().mag()).next_up());
    // `e^x = (e^r)^(2^halvings)`.
    for _ in 0..halvings {
        e = e.sqr();
    }
    e
}

/// Sets that hold the sines and the cosines of a set within `[-8, 8]`: the series to the
/// degrees 81 and 80, with remainders at most `|a|^83 / 83!` and `|a|^82 / 82!`, the next
/// terms, since every derivative of the two is at most 1 in absolute value.
pub fn sin_cos<T: Enclosure>(a: T) -> (T, T) {
    let coarse = a.iv();
    assert!(coarse.lo >= -8.0 && coarse.hi <= 8.0, "an angle outside [-8, 8]");
    let square = a.sqr();
    let (mut sin, mut cos) = (a, T::point(1.0));
    let (mut odd, mut even) = (a, T::point(1.0));
    for n in 1..=40 {
        even = even * square / T::point(((2 * n - 1) * (2 * n)) as f64);
        odd = odd * square / T::point(((2 * n) * (2 * n + 1)) as f64);
        (sin, cos) = if n % 2 == 1 { (sin - odd, cos - even) } else { (sin + odd, cos + even) };
    }
    let next_even = even * square / T::point((81 * 82) as f64);
    let next_odd = odd * square / T::point((82 * 83) as f64);
    (sin.widened(next_odd.iv().mag().next_up()), cos.widened(next_even.iv().mag().next_up()))
}

/// Sets that hold the positional encoding `position` of the Lean model, `rows` rows of `d`:
/// `sin(pos / 10000^(i/d))` at even `i` and `cos(pos / 10000^((i-1)/d))` at odd `i`. The width
/// is a power of two, at least 4, so that `10000^(i/d) = (10^(4/d))^i` is a power of a repeated
/// square root of 10.
pub fn positions<T: Enclosure>(dims: Dims, rows: usize) -> Vec<T> {
    assert!(arithmetic_holds(), "this thread does not round to nearest or flushes subnormal numbers");
    let d = dims.d;
    assert!(d.is_power_of_two() && d >= 4, "the width is not a power of two from 4 on");
    let mut root = T::point(10.0);
    for _ in 2..d.trailing_zeros() {
        root = root.sqrt();
    }
    let mut power = vec![T::point(1.0); d];
    for i in 1..d {
        power[i] = power[i - 1] * root;
    }
    let mut pe = Vec::with_capacity(rows * d);
    for pos in 0..rows {
        for i in 0..d {
            let (sin, cos) = sin_cos(T::point(pos as f64) / power[i - i % 2]);
            pe.push(if i % 2 == 0 { sin } else { cos });
        }
    }
    pe
}

/// `layerNorm` of the Lean model: the entries less their mean, over the root of the mean of the
/// squares of those differences and `floor`, times the gain, plus the bias.
fn normalize<T: Enclosure>(z: &[T], gain: &[f64], bias: &[f64], floor: T) -> Vec<T> {
    assert!(z.len() == gain.len() && z.len() == bias.len(), "gains or biases of another length than the entries");
    let n = T::point(z.len() as f64);
    let mean = z.iter().fold(T::point(0.0), |sum, v| sum + *v) / n;
    let centred: Vec<T> = z.iter().map(|v| *v - mean).collect();
    let root = (centred.iter().fold(T::point(0.0), |sum, c| sum + c.sqr()) / n + floor).sqrt();
    centred.iter().zip(gain).zip(bias).map(|((c, g), b)| (*c / root).scale(*g) + T::point(*b)).collect()
}

/// `1 / total` for a total of exponentials, which is positive. A total without an upper bound,
/// after an exponent beyond 700, leaves the quotients from 0 to the inverse of its lower end.
fn inverse<T: Enclosure>(total: T) -> T {
    let coarse = total.iv();
    assert!(coarse.lo > 0.0, "a total of exponentials that is not positive");
    if coarse.hi.is_finite() { T::point(1.0) / total } else { T::from_iv(Iv::new(0.0, (1.0 / coarse.lo).next_up())) }
}

/// Sets that hold `logits θ relu tokens at` of the Lean model, one for each token of the
/// vocabulary, where `θ` is the flat parameter vector `p` in the layout of `layout.rs` and `pe`
/// is [`positions`] of at least `at + 1` rows. The pass is causal: it reads the tokens up to
/// `at`, and the last block is computed at the position `at` alone.
pub fn logits_at<T: Enclosure>(dims: Dims, p: &[f64], pe: &[T], tokens: &[usize], at: usize) -> Vec<T> {
    assert!(arithmetic_holds(), "this thread does not round to nearest or flushes subnormal numbers");
    let (d, dk, dff) = (dims.d, dims.dk(), dims.dff);
    let floor = T::point(1.0) / T::point(100000.0);
    let root_dk = T::point(dk as f64).sqrt();
    let embed = |t: usize| (0..d).map(|i| T::point(p[dims.emb() + tokens[t] * d + i]) + pe[t * d + i]).collect();
    let mut x: Vec<Vec<T>> = (0..=at).map(embed).collect();
    for l in 0..dims.layers {
        let b = dims.block(l);
        let kv: Vec<Vec<T>> = x.iter().map(|row| T::apply(row, &p[b.wkv..b.wkv + d * 2 * d], 2 * d)).collect();
        let needed = if l + 1 == dims.layers { at..at + 1 } else { 0..at + 1 };
        let mut next = x.clone();
        for t in needed {
            let q = T::apply(&x[t], &p[b.wq..b.wq + d * d], d);
            let mut o = vec![T::point(0.0); d];
            for h in 0..dims.heads {
                let score = |j: usize| {
                    let products = (0..dk).map(|r| q[h * dk + r] * kv[j][h * dk + r]);
                    products.fold(T::point(0.0), |sum, v| sum + v) / root_dk
                };
                let scores: Vec<T> = (0..=t).map(score).collect();
                for j in 0..=t {
                    // The weight of `j` is `1 / sum_k e^(s_k - s_j)`, and the term of `k = j` is 1.
                    let others = (0..=t).filter(|k| *k != j).map(|k| (scores[k] - scores[j]).exp());
                    let weight = inverse(others.fold(T::point(1.0), |sum, v| sum + v));
                    for r in 0..dk {
                        o[h * dk + r] = o[h * dk + r] + weight * kv[j][d + h * dk + r];
                    }
                }
            }
            let mixed = T::apply(&o, &p[b.wo..b.wo + d * d], d);
            let z1: Vec<T> = (0..d).map(|i| x[t][i] + mixed[i]).collect();
            let y1 = normalize(&z1, &p[b.g1..b.g1 + d], &p[b.b1..b.b1 + d], floor);
            let hidden: Vec<T> = T::apply(&y1, &p[b.w1..b.w1 + d * dff], dff).iter().map(|v| v.relu()).collect();
            let f2 = T::apply(&hidden, &p[b.w2..b.w2 + dff * d], d);
            let z2: Vec<T> = (0..d).map(|i| y1[i] + f2[i]).collect();
            next[t] = normalize(&z2, &p[b.g2..b.g2 + d], &p[b.b2..b.b2 + d], floor);
        }
        x = next;
    }
    T::apply(&x[at], &p[dims.wout()..dims.wout() + d * dims.vocab], dims.vocab)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{self, ANSWER_POS, DIV, EOS, EQ, NUM0, P};
    use crate::reference;

    fn holds(interval: Iv, value: f64) -> bool {
        interval.lo <= value && value <= interval.hi
    }

    fn width(interval: Iv) -> f64 {
        interval.hi - interval.lo
    }

    /// Two sets that hold one real have a real in common.
    fn meet(a: Iv, b: Iv) -> bool {
        a.lo <= b.hi && b.lo <= a.hi
    }

    /// The operations enclose the real results: a third lies between the two doubles around
    /// it, a square root squared holds its number, and the ends of a product of two intervals
    /// that contain zero are products of ends.
    #[test]
    fn operations_enclose_the_real_results() {
        let third = Iv::point(1.0) / Iv::point(3.0);
        // 0.333...33 (16 digits) is below a third and its next double is above it.
        assert!(third.lo <= 0.3333333333333333 && third.hi >= 0.33333333333333337);
        assert!(width(third) <= 3.0 * f64::EPSILON);
        assert_eq!(3 * 6004799503160661_i64 + 1, 1_i64 << 54);
        assert_eq!(0.3333333333333333_f64.to_bits(), (6004799503160661_u64 & ((1 << 52) - 1)) | (1021 << 52));
        let two = Iv::point(2.0).sqrt().sqr();
        assert!(two.lo < 2.0 && 2.0 < two.hi && width(two) < 1e-14);
        let product = Iv { lo: -2.0, hi: 3.0 } * Iv { lo: -5.0, hi: 7.0 };
        assert!(product.lo <= -15.0 && product.lo > -15.000001 && product.hi >= 21.0 && product.hi < 21.000001);
        let tenth = Iv::point(0.1) + Iv::point(0.2);
        // The doubles 0.1 and 0.2 sum to more than the double 0.3 and to less than its next.
        assert!(tenth.lo <= 0.3 && tenth.hi >= 0.30000000000000004);
        assert_eq!(Iv { lo: -3.0, hi: 2.0 }.sqr().lo, 0.0);
        assert_eq!(Iv { lo: -3.0, hi: 2.0 }.relu(), Iv { lo: 0.0, hi: 2.0 });
        assert!(holds(Iv { lo: 1.0, hi: 2.0 }.scale(-3.0), -6.0) && holds(Iv { lo: 1.0, hi: 2.0 }.scale(-3.0), -3.0));
        assert!(holds(Iv { lo: 1.0, hi: 2.0 } - Iv { lo: 0.5, hi: 4.0 }, -3.0));
        // An end without a bound stays one, and 0 times a set without a bound is 0.
        let above = Iv { lo: 1.0, hi: f64::INFINITY };
        assert_eq!((above + Iv::point(1.0)).hi, f64::INFINITY);
        assert_eq!(above.scale(0.0), Iv::point(0.0));
        assert!(holds(Iv::point(1.0) / above, 0.5) && (Iv::point(1.0) / above).lo <= 0.0);
        let mapped = Iv::apply(&[above, Iv::point(2.0)], &[0.0, 3.0], 1);
        assert!(mapped.len() == 1 && holds(mapped[0], 6.0) && width(mapped[0]) < 1e-14, "{mapped:?}");
    }

    /// What is not a set is refused: a product of 0 and an end without a bound, and a point
    /// that is not a number.
    #[test]
    fn what_is_not_a_set_is_refused() {
        let refused = |f: fn() -> Iv| std::panic::catch_unwind(f).is_err();
        assert!(refused(|| Iv { lo: 0.0, hi: 0.0 } * Iv { lo: 1.0, hi: f64::INFINITY }));
        assert!(refused(|| Iv::point(f64::NAN)));
        assert!(refused(|| Iv::point(f64::INFINITY)));
        assert!(refused(|| Iv::point(1.0) / Iv { lo: 0.0, hi: 1.0 }));
        assert!(refused(|| Iv { lo: -1.0, hi: 1.0 }.sqrt()));
        assert!(refused(|| Iv::point(1.0).scale(f64::NAN)));
        assert!(refused(|| Iv::apply(&[Iv::point(1.0)], &[f64::NAN], 1)[0]));
        assert!(refused(|| Wide::point(f64::INFINITY).iv()));
        assert!(refused(|| (Wide::point(1e200) * Wide::point(1e200)).iv()));
        assert!(refused(|| Wide::apply(&[Wide::point(1e200)], &[1e200], 1)[0].iv()));
    }

    /// The arithmetic is checked where a pass begins: a thread that flushes subnormal numbers
    /// to zero, as the trainer's `--flush-from` does, is refused, and the others are not.
    #[test]
    fn a_thread_that_flushes_subnormal_numbers_is_refused() {
        assert!(arithmetic_holds());
        let flushing = std::thread::spawn(|| {
            let _restore = crate::arith::Restore;
            crate::arith::flush_subnormals(true);
            let refused = std::panic::catch_unwind(|| positions::<Iv>(Dims::PAPER, 1)).is_err();
            (crate::arith::flushes_subnormals(), arithmetic_holds(), refused)
        });
        assert_eq!(flushing.join().expect("the thread that flushes"), (true, false, true));
        assert!(arithmetic_holds());
    }

    /// A `Wide` holds the digits that a double drops: a third times three less one is 0 to 30
    /// digits, the doubles 0.1 and 0.2 sum to the double 0.3 and `2^-55`, the square of
    /// `1 + 2^-30` has its `2^-60`, and a product too small to split is still enclosed.
    #[test]
    fn a_wide_holds_the_digits_that_a_double_drops() {
        let third = Wide::point(1.0) / Wide::point(3.0);
        let back = (third.scale(3.0) - Wide::point(1.0)).iv();
        assert!(holds(back, 0.0) && width(back) < 1e-30, "{back:?}");
        let rest = (Wide::point(0.1) + Wide::point(0.2) - Wide::point(0.3)).iv();
        assert!(holds(rest, 2f64.powi(-55)) && width(rest) < 1e-30, "{rest:?}");
        let small = 2f64.powi(-30);
        let square = (Wide::point(1.0 + small).sqr() - Wide::point(1.0 + 2.0 * small)).iv();
        assert!(holds(square, 2f64.powi(-60)) && width(square) < 1e-30, "{square:?}");
        let two = (Wide::point(2.0).sqrt().sqr() - Wide::point(2.0)).iv();
        assert!(holds(two, 0.0) && width(two) < 1e-29, "{two:?}");
        let quotient = ((Wide::point(2.0).sqrt() / Wide::point(7.0)).scale(7.0).sqr() - Wide::point(2.0)).iv();
        assert!(holds(quotient, 0.0) && width(quotient) < 1e-29, "{quotient:?}");
        // 1e-150 * 1e-150 is below `TINY`: the head is 0 and the tail holds the product.
        let (head, tail) = product(1e-150, 1e-150);
        // The real product is within 3 parts in 10^16 of the double 1e-300.
        assert!(head == 0.0 && meet(tail, Iv { lo: 1e-300 * (1.0 - 1e-15), hi: 1e-300 * (1.0 + 1e-15) }), "{tail:?}");
        let faint = (Wide::point(1e-150) * Wide::point(1e-150)).iv();
        assert!(meet(faint, tail) && width(tail) < 1e-314 && width(faint) < 1e-313, "{faint:?} {tail:?}");
        assert_eq!(product(3.0, 0.0), (0.0, Iv::point(0.0)));
        // `two_sum` and `product` are exact: 2^53 + 1 and (2^27 + 1)^2 are not doubles.
        assert_eq!(two_sum(9007199254740992.0, 1.0), (9007199254740992.0, 1.0));
        assert_eq!(product(134217729.0, 134217729.0), (18014398777917440.0, Iv::point(1.0)));
        assert_eq!((134217729_u64 * 134217729, 1_u64 << 54), (18014398777917441, 18014398509481984));
        // `max 0` of sets on both sides of 0, below it and above it.
        let across = Wide::from_iv(Iv { lo: -1.0, hi: 2.0 }).relu().iv();
        assert!(across.lo <= 0.0 && across.lo > -1e-300 && holds(across, 2.0) && across.hi < 2.000001, "{across:?}");
        assert!(Wide::from_iv(Iv { lo: -3.0, hi: -1.0 }).relu().iv().mag() < 1e-300);
        let above = Wide::from_iv(Iv { lo: 1.0, hi: 3.0 }).relu().iv();
        assert!(holds(above, 1.0) && holds(above, 3.0) && width(above) < 2.000001);
        // A linear map: 3 * (1/3) + (-7) * (1/7) is 0 to 30 digits.
        let seventh = Wide::point(1.0) / Wide::point(7.0);
        let mapped = Wide::apply(&[third, seventh], &[3.0, 0.0, -7.0, 2.0], 2);
        assert!(holds(mapped[0].iv(), 0.0) && width(mapped[0].iv()) < 1e-30, "{:?}", mapped[0]);
        let twice = (mapped[1] - seventh.scale(2.0)).iv();
        assert!(holds(twice, 0.0) && width(twice) < 1e-30, "{twice:?}");
    }

    /// The series enclose the exponential, the sine and the cosine: `e` and `1/e` between the
    /// doubles around them, the sine of the double nearest to `π`, which is `π` less that
    /// double to 30 digits, and the library's values on a grid.
    #[test]
    fn series_enclose_the_exponential_the_sine_and_the_cosine() {
        // e = 2.71828182845904523536...: the double below it and the next.
        let e = Iv::point(1.0).exp();
        assert!(e.lo <= 2.718281828459045 && e.hi >= 2.7182818284590455 && width(e) < 1e-13, "{e:?}");
        // 1/e = 0.36787944117144232159...
        let inverse = Iv::point(-1.0).exp();
        assert!(inverse.lo <= 0.3678794411714423 && inverse.hi >= 0.36787944117144233, "{inverse:?}");
        assert!(width(inverse) < 1e-14);
        assert!(holds(Iv::point(0.0).exp(), 1.0) && width(Iv::point(0.0).exp()) < 1e-13);
        assert_eq!((Iv::point(701.0).exp().hi, Iv::point(-701.0).exp().lo), (f64::INFINITY, 0.0));
        assert!(Iv::point(701.0).exp().lo >= 1e300 && Iv::point(-701.0).exp().hi <= 1e-300);
        // π - fl(π) = 1.2246467991473532e-16, and sin(fl(π)) is that to its last digit.
        let (sin, cos) = sin_cos(Iv::point(std::f64::consts::PI));
        assert!(holds(sin, 1.2246467991473532e-16) && width(sin) < 1e-13, "{sin:?}");
        assert!(holds(cos, -1.0) || (cos.lo + 1.0).abs() < 1e-14, "{cos:?}");
        for step in -700..=700 {
            let x = step as f64 * 0.917;
            let (low, high) = (Iv::point(x).exp(), x.exp());
            assert!(low.lo <= high * (1.0 + 1e-15) && high * (1.0 - 1e-15) <= low.hi, "exp {x}: {low:?} {high}");
            assert!(width(low) <= 1e-10 * high, "exp {x}: {low:?}");
            let product = Iv::point(x).exp() * Iv::point(-x).exp();
            assert!(holds(product, 1.0) && width(product) < 1e-9, "exp {x} exp -{x}: {product:?}");
        }
        for step in -80..=80 {
            // To 4.965 in absolute value the terms stay small; to 7.944 they reach 416.
            let (x, bound) = if step % 8 == 0 { (step as f64 * 0.0993, 1e-9) } else { (step as f64 * 0.062, 1e-12) };
            let (sin, cos) = sin_cos(Iv::point(x));
            assert!(sin.lo - 1e-15 <= x.sin() && x.sin() <= sin.hi + 1e-15 && width(sin) < bound, "sin {x}: {sin:?}");
            assert!(cos.lo - 1e-15 <= x.cos() && x.cos() <= cos.hi + 1e-15 && width(cos) < bound, "cos {x}: {cos:?}");
            let unit = sin.sqr() + cos.sqr();
            assert!(holds(unit, 1.0) && width(unit) < 100.0 * bound, "sin² + cos² at {x}: {unit:?}");
        }
    }

    /// The series in a `Wide` hold thirty digits: `e` less its double is `1.4456468917292502e-16`
    /// and `π` less its double is `1.2246467991473532e-16`, both to `3e-33`; on a grid the
    /// exponentials of `x` and `-x` multiply to 1, the squares of the sine and the cosine sum to
    /// 1, and each set has a real in common with the interval of doubles.
    #[test]
    fn series_in_a_wide_hold_thirty_digits() {
        let rest = (Wide::point(1.0).exp() - Wide::point(std::f64::consts::E)).iv();
        assert!((rest.lo - 1.4456468917292502e-16).abs() < 1e-29 && width(rest) < 1e-29, "{rest:?}");
        let (sin, cos) = sin_cos(Wide::point(std::f64::consts::PI));
        let (sin, cos) = (sin.iv(), (cos + Wide::point(1.0)).iv());
        assert!((sin.lo - 1.2246467991473532e-16).abs() < 1e-28 && width(sin) < 1e-28, "{sin:?}");
        // 1 + cos(fl(π)) is half the square of that difference, 7.5e-33.
        assert!(cos.lo <= cos.hi && cos.lo > -1e-28 && cos.hi < 1e-28, "{cos:?}");
        for step in -700..=700 {
            let x = step as f64 * 0.917;
            let (up, down) = (Wide::point(x).exp(), Wide::point(-x).exp());
            let unit = (up * down - Wide::point(1.0)).iv();
            // Beyond 575 the smaller of the two is below `TINY`, where a product is an interval
            // of doubles.
            let bound = if x.abs() < 550.0 { 1e-24 } else { 1e-9 };
            assert!(holds(unit, 0.0) && width(unit) < bound, "exp {x} exp -{x}: {unit:?}");
            assert!(meet(up.iv(), Iv::point(x).exp()), "exp {x}: {up:?}");
        }
        for step in -80..=80 {
            let x = if step % 8 == 0 { step as f64 * 0.0993 } else { step as f64 * 0.062 };
            let (sin, cos) = sin_cos(Wide::point(x));
            let unit = (sin.sqr() + cos.sqr() - Wide::point(1.0)).iv();
            assert!(holds(unit, 0.0) && width(unit) < 1e-24, "sin² + cos² at {x}: {unit:?}");
            let (coarse_sin, coarse_cos) = sin_cos(Iv::point(x));
            assert!(meet(sin.iv(), coarse_sin) && meet(cos.iv(), coarse_cos), "sin and cos at {x}");
        }
        // An exponent that is a set: e^[1, 1 + 1e-9] holds e and e^(1 + 1e-9).
        let about = Wide::new(1.0, Iv { lo: 0.0, hi: 1e-9 }).exp().iv();
        assert!(about.lo <= 2.718281828459045 && about.hi >= 2.7182818311773268 && width(about) < 3e-9, "{about:?}");
        // Beyond 700 the exponentials have no upper bound, and their inverse is from 0.
        let beyond = Wide::point(701.0).exp();
        assert!(beyond.iv().lo >= 9e299 && beyond.iv().hi == f64::INFINITY);
        let share = inverse(Wide::point(1.0) + beyond).iv();
        assert!(share.lo <= 0.0 && share.lo > -1e-310 && share.hi <= 1e-299, "{share:?}");
        assert_eq!(inverse(Iv::point(1.0) + Iv::point(701.0).exp()).lo, 0.0);
    }

    /// The positions enclose the sinusoids: 0 and 1 at the position 0, `sin 1` and `cos 1` in the
    /// first two channels of the position 1, the library's values to `1e-12` elsewhere, and
    /// `10000^(i/d)` is `10^(i/2)` at the width 8. The positions in a `Wide` are narrower than
    /// `1e-28` and have a real in common with those.
    #[test]
    fn positions_enclose_the_sinusoids() {
        let dims = Dims::PAPER;
        let pe = positions::<Iv>(dims, 5);
        assert_eq!(pe.len(), 5 * 128);
        for i in 0..128 {
            assert!(holds(pe[i], (i % 2) as f64) && width(pe[i]) < 1e-13, "position 0, channel {i}");
        }
        // sin 1 = 0.84147098480789650665..., cos 1 = 0.54030230586813971740...
        assert!(pe[128].lo <= 0.8414709848078965 && pe[128].hi >= 0.8414709848078966, "{:?}", pe[128]);
        assert!(pe[129].lo <= 0.5403023058681397 && pe[129].hi >= 0.5403023058681398, "{:?}", pe[129]);
        for pos in 0..5 {
            for i in 0..128 {
                let angle = pos as f64 / 10000f64.powf((i - i % 2) as f64 / 128.0);
                let value = if i % 2 == 0 { angle.sin() } else { angle.cos() };
                let found = pe[pos * 128 + i];
                assert!(found.lo - 1e-12 <= value && value <= found.hi + 1e-12, "position {pos}, channel {i}");
                assert!(width(found) < 1e-12, "position {pos}, channel {i}: {found:?}");
            }
        }
        for (index, (narrow, coarse)) in positions::<Wide>(dims, 5).iter().zip(&pe).enumerate() {
            assert!(meet(narrow.iv(), *coarse) && width(narrow.tail) < 1e-28, "entry {index}: {narrow:?}");
        }
        let small = Dims { vocab: 11, d: 8, heads: 2, layers: 2, dff: 16, ctx: 6 };
        let narrow = positions::<Iv>(small, 4);
        // The channel 4 of the position 3 is sin(3 / 10000^(4/8)) = sin(3 / 100).
        assert!(holds(narrow[3 * 8 + 4], 0.03_f64.sin()) || (narrow[3 * 8 + 4].lo - 0.03_f64.sin()).abs() < 1e-15);
        assert!(width(narrow[3 * 8 + 4]) < 1e-14);
    }

    /// A stream of doubles in `[-1, 1)` for parameters.
    fn stream(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// Parameters of the given size, with gains near 1.
    fn random(dims: Dims, scale: f64, seed: u64) -> Vec<f64> {
        let mut next = stream(seed);
        let mut p: Vec<f64> = (0..dims.len()).map(|_| (scale * next()) as f32 as f64).collect();
        for l in 0..dims.layers {
            let b = dims.block(l);
            for i in 0..dims.d {
                (p[b.g1 + i], p[b.g2 + i]) = (1.0 + 0.3 * p[b.g1 + i], 1.0 + 0.3 * p[b.g2 + i]);
            }
        }
        p
    }

    /// At random parameters of a small model and of the paper's, the enclosures are narrow and
    /// are near the logits of the model in doubles, which rounds the positions to `f32`; the
    /// sets in a `Wide` are narrower than `1e-20` and have a real in common with them.
    #[test]
    fn enclosures_are_near_the_model_in_doubles() {
        let small = Dims { vocab: 11, d: 8, heads: 2, layers: 2, dff: 16, ctx: 6 };
        for (dims, scale, seed) in [(small, 1.0, 7), (Dims::PAPER, 0.05, 11)] {
            let p = random(dims, scale, seed);
            let tokens = [0, 7, 10, 3, 1, 5];
            let pe = positions::<Iv>(dims, ANSWER_POS + 1);
            let enclosed = logits_at(dims, &p, &pe, &tokens, ANSWER_POS);
            let wide = logits_at(dims, &p, &positions::<Wide>(dims, ANSWER_POS + 1), &tokens, ANSWER_POS);
            let doubles = reference::logits(dims, &p, &tokens);
            assert_eq!((enclosed.len(), wide.len()), (dims.vocab, dims.vocab));
            for (c, interval) in enclosed.iter().enumerate() {
                let value = doubles[ANSWER_POS * dims.vocab + c];
                assert!(interval.lo <= interval.hi && width(*interval) < 1e-4, "token {c}: {interval:?}");
                assert!((0.5 * (interval.lo + interval.hi) - value).abs() < 1e-4, "token {c}: {interval:?}, {value}");
                assert!(meet(wide[c].iv(), *interval) && width(wide[c].tail) < 1e-20, "token {c}: {:?}", wide[c]);
            }
            // The pass is causal: the token after the answer's position is not read.
            let other = [0, 7, 10, 3, 1, 9];
            assert_eq!(logits_at(dims, &p, &pe, &other, ANSWER_POS), enclosed);
        }
    }

    /// At parameters ten times larger the intervals of doubles are wide, and the sets in a
    /// `Wide` are still narrower than `1e-12` and have a real in common with them.
    #[test]
    fn a_wide_stays_narrow_where_the_intervals_of_doubles_do_not() {
        let dims = Dims::PAPER;
        let p = random(dims, 0.5, 13);
        let tokens = [0, 7, 10, 3, 1, 5];
        let coarse = logits_at(dims, &p, &positions::<Iv>(dims, ANSWER_POS + 1), &tokens, ANSWER_POS);
        let wide = logits_at(dims, &p, &positions::<Wide>(dims, ANSWER_POS + 1), &tokens, ANSWER_POS);
        let widest = coarse.iter().map(|v| width(*v)).fold(0.0, f64::max);
        assert!(widest > 1e-3, "the intervals of doubles are {widest} wide");
        for (c, narrow) in wide.iter().enumerate() {
            assert!(meet(narrow.iv(), coarse[c]) && width(narrow.tail) < 1e-12, "token {c}: {narrow:?}");
        }
    }

    /// The parameters `solver` of `Protocol/SolverParams.lean` in the layout of `layout.rs`, with
    /// the irrational ones in doubles.
    fn solver(dims: Dims) -> Vec<f64> {
        let (d, dk, dff, vocab) = (dims.d, dims.dk(), dims.dff, dims.vocab);
        let position = |pos: usize, i: usize| {
            let angle = pos as f64 / 10000f64.powf((i - i % 2) as f64 / d as f64);
            if i % 2 == 0 { angle.sin() } else { angle.cos() }
        };
        let ind = |k: usize, i: usize| if i == k { 1.0 } else { 0.0 };
        let ballast = 10000.0;
        // `logFeature`: the logarithm to the base 5, and 300 for the residue 0.
        let mut feature = [300.0; P];
        let mut power = 1;
        for k in 0..P - 1 {
            feature[power] = k as f64;
            power = power * 5 % P;
        }
        assert_eq!(power, 1);
        let key_exp = |j: usize| position(j, 1).exp();
        let mix_uniform = -5.0 * (key_exp(1) + key_exp(3)) / (key_exp(1) - key_exp(3));
        let mix_keyed = 2.0 * (3.0 + key_exp(1) + key_exp(3)) / (key_exp(1) - key_exp(3));
        let target = |i: usize| ballast * ind(3, i) - (position(1, 0) - position(3, 0)) * ind(2, i);
        let mut p = vec![0.0; dims.len()];
        for token in 0..vocab {
            for i in 0..d {
                p[dims.emb() + token * d + i] = match token {
                    EOS => -position(0, i),
                    DIV => -position(2, i),
                    EQ => target(i) - position(4, i),
                    _ => ind(0, i) * feature[token.saturating_sub(NUM0) % P],
                };
                // `solverOut`: the residue `r` reads the entry `5 + r` less the entry 4.
                if (NUM0..NUM0 + P).contains(&token) {
                    p[dims.wout() + i * vocab + token] = ind(token - 17, i) - ind(4, i);
                }
            }
        }
        for l in 0..dims.layers {
            let b = dims.block(l);
            for i in 0..d {
                (p[b.g1 + i], p[b.g2 + i]) = (1.0, 1.0);
            }
        }
        let b = dims.block(0);
        // The query, the key and the values of `solver`, at the row 0 of their heads.
        p[b.wq + 3 * d + dk] = (dk as f64).sqrt() / ballast;
        p[b.wkv + 2 * d + dk] = 1.0;
        (p[b.wkv + d], p[b.wkv + d + dk]) = (1.0, 1.0);
        // `proj`: the heads 0 and 1 write the entry 2.
        (p[b.wo + 2], p[b.wo + dk * d + 2]) = (mix_uniform, mix_keyed);
        for u in 0..dff {
            // `solverUp`, with the knot `u - 96`.
            let knot = u as f64 - 96.0;
            p[b.w1 + 2 * dff + u] = 1.0;
            p[b.w1 + 3 * dff + u] = -knot / ballast;
            p[b.w1 + 4 * dff + u] = knot / ballast - 1.0;
        }
        // `solverDown`: the hat of the gap `j - 95` at the entry of `5^(j - 95)`, and the ramp.
        let mut power = 5usize.pow(96 - 95 % 96) % P;
        for j in 0..191 {
            if j > 0 {
                power = power * 5 % P;
            }
            for (step, coefficient) in [1.0, -2.0, 1.0].into_iter().enumerate() {
                p[b.w2 + (j + step) * d + power + 5] += coefficient;
            }
        }
        p[b.w2 + 196 * d + 5] += 1.0;
        p[b.w2 + 197 * d + 5] -= 1.0;
        p
    }

    /// The pass computes the decoder of the Lean model: at the parameters `solver`, the logits
    /// are `solverScale x * markLogits x`, a number from `1/900` to `1/870` at the answer's
    /// token and 0 at the other 238 (`answerLogits_solver`, `solverScale_mem` of
    /// `Protocol/SolverLogits.lean` and `SolverScale.lean`), in both arithmetics.
    #[test]
    fn solver_parameters_give_the_logits_of_the_lean_theorem() {
        let dims = Dims::PAPER;
        let (p, pe) = (solver(dims), positions::<Iv>(dims, ANSWER_POS + 1));
        let wide_pe = positions::<Wide>(dims, ANSWER_POS + 1);
        // 5^1 = 5 at the gap 1: the first power of the table is 5^(-95) = 5^1.
        assert_eq!(5usize.pow(96 - 95 % 96) % P, 5);
        let equations = data::equations();
        let chosen = equations.iter().step_by(613).chain(&equations[..3]).chain(&equations[9309..]);
        let mut seen = 0;
        for equation in chosen {
            let tokens = equation.tokens();
            let enclosed = logits_at(dims, &p, &pe, &tokens[..dims.ctx], ANSWER_POS);
            let wide = logits_at(dims, &p, &wide_pe, &tokens[..dims.ctx], ANSWER_POS);
            let answer = NUM0 + equation.quo as usize;
            for (token, interval) in enclosed.iter().enumerate() {
                assert!(width(*interval) < 1e-9, "{equation:?}, token {token}: {interval:?}");
                let narrow = wide[token];
                assert!(
                    meet(narrow.iv(), *interval) && width(narrow.tail) < 1e-20,
                    "{equation:?}, {token}: {narrow:?}"
                );
                for found in [*interval, narrow.iv()] {
                    if token == answer {
                        assert!(
                            found.lo > 1.0 / 900.0 - 1e-9 && found.hi < 1.0 / 870.0 + 1e-9,
                            "{equation:?}: {found:?}"
                        );
                    } else {
                        assert!(found.lo > -1e-9 && found.hi < 1e-9, "{equation:?}, token {token}: {found:?}");
                    }
                }
            }
            seen += 1;
        }
        assert_eq!(seen, 16 + 3 + 3);
    }
}
