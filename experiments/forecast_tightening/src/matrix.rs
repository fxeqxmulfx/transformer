//! FT2: operator amplification bounded through the smaller Gram matrix.
//! Each FP64 operation is expanded outward; no approximate singular vector.
use serde_json::{Value, json};
#[derive(Clone, Copy)]
pub struct Bound {
    pub frobenius: f64,
    pub gram: f64,
    pub factor: f64,
    pub dimension: usize,
}
pub fn norm_upper(x: &[f64]) -> f64 {
    x.iter()
        .fold(0f64, |s, &v| (s + (v * v).next_up()).next_up())
        .sqrt()
        .next_up()
}
pub fn bound(matrix: &[f64], rows: usize, cols: usize) -> Bound {
    assert_eq!(matrix.len(), rows * cols);
    assert!(matrix.iter().all(|x| x.is_finite()));
    let n = rows.min(cols);
    let inner = rows.max(cols);
    let at = |i, k| {
        if rows <= cols {
            matrix[i * cols + k]
        } else {
            matrix[k * cols + i]
        }
    };
    let mut square = 0f64;
    for i in 0..n {
        for j in i..n {
            let (mut lo, mut hi) = (0f64, 0f64);
            for k in 0..inner {
                let p = at(i, k) * at(j, k);
                lo = (lo + p.next_down()).next_down();
                hi = (hi + p.next_up()).next_up();
            }
            let magnitude = lo.abs().max(hi.abs());
            let term = (magnitude * magnitude).next_up();
            square = (square + term).next_up();
            if i != j {
                square = (square + term).next_up();
            }
        }
    }
    let gram = square.sqrt().next_up().sqrt().next_up();
    let frobenius = norm_upper(matrix);
    assert!(gram.is_finite() && frobenius.is_finite());
    Bound {
        frobenius,
        gram,
        factor: frobenius.min(gram),
        dimension: n,
    }
}
impl Bound {
    pub fn json(self) -> Value {
        json!({"frobenius_upper":self.frobenius,"gram_upper":self.gram,
            "selected":self.factor,"gram_dimension":self.dimension})
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_rank_one_and_rectangular_transpose_have_valid_upper_bounds() {
        let mut identity = vec![0.; 16];
        for i in 0..4 {
            identity[5 * i] = 1.;
        }
        let b = bound(&identity, 4, 4);
        assert!(b.factor >= 2f64.sqrt() && b.factor < 1.415);
        assert!(b.factor < b.frobenius);
        let rank_one = bound(&[1., 2., 3., 2., 4., 6.], 2, 3);
        let transposed = bound(&[1., 2., 2., 4., 3., 6.], 3, 2);
        assert!(rank_one.factor >= 70f64.sqrt());
        assert!((rank_one.factor - 70f64.sqrt()).abs() < 1e-12);
        assert_eq!(rank_one.factor, transposed.factor);
        let zero = bound(&[0.; 6], 2, 3);
        assert!(zero.factor >= 0. && zero.factor < 1e-100);
    }
}
