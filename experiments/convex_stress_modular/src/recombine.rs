//! CM27 fits_recombine/product_dvd with an actual checked Bezout witness.
use crate::{Result, System, checked_residuals};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub system: System,
    pub left_modulus: u16,
    pub right_modulus: u16,
    pub left: Vec<u16>,
    pub right: Vec<u16>,
}
impl Input {
    fn validate(&self) -> Result<()> {
        self.system.validate_point(self.left_modulus, &self.left)?;
        self.system
            .validate_point(self.right_modulus, &self.right)?;
        if self.left_modulus < 2
            || self.right_modulus < 2
            || self.left_modulus as u32 * self.right_modulus as u32 > 257
        {
            return Err("CRT factors or product exceed bounded composite-domain support".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub input: Input,
    pub bezout: Option<[i64; 2]>,
    pub point: Option<Vec<u16>>,
    pub residuals: Option<Vec<i64>>,
    pub inverse_tests: usize,
    pub original_products: usize,
    pub reason: String,
}
pub fn run(input: Input) -> Result<Step> {
    input.validate()?;
    let (m, n) = (input.left_modulus as i64, input.right_modulus as i64);
    for (q, x) in [(m, &input.left), (n, &input.right)] {
        if input.system.residuals(x).iter().any(|d| d % q != 0) {
            return Err("CRT component does not fit the original system".into());
        }
    }
    let mut tests = 0;
    let witness = (0..n).find_map(|u| {
        tests += 1;
        ((1 - u * m) % n == 0).then_some([u, (1 - u * m) / n])
    });
    let products = input.system.rows.len() * input.system.variables;
    let mut out = Step {
        input,
        bezout: witness,
        point: None,
        residuals: None,
        inverse_tests: tests,
        original_products: 2 * products,
        reason: "unsupported_noncoprime_components".into(),
    };
    if let Some([u, v]) = witness {
        let point: Vec<_> = out
            .input
            .left
            .iter()
            .zip(&out.input.right)
            .map(|(&x, &y)| (v * n * x as i64 + u * m * y as i64).rem_euclid(m * n) as u16)
            .collect();
        let residuals = out.input.system.residuals(&point);
        if residuals.iter().any(|d| d % (m * n) != 0) {
            return Err("CRT point failed the original product-modulus equations".into());
        }
        out.point = Some(point);
        out.residuals = Some(residuals);
        out.original_products += products;
        out.reason = "recombined".into();
    }
    Ok(out)
}
pub fn check(input: &Input, out: &Step) -> Result<Value> {
    input.validate()?;
    checked_residuals(&input.system, input.left_modulus, &input.left)?;
    checked_residuals(&input.system, input.right_modulus, &input.right)?;
    if &out.input != input {
        return Err("CRT original input changed".into());
    }
    let (m, n) = (input.left_modulus as i128, input.right_modulus as i128);
    let products = input.system.rows.len() * input.system.variables;
    if let Some([u, v]) = out.bezout {
        if u < 0
            || u as i128 >= n
            || u as i128 * m + v as i128 * n != 1
            || out.inverse_tests != u as usize + 1
            || out.reason != "recombined"
            || out.original_products != 3 * products
        {
            return Err("invalid CRT Bezout witness or work".into());
        }
        let point = out.point.as_ref().ok_or("missing CRT point")?;
        let residuals = checked_residuals(&input.system, (m * n) as u16, point)?;
        if out.residuals.as_ref() != Some(&residuals) {
            return Err("CRT original residuals changed".into());
        }
        for ((&z, &x), &y) in point.iter().zip(&input.left).zip(&input.right) {
            let explicit =
                (v as i128 * n * x as i128 + u as i128 * m * y as i128).rem_euclid(m * n);
            if z as i128 != explicit || z as i128 % m != x as i128 || z as i128 % n != y as i128 {
                return Err("CRT actual representative or component residue differs".into());
            }
        }
    } else if out.point.is_some()
        || out.residuals.is_some()
        || out.inverse_tests != n as usize
        || out.original_products != 2 * products
        || out.reason != "unsupported_noncoprime_components"
        || (0..n).any(|u| (1 - u * m) % n == 0)
    {
        return Err("incorrect CRT refusal or inverse-search extent".into());
    }
    Ok(
        json!({"bezout_checked":out.bezout.is_some(),"component_residues_checked":out.point.is_some(),
        "original_equation_products":out.original_products,"reason":out.reason,
        "unseen_answers_identified":false}),
    )
}
pub fn replay(input: &Input, out: &Step) -> Result<()> {
    if &out.input != input || run(input.clone())? != *out {
        return Err("CRT original-input/representative/work replay mismatch".into());
    }
    Ok(())
}
