//! Exact native counterpart of CM27's residual_identity and lift_iff.
use crate::{Result, System, checked_residuals, finite_system};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub system: System,
    pub modulus: u16,
    pub prime: u16,
    pub base: Vec<u16>,
    pub policy: finite_system::Policy,
}
impl Input {
    fn problem(&self) -> Result<(Vec<i64>, finite_system::Problem, u64)> {
        self.system.validate_point(self.modulus, &self.base)?;
        if !finite_system::prime(self.prime) || self.modulus as u32 * self.prime as u32 > 257 {
            return Err("digit lift exceeds supported prime or total modulus257".into());
        }
        let difference = self.system.residuals(&self.base);
        if difference.iter().any(|d| d % (self.modulus as i64) != 0) {
            return Err("base representatives do not solve the original congruences".into());
        }
        let residual: Vec<_> = difference
            .iter()
            .map(|d| d / (self.modulus as i64))
            .collect();
        let p = self.prime as i64;
        let problem = finite_system::Problem {
            system: finite_system::System {
                prime: self.prime,
                variables: self.system.variables,
                rows: self
                    .system
                    .rows
                    .iter()
                    .map(|r| r.iter().map(|a| a.rem_euclid(p) as u16).collect())
                    .collect(),
                right: residual.iter().map(|b| b.rem_euclid(p) as u16).collect(),
            },
            policy: self.policy.clone(),
        };
        let bound = 1024 * 1024 * 1024
            + 4 * problem.validate()?
            + 64 * (self.system.rows.len() * self.system.variables) as u64;
        if bound > 8 * 1024 * 1024 * 1024 {
            return Err("digit-lift report/check/replay forecast exceeds8GiB".into());
        }
        Ok((residual, problem, bound))
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub base_products: usize,
    pub next_products: usize,
    pub finite_scalar_updates: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub input: Input,
    pub residual: Vec<i64>,
    pub elimination: finite_system::Outcome,
    pub elimination_audit: Value,
    pub digits: Option<Vec<u16>>,
    pub next: Option<Vec<u16>>,
    pub next_residuals: Option<Vec<i64>>,
    pub reason: String,
    pub work: Work,
    pub buffer_bound_bytes: u64,
}
pub fn run(input: Input) -> Result<Step> {
    let (residual, problem, buffer_bound_bytes) = input.problem()?;
    let elimination = finite_system::solve(problem.clone())?;
    let elimination_audit = finite_system::check(&problem, &elimination)?;
    let products = input.system.rows.len() * input.system.variables;
    let mut out = Step {
        work: Work {
            base_products: products,
            next_products: 0,
            finite_scalar_updates: elimination.work.scalar_updates,
        },
        reason: elimination.reason.clone(),
        input,
        residual,
        elimination,
        elimination_audit,
        digits: None,
        next: None,
        next_residuals: None,
        buffer_bound_bytes,
    };
    if let Some(space) = &out.elimination.space {
        // Retain the entire affine space and choose its declared free-zero
        // representative. No uniqueness follows when directions remain.
        let digits = space.particular.clone();
        let next: Vec<_> = out
            .input
            .base
            .iter()
            .zip(&digits)
            .map(|(&x, &y)| x + out.input.modulus * y)
            .collect();
        let difference = out.input.system.residuals(&next);
        let modulus = out.input.modulus * out.input.prime;
        if difference.iter().any(|d| d % (modulus as i64) != 0) {
            return Err("finite digit solution failed the original next-modulus system".into());
        }
        out.digits = Some(digits);
        out.next = Some(next);
        out.next_residuals = Some(difference);
        out.work.next_products = products;
        out.reason = "lifted".into();
    }
    Ok(out)
}
pub fn check(input: &Input, out: &Step) -> Result<Value> {
    let (_, problem, bound) = input.problem()?;
    let base_difference = checked_residuals(&input.system, input.modulus, &input.base)?;
    let residual: Vec<_> = base_difference
        .iter()
        .map(|d| d / (input.modulus as i64))
        .collect();
    for (original, reduced) in input.system.rows.iter().zip(&problem.system.rows) {
        for (&a, &z) in original.iter().zip(reduced) {
            let p = input.prime as i128;
            if ((a as i128 % p + p) % p) as u16 != z {
                return Err("independent coefficient reduction differs".into());
            }
        }
    }
    if &out.input != input
        || out.residual != residual
        || out.buffer_bound_bytes != bound
        || out.elimination_audit != finite_system::check(&problem, &out.elimination)?
        || problem.system.right
            != residual
                .iter()
                .map(|d| d.rem_euclid(input.prime as i64) as u16)
                .collect::<Vec<_>>()
    {
        return Err("integer residual, field reduction or row-operation provenance differs".into());
    }
    let products = input.system.rows.len() * input.system.variables;
    if out.work.base_products != products
        || out.work.finite_scalar_updates != out.elimination.work.scalar_updates
    {
        return Err("digit-lift base or elimination work differs".into());
    }
    if let Some(space) = &out.elimination.space {
        let digits = out
            .digits
            .as_ref()
            .ok_or("missing chosen digit representative")?;
        let next = out.next.as_ref().ok_or("missing lifted representatives")?;
        if digits != &space.particular
            || next.len() != input.system.variables
            || out.reason != "lifted"
            || out.work.next_products != products
        {
            return Err("wrong digit choice, output shape or work".into());
        }
        for ((&x, &y), &z) in input.base.iter().zip(digits).zip(next) {
            if z as u32 != x as u32 + input.modulus as u32 * y as u32 {
                return Err("actual digit-lift identity fails".into());
            }
        }
        let difference = checked_residuals(&input.system, input.modulus * input.prime, next)?;
        for (i, row) in input.system.rows.iter().enumerate() {
            let value: i128 = row
                .iter()
                .zip(digits)
                .map(|(&a, &y)| a as i128 * y as i128)
                .sum();
            if difference[i] as i128 != input.modulus as i128 * (residual[i] as i128 - value) {
                return Err("actual CM27 residual identity fails".into());
            }
        }
        if out.next_residuals.as_ref() != Some(&difference) {
            return Err("saved final original residuals differ".into());
        }
    } else if out.digits.is_some()
        || out.next.is_some()
        || out.next_residuals.is_some()
        || out.reason != out.elimination.reason
        || out.work.next_products != 0
    {
        return Err("incomplete or inconsistent digit system produced a point".into());
    }
    Ok(
        json!({"original_equations_checked":products+out.work.next_products,"residual_identity_checked":out.next.is_some(),
        "unique_digit":out.elimination.space.as_ref().is_some_and(|s|s.directions.is_empty()),
        "reason":out.reason,"unseen_answers_identified":false}),
    )
}
pub fn replay(input: &Input, out: &Step) -> Result<()> {
    if &out.input != input || run(input.clone())? != *out {
        return Err("digit-lift original-input/full-state/work replay mismatch".into());
    }
    Ok(())
}
