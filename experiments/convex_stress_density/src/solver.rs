//! Bounded ADMM for CM21's two convex linear programs; no rank constraint.
//! PSD/trace projection and affine projection are separate exact mathematical
//! operators, approximated in f64. Acceptance uses primal and dual diagnostics,
//! not iterate change alone. No convergence or interval certificate is claimed.
use crate::{C, Result, affine::Affine, matrix, memory_bound};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Options {
    pub iterations: usize,
    pub rho: f64,
    pub feasibility_tolerance: f64,
    pub gap_tolerance: f64,
    pub rank_cutoff: f64,
    pub purity_tolerance: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            iterations: 4096,
            rho: 1.,
            feasibility_tolerance: 1e-8,
            gap_tolerance: 1e-8,
            rank_cutoff: 1e-11,
            purity_tolerance: 1e-6,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        if !(1..=16_384).contains(&self.iterations)
            || !self.rho.is_finite()
            || !(1e-4..=1e4).contains(&self.rho)
            || [
                self.feasibility_tolerance,
                self.gap_tolerance,
                self.purity_tolerance,
            ]
            .iter()
            .any(|x| !x.is_finite() || !(1e-12..=1e-3).contains(x))
            || !self.rank_cutoff.is_finite()
            || !(1e-14..=1e-6).contains(&self.rank_cutoff)
        {
            return Err("invalid bounded density optimization options".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub iteration: usize,
    pub objective: f64,
    pub dual_bound: f64,
    pub dual_gap: f64,
    pub affine_residual: f64,
    pub splitting_residual: f64,
    pub dual_step: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stage {
    pub accepted: bool,
    pub reason: String,
    pub affine_rank: usize,
    pub affine_consistency: f64,
    pub matrix: Vec<C>,
    pub affine_state: Vec<f64>,
    pub scaled_dual: Vec<f64>,
    pub multipliers: Vec<f64>,
    pub trace: Vec<Step>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Extraction {
    pub n: usize,
    pub rank: usize,
    pub rows: [usize; 2],
    pub options: Options,
    pub buffer_bound_bytes: u64,
    pub centered_residual: f64,
    pub gram_residual: f64,
    pub accepted: bool,
    pub reason: String,
    pub stages: Vec<Stage>,
    pub phase_column: Option<Vec<C>>,
    pub purity_error: Option<f64>,
    pub diagonal_residual: Option<f64>,
}

pub fn input_check(b: &[C], n: usize, rank: usize, rows: [usize; 2], o: Options) -> Result<u64> {
    let bound = memory_bound(n, rank)?;
    o.validate()?;
    if b.len() != n * rank
        || rows[0] >= n
        || rows[1] >= n
        || rows[0] == rows[1]
        || b.iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite() || z.norm() > 1e4)
    {
        return Err("invalid features or observable row pair".into());
    }
    Ok(bound)
}

pub(crate) fn stage(a: &Affine, c: &[f64], rank: usize, o: Options) -> Result<Stage> {
    let mut z = vec![0.; rank * rank];
    for value in &mut z[..rank] {
        *value = 1. / rank as f64;
    }
    let mut u = vec![0.; rank * rank];
    let mut out = Stage {
        accepted: false,
        reason: "iteration_cap".into(),
        affine_rank: a.rank,
        affine_consistency: a.consistency,
        matrix: matrix::unpack(&z, rank),
        affine_state: z.clone(),
        scaled_dual: u.clone(),
        multipliers: vec![0.; a.rows.len()],
        trace: vec![],
    };
    if a.consistency > o.feasibility_tolerance {
        out.reason = "inconsistent_affine_constraints".into();
        return Ok(out);
    }
    for iteration in 1..=o.iterations {
        let t: Vec<_> = z
            .iter()
            .zip(&u)
            .zip(c)
            .map(|((&z, &u), &c)| z - u + c / o.rho)
            .collect();
        let density = matrix::density(&matrix::unpack(&t, rank), rank)?;
        let x = matrix::pack(&density, rank);
        let p: Vec<_> = x.iter().zip(&u).map(|(x, u)| x + u).collect();
        let (next, lambda) = a.project(&p);
        for j in 0..u.len() {
            u[j] += x[j] - next[j];
        }
        let y: Vec<_> = lambda.iter().map(|l| o.rho * l).collect();
        let aty = a.transpose(&y);
        let slack: Vec<_> = c.iter().zip(aty).map(|(c, a)| c - a).collect();
        let max = matrix::eigen(&matrix::unpack(&slack, rank), rank)?.values[rank - 1];
        let objective = matrix::dot(c, &x);
        let bound = max + matrix::dot(&y, &a.targets);
        let step = Step {
            iteration,
            objective,
            dual_bound: bound,
            dual_gap: bound - objective,
            affine_residual: a.residual(&x),
            splitting_residual: matrix::distance(&x, &next),
            dual_step: o.rho * matrix::distance(&z, &next),
        };
        // A negative apparent gap can be caused by primal infeasibility. Require
        // its absolute magnitude, as well as feasibility and both ADMM residuals.
        out.accepted = step.affine_residual <= o.feasibility_tolerance
            && step.splitting_residual <= o.feasibility_tolerance
            && step.dual_step <= o.feasibility_tolerance
            && step.dual_gap.abs() <= o.gap_tolerance;
        z = next;
        out.matrix = density;
        out.affine_state.clone_from(&z);
        out.scaled_dual.clone_from(&u);
        out.multipliers = y;
        out.trace.push(step);
        if out.accepted {
            out.reason = "numerical_tolerances_met".into();
            break;
        }
    }
    Ok(out)
}

pub fn extract(b: &[C], n: usize, rank: usize, rows: [usize; 2], o: Options) -> Result<Extraction> {
    let bound = input_check(b, n, rank, rows, o)?;
    let center = (0..rank)
        .map(|j| (0..n).map(|i| b[i + n * j]).sum::<C>().norm() / n as f64)
        .fold(0., f64::max);
    let mut gram = 0f64;
    for j in 0..rank {
        for i in 0..rank {
            let v: C = (0..n).map(|x| b[x + n * i].conj() * b[x + n * j]).sum();
            gram = gram.max((v / n as f64 - C::new(f64::from(i == j), 0.)).norm());
        }
    }
    let mut out = Extraction {
        n,
        rank,
        rows,
        options: o,
        buffer_bound_bytes: bound,
        centered_residual: center,
        gram_residual: gram,
        accepted: false,
        reason: "first_stage_refused".into(),
        stages: vec![],
        phase_column: None,
        purity_error: None,
        diagonal_residual: None,
    };
    if center > o.feasibility_tolerance || gram > o.feasibility_tolerance {
        out.reason = "features_not_centered_and_normalized".into();
        return Ok(out);
    }
    let mut ar: Vec<_> = (0..n)
        .map(|x| matrix::objective(b, n, rank, x, x, false))
        .collect();
    let mut targets = vec![1.; n];
    let re = matrix::objective(b, n, rank, rows[0], rows[1], false);
    let im = matrix::objective(b, n, rank, rows[0], rows[1], true);
    let a = Affine::new(ar.clone(), targets.clone(), o.rank_cutoff)?;
    out.stages.push(stage(&a, &re, rank, o)?);
    if !out.stages[0].accepted {
        return Ok(out);
    }
    targets.push(
        out.stages[0]
            .trace
            .last()
            .expect("accepted stage has trace")
            .objective,
    );
    ar.push(re);
    let a = Affine::new(ar, targets, o.rank_cutoff)?;
    out.stages.push(stage(&a, &im, rank, o)?);
    if !out.stages[1].accepted {
        out.reason = "second_stage_refused".into();
        return Ok(out);
    }
    let m = &out.stages[1].matrix;
    let e = matrix::eigen(m, rank)?;
    let purity = (1. - e.values[rank - 1]).abs();
    let diagonal = (0..n)
        .map(|x| (matrix::reading(b, n, rank, m, x, x) - 1.).norm())
        .fold(0., f64::max);
    out.purity_error = Some(purity);
    out.diagonal_residual = Some(diagonal);
    if purity > o.purity_tolerance || diagonal > o.feasibility_tolerance {
        out.reason = "mixed_or_infeasible_endpoint".into();
        return Ok(out);
    }
    out.phase_column = Some(
        (0..n)
            .map(|x| matrix::reading(b, n, rank, m, x, rows[0]))
            .collect(),
    );
    out.accepted = true;
    out.reason = "numerically_pure_feasible_endpoint".into();
    Ok(out)
}

/// Replay from the same observable inputs, including every stage and diagnostic.
pub fn replay(b: &[C], saved: &Extraction) -> Result<()> {
    let actual = extract(b, saved.n, saved.rank, saved.rows, saved.options)?;
    if actual != *saved {
        return Err("density extraction replay mismatch".into());
    }
    Ok(())
}
