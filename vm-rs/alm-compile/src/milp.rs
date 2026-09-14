//! The scheduling problem itself: the mixed-integer program `milp.py` hands to
//! a solver, built as plain data so the formulation can be checked, exported,
//! and solved by whatever is available.
//!
//! The unknowns are a layer `k` per gate and, for a persist, a bit `z` saying
//! whether it runs before or after the FFN.  Everything else — when a
//! dimension is last read, whether it is alive at a boundary, whether it has
//! to occupy a slot there — follows from those, and is pinned by indicator
//! constraints so the solver cannot cheat.  The objective is `D_half`, half
//! the residual width, bounded below by the occupancy of every boundary and by
//! the head count of every attention block.

use crate::expr::DimId;
use crate::interpreter::MachineGraph;
use crate::scheduler::{output_dims, OpKind, Phase, SchedGraph};
use std::collections::HashMap;
use std::fmt::Write as _;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VarKind {
    Integer,
    Binary,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cmp {
    Le,
    Ge,
    Eq,
}

pub struct Var {
    pub name: String,
    pub lo: f64,
    pub hi: Option<f64>,
    pub kind: VarKind,
}

pub struct Row {
    pub terms: Vec<(usize, f64)>,
    pub op: Cmp,
    pub rhs: f64,
}

/// A linear form: `sum(coeff * var) + constant`.
#[derive(Clone, Default)]
pub struct Lin {
    pub terms: Vec<(usize, f64)>,
    pub constant: f64,
}

impl Lin {
    pub fn constant(c: f64) -> Lin {
        Lin { terms: Vec::new(), constant: c }
    }
    pub fn var(i: usize) -> Lin {
        Lin { terms: vec![(i, 1.0)], constant: 0.0 }
    }
    pub fn term(mut self, i: usize, coeff: f64) -> Lin {
        self.terms.push((i, coeff));
        self
    }
    pub fn plus(mut self, other: &Lin) -> Lin {
        self.terms.extend_from_slice(&other.terms);
        self.constant += other.constant;
        self
    }
    pub fn minus(mut self, other: &Lin) -> Lin {
        self.terms.extend(other.terms.iter().map(|&(i, c)| (i, -c)));
        self.constant -= other.constant;
        self
    }
    pub fn shift(mut self, c: f64) -> Lin {
        self.constant += c;
        self
    }
    pub fn scale(mut self, k: f64) -> Lin {
        for t in self.terms.iter_mut() {
            t.1 *= k;
        }
        self.constant *= k;
        self
    }
}

/// The program, and the handles needed to read a solution back as a schedule.
pub struct Milp {
    pub vars: Vec<Var>,
    pub rows: Vec<Row>,
    /// The single objective variable, minimized: half the residual width.
    pub d_half: usize,
    /// The layer of each op, indexed as `SchedGraph::ops`.
    pub k: Vec<usize>,
    /// The after-FFN bit of each persist op.
    pub z: HashMap<usize, usize>,
    /// A full assignment satisfying every row, when one was supplied.
    pub witness: Option<Vec<f64>>,
}

impl Milp {
    /// The phase assignment a solution stands for.
    pub fn schedule(&self, sg: &SchedGraph, values: &[f64]) -> Vec<Phase> {
        (0..sg.ops.len())
            .map(|i| {
                let layer = values[self.k[i]].round() as Phase;
                let second = self.z.get(&i).is_some_and(|&v| values[v].round() as i64 == 1);
                sg.phase_in(i, layer, second)
            })
            .collect()
    }

    /// Rows an assignment breaks, as `(index, slack)` — empty when it is
    /// feasible.  The tolerance is the one an integer program deserves: the
    /// coefficients are small integers, so a real violation is at least one.
    pub fn violations(&self, values: &[f64]) -> Vec<(usize, f64)> {
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                let lhs = r.terms.iter().map(|&(v, c)| c * values[v]).sum::<f64>();
                let slack = match r.op {
                    Cmp::Le => r.rhs - lhs,
                    Cmp::Ge => lhs - r.rhs,
                    Cmp::Eq => -(lhs - r.rhs).abs(),
                };
                (slack < -1e-6).then_some((i, slack))
            })
            .collect()
    }

    /// The program in CPLEX LP format, which every solver worth using reads.
    pub fn to_lp(&self) -> String {
        let mut s = String::from("Minimize\n obj: ");
        let _ = writeln!(s, "{}", self.vars[self.d_half].name);
        s.push_str("Subject To\n");
        for (i, r) in self.rows.iter().enumerate() {
            let _ = write!(s, " c{i}:");
            for &(v, c) in &r.terms {
                let _ = write!(s, " {} {}", if c < 0.0 { "-" } else { "+" }, c.abs());
                let _ = write!(s, " {}", self.vars[v].name);
            }
            let op = match r.op {
                Cmp::Le => "<=",
                Cmp::Ge => ">=",
                Cmp::Eq => "=",
            };
            let _ = writeln!(s, " {op} {}", r.rhs);
        }
        s.push_str("Bounds\n");
        for v in &self.vars {
            match v.hi {
                Some(hi) => {
                    let _ = writeln!(s, " {} <= {} <= {hi}", v.lo, v.name);
                }
                None => {
                    let _ = writeln!(s, " {} <= {} <= +inf", v.lo, v.name);
                }
            }
        }
        s.push_str("Generals\n");
        for v in self.vars.iter().filter(|v| v.kind == VarKind::Integer) {
            let _ = writeln!(s, " {}", v.name);
        }
        s.push_str("Binaries\n");
        for v in self.vars.iter().filter(|v| v.kind == VarKind::Binary) {
            let _ = writeln!(s, " {}", v.name);
        }
        s.push_str("End\n");
        s
    }
}

/// Builds the program, carrying an optional witness assignment alongside it so
/// every variable is given the value the supplied schedule implies.
struct Builder<'a> {
    vars: Vec<Var>,
    rows: Vec<Row>,
    values: Vec<f64>,
    has_witness: bool,
    pa: Vec<Phase>,
    death: HashMap<DimId, Phase>,
    sg: &'a SchedGraph,
    p: Phase,
}

impl Builder<'_> {
    fn var(&mut self, name: String, lo: f64, hi: Option<f64>, kind: VarKind, value: f64) -> usize {
        self.vars.push(Var { name, lo, hi, kind });
        self.values.push(value);
        self.vars.len() - 1
    }

    fn bin(&mut self, name: String, value: bool) -> usize {
        self.var(name, 0.0, Some(1.0), VarKind::Binary, f64::from(value))
    }

    /// `lhs op rhs`, stored with the constants moved to the right.
    fn row(&mut self, lhs: Lin, op: Cmp, rhs: Lin) {
        let combined = lhs.minus(&rhs);
        self.rows.push(Row { terms: combined.terms, op, rhs: -combined.constant });
    }

    /// The phase of an op, as a linear form in its layer and persist bit.
    fn phase_of(&self, op: usize, k: &[usize], z: &HashMap<usize, usize>) -> Lin {
        match self.sg.kind(op) {
            OpKind::Lookup => Lin::var(k[op]).scale(4.0),
            OpKind::ReGlu => Lin::var(k[op]).scale(4.0).shift(2.0),
            OpKind::Persist => Lin::var(k[op]).scale(4.0).shift(1.0).term(z[&op], 2.0),
        }
    }

    /// The witness phase of an op; zero when no witness was supplied, in which
    /// case the values are never looked at.
    fn w_phase(&self, op: usize) -> Phase {
        self.pa.get(op).copied().unwrap_or(0)
    }

    fn w_death(&self, d: DimId) -> Phase {
        self.death.get(&d).copied().unwrap_or(0)
    }

    /// `v = a AND b`, as the three inequalities a solver needs.
    fn and2(&mut self, name: String, a: usize, b: usize) -> usize {
        let v = self.bin(name, self.values[a] > 0.5 && self.values[b] > 0.5);
        self.row(Lin::var(v), Cmp::Le, Lin::var(a));
        self.row(Lin::var(v), Cmp::Le, Lin::var(b));
        self.row(Lin::var(v), Cmp::Ge, Lin::var(a).term(b, 1.0).shift(-1.0));
        v
    }

    /// `v = [expr <= c]`, for an expression bounded by the phase count.
    fn is_le(&mut self, name: String, expr: &Lin, c: Phase, witness: bool) -> usize {
        let p = self.p as f64;
        let v = self.bin(name, witness);
        // expr <= c + P(1 - v)
        self.row(expr.clone(), Cmp::Le, Lin::constant(c as f64 + p).term(v, -p));
        // expr >= c + 1 - P v
        self.row(expr.clone(), Cmp::Ge, Lin::constant((c + 1) as f64).term(v, -p));
        v
    }

    /// `v = [expr >= c]`, likewise.
    fn is_ge(&mut self, name: String, expr: &Lin, c: Phase, witness: bool) -> usize {
        let p = self.p as f64;
        let v = self.bin(name, witness);
        // expr >= c - P(1 - v)
        self.row(expr.clone(), Cmp::Ge, Lin::constant(c as f64 - p).term(v, p));
        // expr <= c - 1 + P v
        self.row(expr.clone(), Cmp::Le, Lin::constant((c - 1) as f64).term(v, p));
        v
    }
}

/// Build the scheduling program over `n_layers` layers.
///
/// `witness` is an existing schedule: when given, every variable is assigned
/// the value that schedule implies, so the formulation can be checked against
/// a schedule already known to be good instead of only against a solver.
pub fn build(
    mg: &MachineGraph,
    sg: &SchedGraph,
    n_layers: usize,
    max_ffn: Option<usize>,
    witness: Option<&[Phase]>,
) -> Milp {
    let g = &mg.graph;
    let outs = output_dims(mg);
    let protected = [g.position, g.inv_log_pos, g.position_sq];
    let n = n_layers as Phase;
    let p = 4 * n;

    // The witness deaths: as early as every consumer allows.
    let mut w_death: HashMap<DimId, Phase> = HashMap::new();
    if let Some(pa) = witness {
        for &d in &sg.all_dims {
            if outs.contains(&d) {
                continue;
            }
            let cons = sg.consumers.get(&d).map(|v| v.as_slice()).unwrap_or(&[]);
            if cons.is_empty() && d != g.position {
                continue;
            }
            w_death.insert(d, cons.iter().map(|&i| pa[i]).max().unwrap_or(0));
        }
        if let Some(&dp) = w_death.get(&g.position) {
            let latest = sg.persists().map(|i| pa[i]).filter(|q| q % 4 == 1).max();
            w_death.insert(g.position, dp.max(latest.unwrap_or(dp)));
        }
    }

    let mut b = Builder {
        vars: Vec::new(),
        rows: Vec::new(),
        values: Vec::new(),
        has_witness: witness.is_some(),
        pa: witness.map(|w| w.to_vec()).unwrap_or_default(),
        death: w_death,
        sg,
        p,
    };

    let d_half_value = 0.0;
    let d_half = b.var("D_half".into(), 0.0, None, VarKind::Integer, d_half_value);

    let k: Vec<usize> = (0..sg.ops.len())
        .map(|i| {
            let v = (b.w_phase(i) / 4) as f64;
            b.var(format!("k_{i}"), 0.0, Some((n - 1) as f64), VarKind::Integer, v)
        })
        .collect();
    let z: HashMap<usize, usize> = sg
        .persists()
        .map(|i| {
            let v = b.w_phase(i) % 4 == 3;
            (i, b.bin(format!("z_{i}"), v))
        })
        .collect();

    // ── Precedence, and the ties that must not be split ──────────
    for i in 0..sg.ops.len() {
        let here = b.phase_of(i, &k, &z);
        for &dep in &sg.op_deps[i] {
            let there = b.phase_of(dep, &k, &z);
            b.row(here.clone(), Cmp::Ge, there.shift(1.0));
        }
    }
    for i in 0..sg.ops.len() {
        for &lu in &sg.tight_to[i] {
            b.row(Lin::var(k[i]), Cmp::Eq, Lin::var(k[lu]));
        }
    }

    // ── When each dimension is last read ─────────────────────────
    let mut death: HashMap<DimId, usize> = HashMap::new();
    for &d in &sg.all_dims {
        if outs.contains(&d) {
            continue;
        }
        let cons: Vec<usize> = sg.consumers.get(&d).cloned().unwrap_or_default();
        if cons.is_empty() && d != g.position {
            continue;
        }
        let v = b.w_death(d) as f64;
        let dv = b.var(format!("death_{d}"), 0.0, Some((p - 1) as f64), VarKind::Integer, v);
        for c_op in cons {
            let ph = b.phase_of(c_op, &k, &z);
            b.row(Lin::var(dv), Cmp::Ge, ph);
        }
        death.insert(d, dv);
    }
    // Position keys the passthrough of every persist1, so it must outlive them.
    if let Some(&dv) = death.get(&g.position) {
        for i in sg.persists() {
            let ph = b.phase_of(i, &k, &z).term(z[&i], -(p as f64));
            b.row(Lin::var(dv), Cmp::Ge, ph);
        }
    }

    // ── At most `max_ffn` ReGLUs in a layer ──────────────────────
    if let Some(max_ffn) = max_ffn {
        let mut by_layer: Vec<Vec<usize>> = vec![Vec::new(); n as usize];
        for i in sg.reglus() {
            let mut sum = Lin::default();
            let mut pick = Lin::default();
            for l in 0..n {
                let v = b.bin(format!("fb_{i}_{l}"), b.w_phase(i) / 4 == l);
                by_layer[l as usize].push(v);
                sum = sum.term(v, 1.0);
                pick = pick.term(v, l as f64);
            }
            b.row(sum, Cmp::Eq, Lin::constant(1.0));
            b.row(Lin::var(k[i]), Cmp::Eq, pick);
        }
        for vs in by_layer {
            let mut sum = Lin::default();
            for v in vs {
                sum = sum.term(v, 1.0);
            }
            b.row(sum, Cmp::Le, Lin::constant(max_ffn as f64));
        }
    }

    // ── Which layer each lookup and each persist lands in ────────
    let mut lu_at: HashMap<(usize, Phase), usize> = HashMap::new();
    for i in sg.lookups() {
        let (mut sum, mut pick) = (Lin::default(), Lin::default());
        for l in 0..n {
            let v = b.bin(format!("la_{i}_{l}"), b.w_phase(i) / 4 == l);
            lu_at.insert((i, l), v);
            sum = sum.term(v, 1.0);
            pick = pick.term(v, l as f64);
        }
        b.row(sum, Cmp::Eq, Lin::constant(1.0));
        b.row(Lin::var(k[i]), Cmp::Eq, pick);
    }

    let mut p1_at: HashMap<(usize, Phase), usize> = HashMap::new();
    for i in sg.persists() {
        let (mut sum, mut pick) = (Lin::default(), Lin::default());
        let mut here: Vec<(Phase, usize)> = Vec::new();
        for l in 0..n {
            let v = b.bin(format!("pl_{i}_{l}"), b.w_phase(i) / 4 == l);
            here.push((l, v));
            sum = sum.term(v, 1.0);
            pick = pick.term(v, l as f64);
        }
        b.row(sum, Cmp::Eq, Lin::constant(1.0));
        b.row(Lin::var(k[i]), Cmp::Eq, pick);
        for (l, pl) in here {
            let v = b.bin(format!("p1_{i}_{l}"), b.w_phase(i) == 4 * l + 1);
            b.row(Lin::var(v), Cmp::Le, Lin::var(pl));
            b.row(Lin::var(v), Cmp::Le, Lin::constant(1.0).term(z[&i], -1.0));
            b.row(Lin::var(v), Cmp::Ge, Lin::var(pl).term(z[&i], -1.0));
            p1_at.insert((i, l), v);
        }
    }

    // ── Passthrough: what a persist1 reads but its layer does not make ──
    let mut p1_deps: HashMap<DimId, Vec<usize>> = HashMap::new();
    for i in sg.persists() {
        for &d in &sg.deps[i] {
            p1_deps.entry(d).or_default().push(i);
        }
    }
    let mut pd_var: HashMap<(DimId, Phase), usize> = HashMap::new();
    let mut pd_dims: Vec<DimId> = p1_deps.keys().copied().collect();
    pd_dims.sort_unstable();
    for d in pd_dims {
        let ps = p1_deps[&d].clone();
        let lu_of = g.dim(d).is_lookup().then(|| sg.dim_to_op[&d]);
        for l in 0..n {
            let made_here = lu_of.is_some_and(|lu| b.w_phase(lu) == 4 * l);
            let kept = ps.iter().any(|&q| b.w_phase(q) == 4 * l + 1);
            let v = b.bin(format!("pd_{d}_{l}"), kept && !made_here);
            let mut sum = Lin::default();
            for &q in &ps {
                sum = sum.term(p1_at[&(q, l)], 1.0);
            }
            b.row(Lin::var(v), Cmp::Le, sum);
            match lu_of.and_then(|lu| lu_at.get(&(lu, l)).copied()) {
                Some(la) => {
                    b.row(Lin::var(v), Cmp::Le, Lin::constant(1.0).term(la, -1.0));
                    for &q in &ps {
                        b.row(Lin::var(v), Cmp::Ge, Lin::var(p1_at[&(q, l)]).term(la, -1.0));
                    }
                }
                None => {
                    for &q in &ps {
                        b.row(Lin::var(v), Cmp::Ge, Lin::var(p1_at[&(q, l)]));
                    }
                }
            }
            pd_var.insert((d, l), v);
        }
    }

    // ── A dimension read within its own half-layer takes no slot ──
    let mut ns: HashMap<DimId, usize> = HashMap::new();
    for &d in &sg.all_dims {
        if outs.contains(&d) || !death.contains_key(&d) {
            continue;
        }
        let Some(&prod) = sg.dim_to_op.get(&d) else { continue };
        if !(g.dim(d).is_lookup() || g.dim(d).is_reglu()) {
            continue;
        }
        let live = b.w_death(d) >= b.w_phase(prod) + 2;
        let v = b.bin(format!("ns_{d}"), live);
        let ph = b.phase_of(prod, &k, &z);
        let dv = Lin::var(death[&d]);
        // death >= phase + 2 - P(1 - ns)
        b.row(dv.clone(), Cmp::Ge, ph.clone().shift(2.0 - p as f64).term(v, p as f64));
        // death <= phase + 1 + P ns
        b.row(dv, Cmp::Le, ph.shift(1.0).term(v, p as f64));
        ns.insert(d, v);
    }

    // ── Occupancy at every odd boundary ──────────────────────────
    let mut alive_sum: HashMap<Phase, Lin> = HashMap::new();
    for c in (1..p).step_by(2) {
        let mut ew = Lin::default();
        let mut alive = Lin::default();
        for &d in &sg.all_dims {
            let prod = sg.dim_to_op.get(&d).copied();
            let is_input = g.dim(d).is_input();
            if outs.contains(&d) || protected.contains(&d) {
                if is_input {
                    ew = ew.shift(1.0);
                    alive = alive.shift(1.0);
                } else if let Some(prod) = prod {
                    let ph = b.phase_of(prod, &k, &z);
                    let bb = b.is_le(format!("b_{d}_{c}"), &ph, c, b.w_phase(prod) <= c);
                    ew = ew.term(bb, 1.0);
                    alive = alive.term(bb, 1.0);
                }
                continue;
            }
            let Some(&dv) = death.get(&d) else { continue };
            let dl = Lin::var(dv);
            if is_input {
                let eu = b.is_ge(format!("ew_{d}_{c}"), &dl, c - 1, b.w_death(d) >= c - 1);
                ew = ew.term(eu, 1.0);
                let au = b.is_ge(format!("a_{d}_{c}"), &dl, c + 1, b.w_death(d) > c);
                alive = alive.term(au, 1.0);
            } else if let Some(prod) = prod {
                let ph = b.phase_of(prod, &k, &z);
                let bb = b.is_le(format!("b_{d}_{c}"), &ph, c, b.w_phase(prod) <= c);
                let eu = b.is_ge(format!("eu_{d}_{c}"), &dl, c - 1, b.w_death(d) >= c - 1);
                let ev = match ns.get(&d) {
                    Some(&slot) => {
                        let both = b.and2(format!("ew0_{d}_{c}"), bb, eu);
                        b.and2(format!("ew_{d}_{c}"), both, slot)
                    }
                    None => b.and2(format!("ew_{d}_{c}"), bb, eu),
                };
                ew = ew.term(ev, 1.0);
                let au = b.is_ge(format!("au_{d}_{c}"), &dl, c + 1, b.w_death(d) > c);
                let av = b.and2(format!("a_{d}_{c}"), bb, au);
                alive = alive.term(av, 1.0);
            }
        }
        b.row(Lin::var(d_half).scale(2.0), Cmp::Ge, ew);
        alive_sum.insert(c, alive);
    }

    // ── Heads: one per lookup pair, half per dimension erased or carried ──
    let n_inputs = sg.inputs.len() as f64;
    for l in 0..n {
        let mut rhs = Lin::default();
        for i in sg.lookups() {
            let la = lu_at[&(i, l)];
            let lu = match sg.ops[i] {
                crate::scheduler::Op::Lookup(id) => g.lookup(id),
                crate::scheduler::Op::Dim(_) => unreachable!(),
            };
            let heads = lu.value_exprs.len().div_ceil(2);
            rhs = rhs.term(la, 2.0 * heads as f64).term(la, sg.produced[i].len() as f64);
        }
        for i in sg.persists() {
            rhs = rhs.term(p1_at[&(i, l)], 1.0);
        }
        for (&(_, at), &v) in pd_var.iter() {
            if at == l {
                rhs = rhs.term(v, 1.0);
            }
        }
        let prev = match alive_sum.get(&(4 * l - 1)) {
            Some(a) => a.clone(),
            None => Lin::constant(n_inputs),
        };
        rhs = rhs.plus(&prev).minus(&alive_sum[&(4 * l + 1)]);
        b.row(Lin::var(d_half).scale(2.0), Cmp::Ge, rhs);
    }

    // The objective is the only variable with no upper bound; give the witness
    // the smallest value every row it appears in allows.
    if b.has_witness {
        let needed = b
            .rows
            .iter()
            .filter(|r| r.op == Cmp::Ge && r.terms.iter().any(|&(v, _)| v == d_half))
            .map(|r| {
                let rest: f64 = r
                    .terms
                    .iter()
                    .filter(|&&(v, _)| v != d_half)
                    .map(|&(v, c)| c * b.values[v])
                    .sum();
                let coeff: f64 = r.terms.iter().filter(|&&(v, _)| v == d_half).map(|t| t.1).sum();
                (r.rhs - rest) / coeff
            })
            .fold(0.0f64, f64::max);
        b.values[d_half] = needed.ceil();
    }

    Milp { vars: b.vars, rows: b.rows, d_half, k, z, witness: b.has_witness.then_some(b.values) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{min_layers, phases_from_plan, SchedGraph};

    /// The schedule the release ships satisfies every row of the program as
    /// ported, and needs exactly the half-width the release reports.  That is
    /// the check a formulation can be given without a solver: a known-good
    /// point must be feasible, and must cost what it is said to cost.
    #[test]
    fn the_released_schedule_is_a_feasible_point_of_this_program() {
        let mg = crate::interpreter::build();
        let sg = SchedGraph::build(&mg.graph);
        let plan = crate::plan::Plan::load(crate::release::PLAN, &mg.graph).unwrap();
        let pa = phases_from_plan(&plan, &sg).unwrap();

        let milp = build(&mg, &sg, min_layers(&sg), None, Some(&pa));
        let values = milp.witness.clone().unwrap();
        let broken = milp.violations(&values);
        assert!(
            broken.is_empty(),
            "{} rows broken, first is {:?}",
            broken.len(),
            broken.first().map(|&(i, s)| (
                milp.rows[i]
                    .terms
                    .iter()
                    .map(|&(v, c)| (milp.vars[v].name.clone(), c))
                    .collect::<Vec<_>>(),
                milp.rows[i].op,
                milp.rows[i].rhs,
                s
            ))
        );
        assert_eq!(values[milp.d_half], 19.0, "the released plan is d_model 38");
        assert_eq!(milp.schedule(&sg, &values), pa, "the solution reads back as the schedule");
    }

    /// A formulation is only as strong as its tighter direction.  Every bit in
    /// the program stands for a fact about the schedule — this dimension is
    /// born by here, that one is still read after there — and the schedule has
    /// to *force* it, not merely permit it.  A bit the schedule leaves free is
    /// a bit the solver sets to whatever is cheapest, and the objective stops
    /// being the width: it was a big-M added on the wrong side of one
    /// indicator that let HiGHS prove a residual stream of 26 optimal for a
    /// schedule that needs 40.  So: around the released witness, flipping any
    /// single bit must break a row.
    #[test]
    fn no_bit_of_the_program_is_free_to_flip() {
        let mg = crate::interpreter::build();
        let sg = SchedGraph::build(&mg.graph);
        let plan = crate::plan::Plan::load(crate::release::PLAN, &mg.graph).unwrap();
        let pa = phases_from_plan(&plan, &sg).unwrap();
        let milp = build(&mg, &sg, min_layers(&sg), None, Some(&pa));

        let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); milp.vars.len()];
        for (i, r) in milp.rows.iter().enumerate() {
            for &(v, _) in &r.terms {
                rows_of[v].push(i);
            }
        }
        let holds = |i: usize, vals: &[f64]| {
            let r = &milp.rows[i];
            let lhs: f64 = r.terms.iter().map(|&(v, c)| c * vals[v]).sum();
            match r.op {
                Cmp::Le => lhs <= r.rhs + 1e-6,
                Cmp::Ge => lhs >= r.rhs - 1e-6,
                Cmp::Eq => (lhs - r.rhs).abs() <= 1e-6,
            }
        };

        let mut values = milp.witness.clone().unwrap();
        let free: Vec<&str> = (0..milp.vars.len())
            .filter(|&v| milp.vars[v].kind == VarKind::Binary)
            .filter(|&v| {
                values[v] = 1.0 - values[v];
                let pinned = rows_of[v].iter().any(|&i| !holds(i, &values));
                values[v] = 1.0 - values[v];
                !pinned
            })
            .map(|v| milp.vars[v].name.as_str())
            .collect();
        assert!(
            free.is_empty(),
            "{} of {} bits are free, first are {:?}",
            free.len(),
            milp.vars.len(),
            &free[..free.len().min(8)]
        );
    }

    /// Sizes, so a change to the graph or the formulation that quietly blows
    /// the program up is visible rather than merely slow.
    #[test]
    fn the_program_is_fourteen_thousand_variables_over_a_hundred_and_ninety_gates() {
        let mg = crate::interpreter::build();
        let sg = SchedGraph::build(&mg.graph);
        let milp = build(&mg, &sg, min_layers(&sg), None, None);
        assert_eq!((sg.reglus().len(), sg.persists().len(), sg.lookups().len()), (105, 33, 21));
        assert_eq!(sg.all_dims.len(), 186);
        let binaries = milp.vars.iter().filter(|v| v.kind == VarKind::Binary).count();
        assert_eq!((binaries, milp.vars.len(), milp.rows.len()), (14_070, 14_392, 34_498));
    }
}
