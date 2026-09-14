//! The schedule: which phase every gate runs in, and what that costs.
//!
//! A port of `scheduler/milp.py` without the solve — the dependency graph it
//! builds, the lifetimes a phase assignment implies, the width objective the
//! assignment is scored by, and the `plan.yaml` that records it.  Fed the
//! released schedule, the four together reproduce the released plan, which is
//! what keeps the port honest.
//!
//! Phases run in fours: `4L` attention, `4L+1` persist1, `4L+2` FFN,
//! `4L+3` persist2.  A dimension is born at the phase of the gate producing it
//! and dies at the last phase that reads it; `d_model` is twice the worst
//! half-layer occupancy over every odd boundary.

use crate::expr::{DimId, Expr};
use crate::graph::{DimData, Graph, LookUpId, TieBreak};
use crate::interpreter::MachineGraph;
use std::collections::{BTreeMap, HashMap, HashSet};

/// A phase index; `-1` is the embedding, before any layer has run.
pub type Phase = i64;

/// A gate the scheduler places.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Op {
    /// An attention lookup, producing all of its value channels at once.
    Lookup(LookUpId),
    /// The single dimension a ReGLU or a persist produces.
    Dim(DimId),
}

/// Which of the three phase families an op belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpKind {
    ReGlu,
    Persist,
    Lookup,
}

/// The dependency graph the scheduler works over.
///
/// `ops` is ReGLUs, then persists, then lookups — the Python order, which the
/// per-layer lists of the plan file inherit.
pub struct SchedGraph {
    pub ops: Vec<Op>,
    n_reglus: usize,
    n_persists: usize,
    /// Dimensions read straight off the embedding.
    pub inputs: Vec<DimId>,
    /// What each op produces, in the order the plan lists it.
    pub produced: Vec<Vec<DimId>>,
    /// The dimensions each op reads.
    pub deps: Vec<Vec<DimId>>,
    /// The op producing a dimension; inputs are absent.
    pub dim_to_op: HashMap<DimId, usize>,
    /// The ops each op must follow.
    pub op_deps: Vec<Vec<usize>>,
    /// The ops reading each dimension.
    pub consumers: HashMap<DimId, Vec<usize>>,
    /// Average-tie-break lookups an op must share a layer with: their value is
    /// only well defined while the whole tie is still in the residual stream.
    pub tight_to: Vec<Vec<usize>>,
    /// Inputs first, then every produced dimension in op order.
    pub all_dims: Vec<DimId>,
}

impl SchedGraph {
    pub fn build(g: &Graph) -> SchedGraph {
        let reglus: Vec<DimId> = g.dims.iter().filter(|d| d.is_reglu()).map(|d| d.id).collect();
        let persists: Vec<DimId> = g.dims.iter().filter(|d| d.is_persist()).map(|d| d.id).collect();
        let inputs: Vec<DimId> = g.dims.iter().filter(|d| d.is_input()).map(|d| d.id).collect();

        let n_reglus = reglus.len();
        let n_persists = persists.len();
        let mut ops: Vec<Op> = Vec::new();
        ops.extend(reglus.iter().map(|&d| Op::Dim(d)));
        ops.extend(persists.iter().map(|&d| Op::Dim(d)));
        ops.extend(g.lookups.iter().map(|l| Op::Lookup(l.id)));

        let terms_of = |e: &Expr, out: &mut Vec<DimId>| out.extend(e.terms().iter().map(|t| t.0));
        let mut produced: Vec<Vec<DimId>> = Vec::with_capacity(ops.len());
        let mut deps: Vec<Vec<DimId>> = Vec::with_capacity(ops.len());
        for &op in &ops {
            let mut p = Vec::new();
            let mut d = Vec::new();
            match op {
                Op::Dim(dim) => {
                    p.push(dim);
                    match &g.dim(dim).data {
                        DimData::ReGlu { a, b } => {
                            terms_of(a, &mut d);
                            terms_of(b, &mut d);
                        }
                        DimData::Persist { expr } => terms_of(expr, &mut d),
                        other => panic!("{other:?} is not a schedulable op"),
                    }
                }
                Op::Lookup(l) => {
                    let lu = g.lookup(l);
                    p.extend_from_slice(&lu.dims);
                    for e in lu.query_2d.iter().chain(lu.key_2d.iter()).chain(lu.value_exprs.iter())
                    {
                        terms_of(e, &mut d);
                    }
                    d.push(g.inv_log_pos);
                }
            }
            d.sort_unstable();
            d.dedup();
            produced.push(p);
            deps.push(d);
        }

        let mut dim_to_op: HashMap<DimId, usize> = HashMap::new();
        for (i, dims) in produced.iter().enumerate() {
            for &d in dims {
                dim_to_op.insert(d, i);
            }
        }

        let mut op_deps: Vec<Vec<usize>> = vec![Vec::new(); ops.len()];
        let mut consumers: HashMap<DimId, Vec<usize>> = HashMap::new();
        for (i, dims) in deps.iter().enumerate() {
            for &d in dims {
                consumers.entry(d).or_default().push(i);
                match dim_to_op.get(&d) {
                    Some(&pred) if pred != i => op_deps[i].push(pred),
                    _ => {}
                }
            }
            op_deps[i].sort_unstable();
            op_deps[i].dedup();
        }

        let mut tight_to: Vec<Vec<usize>> = vec![Vec::new(); ops.len()];
        for i in 0..n_reglus + n_persists {
            for &d in &deps[i] {
                if !g.dim(d).is_lookup() {
                    continue;
                }
                if let Some(&lu_op) = dim_to_op.get(&d) {
                    if let Op::Lookup(l) = ops[lu_op] {
                        if matches!(g.lookup(l).tie_break, TieBreak::Average) {
                            tight_to[i].push(lu_op);
                        }
                    }
                }
            }
            tight_to[i].sort_unstable();
            tight_to[i].dedup();
        }

        let mut all_dims = inputs.clone();
        let mut seen: HashSet<DimId> = all_dims.iter().copied().collect();
        for dims in &produced {
            for &d in dims {
                if seen.insert(d) {
                    all_dims.push(d);
                }
            }
        }

        SchedGraph {
            ops,
            n_reglus,
            n_persists,
            inputs,
            produced,
            deps,
            dim_to_op,
            op_deps,
            consumers,
            tight_to,
            all_dims,
        }
    }

    pub fn kind(&self, op: usize) -> OpKind {
        if op < self.n_reglus {
            OpKind::ReGlu
        } else if op < self.n_reglus + self.n_persists {
            OpKind::Persist
        } else {
            OpKind::Lookup
        }
    }

    /// Op indices of the ReGLUs, the persists and the lookups.
    pub fn reglus(&self) -> std::ops::Range<usize> {
        0..self.n_reglus
    }
    pub fn persists(&self) -> std::ops::Range<usize> {
        self.n_reglus..self.n_reglus + self.n_persists
    }
    pub fn lookups(&self) -> std::ops::Range<usize> {
        self.n_reglus + self.n_persists..self.ops.len()
    }

    /// The phase an op lands in, given its layer.
    pub fn phase_in(&self, op: usize, layer: Phase, second_persist: bool) -> Phase {
        match self.kind(op) {
            OpKind::Lookup => 4 * layer,
            OpKind::ReGlu => 4 * layer + 2,
            OpKind::Persist => 4 * layer + 1 + 2 * Phase::from(second_persist),
        }
    }
}

/// Critical-path length in layers, respecting phase parity: the fewest layers
/// any schedule could use, and the layer count the MILP is given.
pub fn min_layers(sg: &SchedGraph) -> usize {
    let n = sg.ops.len();
    let mut phase: Vec<Option<Phase>> = vec![None; n];
    let mut left = n;
    while left > 0 {
        let mut progress = false;
        for i in 0..n {
            if phase[i].is_some() || !sg.op_deps[i].iter().all(|&p| phase[p].is_some()) {
                continue;
            }
            let mut lo = sg.op_deps[i].iter().filter_map(|&p| phase[p]).max().unwrap_or(-1) + 1;
            lo += match sg.kind(i) {
                OpKind::Lookup => (-lo).rem_euclid(4),
                OpKind::ReGlu => (2 - lo).rem_euclid(4),
                OpKind::Persist => Phase::from(lo % 2 == 0),
            };
            phase[i] = Some(lo);
            left -= 1;
            progress = true;
        }
        assert!(progress, "cycle in the dependency graph");
    }
    (phase.iter().flatten().max().copied().unwrap_or(0) / 4 + 1) as usize
}

// ── Lifetimes and widths ────────────────────────────────────────────

/// Everything a phase assignment implies, and the report the plan file carries.
pub struct Analysis {
    pub num_layers: usize,
    pub max_phase: Phase,
    /// Per layer, op indices: the lookups, persist1, the ReGLUs, persist2.
    pub layers: Vec<[Vec<usize>; 4]>,
    pub dim_birth: HashMap<DimId, Phase>,
    pub dim_death: HashMap<DimId, Phase>,
    /// The dimensions still live after each odd boundary.
    pub alive_after: BTreeMap<Phase, Vec<DimId>>,
    pub lin_widths: BTreeMap<Phase, usize>,
    pub exp_lin_widths: BTreeMap<Phase, usize>,
    pub max_dep: usize,
    pub max_lin: usize,
    pub max_exp_lin: usize,
    pub d_model: usize,
}

/// The dimensions the output head scores with: they must survive to the end.
pub fn output_dims(mg: &MachineGraph) -> HashSet<DimId> {
    let mut out = HashSet::new();
    for (_, e) in &mg.output_tokens {
        out.extend(e.terms().iter().map(|t| t.0));
    }
    out
}

/// Rank of a set of row vectors, by elimination against a running basis.
///
/// `milp.py` hands `numpy.linalg.matrix_rank` the rows already scaled to unit
/// norm, so a residual far below one is numerical zero and no absolute
/// tolerance has to be guessed.
fn rank(rows: impl Iterator<Item = Vec<f64>>) -> usize {
    const TOL: f64 = 1e-9;
    let mut basis: Vec<(usize, Vec<f64>)> = Vec::new();
    for mut r in rows {
        let norm = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        for x in r.iter_mut() {
            *x /= norm;
        }
        for (p, b) in &basis {
            let f = r[*p];
            if f != 0.0 {
                for (x, y) in r.iter_mut().zip(b.iter()) {
                    *x -= f * y;
                }
            }
        }
        let (pivot, size) = r.iter().enumerate().fold((0usize, 0.0f64), |best, (i, x)| {
            if x.abs() > best.1 {
                (i, x.abs())
            } else {
                best
            }
        });
        if size < TOL {
            continue;
        }
        let lead = r[pivot];
        for x in r.iter_mut() {
            *x /= lead;
        }
        basis.push((pivot, r));
    }
    basis.len()
}

/// The expression vectors an op is defined by, in a fixed dimension basis.
fn op_vectors(g: &Graph, op: Op, vec_of: &dyn Fn(&Expr) -> Vec<f64>) -> Vec<Vec<f64>> {
    match op {
        Op::Dim(d) => match &g.dim(d).data {
            DimData::Persist { expr } => vec![vec_of(expr)],
            DimData::ReGlu { a, b } => vec![vec_of(a), vec_of(b)],
            other => panic!("{other:?} is not a schedulable op"),
        },
        Op::Lookup(l) => {
            let lu = g.lookup(l);
            lu.query_2d
                .iter()
                .chain(lu.key_2d.iter())
                .chain(lu.value_exprs.iter())
                .map(vec_of)
                .collect()
        }
    }
}

/// Linear width at each boundary: the rank of everything still to be computed,
/// restricted to the dimensions already born.  That is the number of residual
/// slots a perfect linear re-encoding would need, a lower bound the plan
/// reports next to the slot count actually used.
fn widths_at_boundaries(
    sg: &SchedGraph,
    pa: &[Phase],
    dim_birth: &HashMap<DimId, Phase>,
    max_phase: Phase,
    boundaries: &[Phase],
    op_vecs: &[Vec<Vec<f64>>],
    out_vecs: &[Vec<f64>],
) -> BTreeMap<Phase, usize> {
    let mut widths = BTreeMap::new();
    for &c in boundaries {
        let past: Vec<bool> = sg
            .all_dims
            .iter()
            .map(|d| dim_birth.get(d).copied().unwrap_or(max_phase + 1) <= c)
            .collect();
        let mask = |v: &Vec<f64>| -> Vec<f64> {
            v.iter().zip(past.iter()).map(|(x, &keep)| if keep { *x } else { 0.0 }).collect()
        };
        let rows = sg
            .ops
            .iter()
            .enumerate()
            .filter(|(i, _)| pa[*i] > c)
            .flat_map(|(i, _)| op_vecs[i].iter().map(&mask))
            .chain(out_vecs.iter().map(&mask));
        widths.insert(c, rank(rows));
    }
    widths
}

/// Everything the plan file reports, derived from a phase assignment.
pub fn analyze(mg: &MachineGraph, sg: &SchedGraph, pa: &[Phase]) -> Analysis {
    let g = &mg.graph;
    let outs = output_dims(mg);
    let protected = [g.position, g.inv_log_pos, g.position_sq];

    let max_phase = pa.iter().copied().max().unwrap_or(0);
    let num_layers = (max_phase / 4 + 1) as usize;

    let mut layers: Vec<[Vec<usize>; 4]> =
        (0..num_layers).map(|_| [Vec::new(), Vec::new(), Vec::new(), Vec::new()]).collect();
    for (i, &p) in pa.iter().enumerate() {
        let (l, slot) = ((p / 4) as usize, (p % 4) as usize);
        let wanted = match slot {
            0 => OpKind::Lookup,
            2 => OpKind::ReGlu,
            _ => OpKind::Persist,
        };
        assert_eq!(sg.kind(i), wanted, "op {i} is scheduled into the wrong phase family");
        layers[l][slot].push(i);
    }

    // ── Birth and death ─────────────────────────────────────────
    let mut dim_birth: HashMap<DimId, Phase> = HashMap::new();
    for &d in &sg.all_dims {
        if g.dim(d).is_input() {
            dim_birth.insert(d, -1);
        } else if let Some(&prod) = sg.dim_to_op.get(&d) {
            dim_birth.insert(d, pa[prod]);
        }
    }

    let last_boundary = 4 * num_layers as Phase - 1;
    let mut dim_death: HashMap<DimId, Phase> = HashMap::new();
    for &d in &sg.all_dims {
        if outs.contains(&d) || protected.contains(&d) {
            dim_death.insert(d, last_boundary + 1);
            continue;
        }
        let last =
            sg.consumers.get(&d).map_or(-1, |c| c.iter().map(|&i| pa[i]).max().unwrap_or(-1));
        if last >= 0 {
            dim_death.insert(d, last);
        } else if let Some(&b) = dim_birth.get(&d) {
            dim_death.insert(d, b);
        }
    }

    let boundaries: Vec<Phase> =
        (0..num_layers as Phase).flat_map(|l| [4 * l + 1, 4 * l + 3]).collect();
    let mut alive_after: BTreeMap<Phase, Vec<DimId>> = BTreeMap::new();
    for &c in &boundaries {
        let live: Vec<DimId> = sg
            .all_dims
            .iter()
            .copied()
            .filter(|d| match (dim_birth.get(d), dim_death.get(d)) {
                (Some(&b), Some(&x)) => b <= c && x > c,
                _ => false,
            })
            .collect();
        alive_after.insert(c, live);
    }

    // ── Linear widths, plain and with persists expanded ──────────
    let col: HashMap<DimId, usize> = sg.all_dims.iter().enumerate().map(|(i, &d)| (d, i)).collect();
    let n = sg.all_dims.len();
    let expr_vec = |e: &Expr| -> Vec<f64> {
        let mut v = vec![0.0; n];
        for &(d, c) in e.terms() {
            if let Some(&i) = col.get(&d) {
                v[i] = c;
            }
        }
        v
    };

    let op_vecs: Vec<Vec<Vec<f64>>> =
        sg.ops.iter().map(|&op| op_vectors(g, op, &expr_vec)).collect();
    let out_vecs: Vec<Vec<f64>> = mg.output_tokens.iter().map(|(_, e)| expr_vec(e)).collect();
    let lin_widths =
        widths_at_boundaries(sg, pa, &dim_birth, max_phase, &boundaries, &op_vecs, &out_vecs);

    let mut memo: HashMap<DimId, Vec<f64>> = HashMap::new();
    for &d in &sg.all_dims {
        expand_dim(g, d, &col, n, &mut memo);
    }
    let expr_vec_exp = |e: &Expr| -> Vec<f64> {
        let mut v = vec![0.0; n];
        for &(d, c) in e.terms() {
            if let Some(base) = memo.get(&d) {
                for (x, y) in v.iter_mut().zip(base.iter()) {
                    *x += c * y;
                }
            }
        }
        v
    };
    let op_vecs_exp: Vec<Vec<Vec<f64>>> =
        sg.ops.iter().map(|&op| op_vectors(g, op, &expr_vec_exp)).collect();
    let out_vecs_exp: Vec<Vec<f64>> =
        mg.output_tokens.iter().map(|(_, e)| expr_vec_exp(e)).collect();
    let exp_lin_widths = widths_at_boundaries(
        sg,
        pa,
        &dim_birth,
        max_phase,
        &boundaries,
        &op_vecs_exp,
        &out_vecs_exp,
    );

    let max_dep = alive_after.values().map(|v| v.len()).max().unwrap_or(0);
    let max_lin = lin_widths.values().copied().max().unwrap_or(0);
    let max_exp_lin = exp_lin_widths.values().copied().max().unwrap_or(0);
    let d_model = width_objective(mg, sg, pa, min_layers(sg));

    Analysis {
        num_layers,
        max_phase,
        layers,
        dim_birth,
        dim_death,
        alive_after,
        lin_widths,
        exp_lin_widths,
        max_dep,
        max_lin,
        max_exp_lin,
        d_model,
    }
}

/// A dimension written in terms of the dimensions a persist is not: persists
/// carry no information of their own, so expanding them shows the width a
/// re-encoding could reach if it were free to recompute them.
fn expand_dim(
    g: &Graph,
    d: DimId,
    col: &HashMap<DimId, usize>,
    n: usize,
    memo: &mut HashMap<DimId, Vec<f64>>,
) -> Vec<f64> {
    if let Some(v) = memo.get(&d) {
        return v.clone();
    }
    let v = match &g.dim(d).data {
        DimData::Persist { expr } => {
            let terms: Vec<(DimId, f64)> = expr.terms().to_vec();
            let mut v = vec![0.0; n];
            for (sub, c) in terms {
                let base = expand_dim(g, sub, col, n, memo);
                for (x, y) in v.iter_mut().zip(base.iter()) {
                    *x += c * y;
                }
            }
            v
        }
        _ => {
            let mut v = vec![0.0; n];
            if let Some(&i) = col.get(&d) {
                v[i] = 1.0;
            }
            v
        }
    };
    memo.insert(d, v.clone());
    v
}

// ── The width objective ─────────────────────────────────────────────

/// `d_model` for a schedule: twice the tightest half-width every MILP bound
/// allows, with every dimension dying as soon as its last consumer has read
/// it.  The solver minimizes exactly this, and on the released schedule the
/// value comes back out as the `milp_d_model` the plan file reports.
///
/// Two families bound it.  Occupancy: at every odd boundary, the slots that
/// must hold a value — born, not yet dead, and not consumed inside the same
/// half-layer.  Head count: an attention layer needs a head per lookup pair,
/// plus half a head for every dimension it erases or carries past.
pub fn width_objective(mg: &MachineGraph, sg: &SchedGraph, pa: &[Phase], n_layers: usize) -> usize {
    let g = &mg.graph;
    let outs = output_dims(mg);
    let protected = [g.position, g.inv_log_pos, g.position_sq];
    let p_max = 4 * n_layers as Phase;

    // death: the last phase that reads a dimension.  Position must additionally
    // outlive every persist1, which keys its passthrough on the hull.
    let mut death: HashMap<DimId, Phase> = HashMap::new();
    for &d in &sg.all_dims {
        if outs.contains(&d) {
            continue;
        }
        let cons = sg.consumers.get(&d).map(|v| v.as_slice()).unwrap_or(&[]);
        if cons.is_empty() && d != g.position {
            continue;
        }
        death.insert(d, cons.iter().map(|&i| pa[i]).max().unwrap_or(0));
    }
    if let Some(&dp) = death.get(&g.position) {
        let latest_p1 =
            sg.persists().map(|i| pa[i]).filter(|p| p % 4 == 1).max().unwrap_or(Phase::MIN);
        death.insert(g.position, dp.max(latest_p1));
    }

    // A lookup or ReGLU dimension consumed inside its own half-layer never
    // reaches a boundary, so it costs no slot.
    let needs_slot = |d: DimId| -> bool {
        match (sg.dim_to_op.get(&d), death.get(&d)) {
            (Some(&prod), Some(&x)) => x >= pa[prod] + 2,
            _ => true,
        }
    };

    let mut alive_sum: HashMap<Phase, i64> = HashMap::new();
    let mut bound: i64 = 0;
    let n_inputs = sg.inputs.len() as i64;

    for c in (1..p_max).step_by(2) {
        let mut ew: i64 = 0;
        let mut alive: i64 = 0;
        for &d in &sg.all_dims {
            let prod = sg.dim_to_op.get(&d).copied();
            let is_input = g.dim(d).is_input();
            if outs.contains(&d) || protected.contains(&d) {
                if is_input {
                    ew += 1;
                    alive += 1;
                } else if let Some(prod) = prod {
                    let born = i64::from(pa[prod] <= c);
                    ew += born;
                    alive += born;
                }
                continue;
            }
            let Some(&x) = death.get(&d) else { continue };
            if is_input {
                ew += i64::from(x >= c - 1);
                alive += i64::from(x > c);
            } else if let Some(prod) = prod {
                let born = pa[prod] <= c;
                let slot = g.dim(d).is_persist() || needs_slot(d);
                ew += i64::from(born && x >= c - 1 && slot);
                alive += i64::from(born && x > c);
            }
        }
        alive_sum.insert(c, alive);
        bound = bound.max(ew);
    }

    // Passthrough: a dimension a persist1 reads but the layer's attention does
    // not produce has to be carried across the attention block by a head.
    let mut p1_deps: HashMap<DimId, Vec<usize>> = HashMap::new();
    for i in sg.persists() {
        for &d in &sg.deps[i] {
            p1_deps.entry(d).or_default().push(i);
        }
    }

    for l in 0..n_layers as Phase {
        let at_layer = |i: usize, slot: Phase| pa[i] == 4 * l + slot;
        let n_lu: i64 = sg
            .lookups()
            .filter(|&i| at_layer(i, 0))
            .map(|i| match sg.ops[i] {
                Op::Lookup(lu) => (g.lookup(lu).value_exprs.len() as i64 + 1) / 2,
                Op::Dim(_) => unreachable!(),
            })
            .sum();
        let born: i64 = sg
            .lookups()
            .filter(|&i| at_layer(i, 0))
            .map(|i| sg.produced[i].len() as i64)
            .sum::<i64>()
            + sg.persists().filter(|&i| at_layer(i, 1)).count() as i64;
        let pt: i64 = p1_deps
            .iter()
            .filter(|(d, ps)| {
                let produced_here = sg
                    .dim_to_op
                    .get(*d)
                    .is_some_and(|&prod| g.dim(**d).is_lookup() && at_layer(prod, 0));
                !produced_here && ps.iter().any(|&p| at_layer(p, 1))
            })
            .count() as i64;
        let prev = if l == 0 { n_inputs } else { alive_sum[&(4 * l - 1)] };
        let dying = prev - alive_sum[&(4 * l + 1)] + born;
        bound = bound.max(2 * n_lu + dying + pt);
    }

    2 * ((bound.max(0) + 1) / 2) as usize
}

// ── The plan file ───────────────────────────────────────────────────

/// pyyaml folds a flow sequence at this column, and indents the fold by two.
const BEST_WIDTH: usize = 200;

/// A name as pyyaml would write it inside a flow sequence: plain unless a
/// flow indicator forces single quotes.
fn scalar(s: &str) -> String {
    let plain = !s.is_empty()
        && !s.contains([',', '[', ']', '{', '}', '#', '\'', '"', ':', ' '])
        && !s.starts_with(['-', '?', '&', '*', '!', '|', '>', '%', '@', '`']);
    if plain {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "''"))
    }
}

/// One `key: [a, b, c]`, folded the way pyyaml folds it: the break is taken
/// after the comma, once the line has already passed the width.
fn flow_seq(out: &mut String, indent: usize, key: &str, items: &[String]) {
    let head = format!("{:indent$}{key}: [", "", indent = indent);
    let mut col = head.len();
    out.push_str(&head);
    let cont = indent + 2;
    let mut fresh = true;
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
            col += 1;
            if col > BEST_WIDTH {
                out.push('\n');
                out.push_str(&" ".repeat(cont));
                col = cont;
                fresh = true;
            }
        }
        if !fresh {
            out.push(' ');
            col += 1;
        }
        out.push_str(item);
        col += item.len();
        fresh = false;
    }
    out.push_str("]\n");
}

/// The plan file, as `_write_plan` writes it.
pub fn write_plan(g: &Graph, sg: &SchedGraph, a: &Analysis) -> String {
    let name = |d: &DimId| scalar(g.name_of(*d));
    let mut out = String::new();
    out.push_str("summary:\n");
    out.push_str(&format!("  layers: {}\n", a.num_layers));
    out.push_str(&format!("  milp_d_model: {}\n", a.d_model));
    out.push_str(&format!("  max_dep_width: {}\n", a.max_dep));
    out.push_str(&format!("  max_lin_width: {}\n", a.max_lin));
    out.push_str("layers:\n");

    for l in 0..a.num_layers {
        let [attn, p1, ffn, p2] = &a.layers[l];
        let (c1, c3) = (4 * l as Phase + 1, 4 * l as Phase + 3);
        out.push_str(&format!("- layer: {l}\n"));

        let attn_dims: Vec<String> =
            attn.iter().flat_map(|&i| sg.produced[i].iter().map(name)).collect();
        flow_seq(&mut out, 2, "attention", &attn_dims);
        flow_seq(
            &mut out,
            2,
            "persist1",
            &p1.iter().flat_map(|&i| sg.produced[i].iter().map(name)).collect::<Vec<_>>(),
        );
        boundary(&mut out, g, a, "after_persist1", c1);
        flow_seq(
            &mut out,
            2,
            "ffn",
            &ffn.iter().flat_map(|&i| sg.produced[i].iter().map(name)).collect::<Vec<_>>(),
        );
        flow_seq(
            &mut out,
            2,
            "persist2",
            &p2.iter().flat_map(|&i| sg.produced[i].iter().map(name)).collect::<Vec<_>>(),
        );
        if c3 <= a.max_phase {
            boundary(&mut out, g, a, "after_persist2", c3);
        }
    }
    out
}

/// The report for one odd boundary: how many slots are live, how few would do,
/// and which dimensions they are.
fn boundary(out: &mut String, g: &Graph, a: &Analysis, key: &str, c: Phase) {
    let live = a.alive_after.get(&c).map(|v| v.as_slice()).unwrap_or(&[]);
    out.push_str(&format!("  {key}:\n"));
    out.push_str(&format!("    dep_width: {}\n", live.len()));
    out.push_str(&format!("    lin_width: {}\n", a.lin_widths.get(&c).copied().unwrap_or(0)));
    let mut names: Vec<String> = live.iter().map(|d| g.name_of(*d).to_string()).collect();
    names.sort();
    let names: Vec<String> = names.iter().map(|n| scalar(n)).collect();
    flow_seq(out, 4, "dims", &names);
}

/// The phase assignment a plan file records, read back out of it.  The plan
/// names every gate exactly once, so the schedule it describes is recoverable
/// without re-running the solver.
pub fn phases_from_plan(plan: &crate::plan::Plan, sg: &SchedGraph) -> Result<Vec<Phase>, String> {
    let at: HashMap<Op, usize> = sg.ops.iter().enumerate().map(|(i, &op)| (op, i)).collect();
    let mut pa: Vec<Option<Phase>> = vec![None; sg.ops.len()];
    let mut place = |op: Op, p: Phase| -> Result<(), String> {
        let i = *at.get(&op).ok_or_else(|| format!("the plan schedules an unknown gate {op:?}"))?;
        match pa[i] {
            Some(_) => Err(format!("the plan schedules {op:?} twice")),
            None => {
                pa[i] = Some(p);
                Ok(())
            }
        }
    };
    for entry in &plan.layers {
        let l = entry.layer as Phase;
        for &lu in &entry.attention {
            place(Op::Lookup(lu), 4 * l)?;
        }
        for (slot, dims) in [(1, &entry.persist1), (2, &entry.ffn), (3, &entry.persist2)] {
            for &d in dims.iter() {
                place(Op::Dim(d), 4 * l + slot)?;
            }
        }
    }
    pa.iter()
        .enumerate()
        .map(|(i, p)| p.ok_or_else(|| format!("the plan leaves {:?} unscheduled", sg.ops[i])))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `attention:` list of a layer is a Python set of dimension objects,
    /// so the release fixed an order no port can reproduce.  Everything else
    /// in the file is deterministic, so compare those lists as sets and the
    /// rest of the line byte for byte.
    fn canonical(text: &str) -> Vec<String> {
        text.lines()
            .map(|line| {
                let body = line.trim_start();
                match body.strip_prefix("attention: [").and_then(|r| r.strip_suffix(']')) {
                    Some(items) => {
                        let mut v: Vec<&str> =
                            items.split(", ").filter(|s| !s.is_empty()).collect();
                        v.sort();
                        format!("attention: [{}]", v.join(", "))
                    }
                    None => line.to_string(),
                }
            })
            .collect()
    }

    #[test]
    fn the_released_plan_is_what_this_schedule_reports() {
        let released = crate::release::PLAN;
        let mg = crate::interpreter::build();
        let sg = SchedGraph::build(&mg.graph);
        let plan = crate::plan::Plan::load(released, &mg.graph).unwrap();
        let pa = phases_from_plan(&plan, &sg).unwrap();
        let a = analyze(&mg, &sg, &pa);
        let ours = write_plan(&mg.graph, &sg, &a);

        let (want, got) = (canonical(released), canonical(&ours));
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(w, g, "line {} differs", i + 1);
        }
        assert_eq!(want.len(), got.len(), "the plan is a different number of lines");
    }

    #[test]
    fn the_critical_path_is_seven_layers_deep() {
        let mg = crate::interpreter::build();
        let sg = SchedGraph::build(&mg.graph);
        assert_eq!(min_layers(&sg), 7);
    }

    #[test]
    fn every_gate_is_scheduled_once_and_produces_what_it_is_named_for() {
        let mg = crate::interpreter::build();
        let sg = SchedGraph::build(&mg.graph);
        let produced: usize = sg.produced.iter().map(|p| p.len()).sum();
        assert_eq!(produced + sg.inputs.len(), sg.all_dims.len());
        for i in 0..sg.ops.len() {
            for &d in &sg.produced[i] {
                assert_eq!(sg.dim_to_op[&d], i);
            }
        }
    }
}
