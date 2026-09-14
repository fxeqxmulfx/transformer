//! The computation graph: dimensions, lookups, and the combinators that build
//! them.  A port of `transformer_vm/graph/core.py`.
//!
//! Python keeps the graph in module globals and resets them with
//! `reset_graph()`; here it is a [`Graph`] value, and `Graph::new()` is the
//! reset.  Dimension ids are assigned in creation order and are load-bearing:
//! the default names (`reglu_<id>`, `persist_<id>`) appear verbatim in
//! `plan.yaml`, and the slot allocator breaks ties by `id`.

use std::collections::HashMap;

use crate::expr::{DimId, Expr, ExprKey};

/// Large constant that zeroes an attention key out (the `clear_key` mechanism).
pub const BIG: f64 = 1e30;
/// Offset applied to every attention key, for numerical-stability tuning.
pub const KEY_OFFSET: f64 = 0.0;
/// Tie-break weight favouring more recent tokens in hardmax attention.
///
/// Not a tie-break: it is a noise margin, and the machine stops without it.
/// See `todo3.md` section 2a and `alm-hull/src/gap.rs`.
pub const LATEST_ALPHA: f64 = 0.3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TieBreak {
    Latest,
    Average,
}

/// Index of a lookup in the graph's arena — also its `LookUp.id`.
pub type LookUpId = u32;

/// What a dimension is, and what it carries.
#[derive(Clone, Debug)]
pub enum DimData {
    /// Read straight off the token embedding or the positional encoding.
    Input,
    /// A cumulative sum (built, but unused by the WASM machine).
    CumSum { value: Expr },
    /// A linear combination materialized into its own residual slot.
    Persist { expr: Expr },
    /// `relu(b) * a`, one FFN neuron.
    ReGlu { a: Expr, b: Expr },
    /// One value channel of an attention lookup.
    LookUp { lookup: LookUpId, value_index: usize },
}

#[derive(Clone, Debug)]
pub struct Dim {
    pub id: DimId,
    pub name: String,
    pub data: DimData,
}

impl Dim {
    pub fn is_input(&self) -> bool {
        matches!(self.data, DimData::Input)
    }
    pub fn is_reglu(&self) -> bool {
        matches!(self.data, DimData::ReGlu { .. })
    }
    pub fn is_persist(&self) -> bool {
        matches!(self.data, DimData::Persist { .. })
    }
    pub fn is_lookup(&self) -> bool {
        matches!(self.data, DimData::LookUp { .. })
    }
}

#[derive(Clone, Debug)]
pub struct LookUp {
    pub id: LookUpId,
    pub name: Option<String>,
    pub value_exprs: Vec<Expr>,
    /// `[qx, qy]`.
    pub query_2d: [Expr; 2],
    /// `[kx, ky]`.
    pub key_2d: [Expr; 2],
    pub tie_break: TieBreak,
    pub dims: Vec<DimId>,
}

pub struct Graph {
    pub dims: Vec<Dim>,
    pub lookups: Vec<LookUp>,
    pub one: DimId,
    pub position: DimId,
    pub inv_log_pos: DimId,
    pub position_sq: DimId,
    multiply_cache: HashMap<(ExprKey, ExprKey), Expr>,
    reglu_cache: HashMap<(ExprKey, ExprKey), DimId>,
    stepglu_cache: HashMap<(ExprKey, ExprKey), Expr>,
    clear_key_cache: HashMap<ExprKey, Expr>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// `reset_graph()`: a fresh graph carrying only the four positional inputs,
    /// in the order every ALM expects them.
    pub fn new() -> Self {
        let mut g = Graph {
            dims: Vec::new(),
            lookups: Vec::new(),
            one: 0,
            position: 1,
            inv_log_pos: 2,
            position_sq: 3,
            multiply_cache: HashMap::new(),
            reglu_cache: HashMap::new(),
            stepglu_cache: HashMap::new(),
            clear_key_cache: HashMap::new(),
        };
        g.one = g.input("one");
        g.position = g.input("position");
        g.inv_log_pos = g.input("inv_log_pos");
        g.position_sq = g.input("position_sq");
        g
    }

    fn push(&mut self, name: String, data: DimData) -> DimId {
        let id = self.dims.len() as DimId;
        self.dims.push(Dim { id, name, data });
        id
    }

    pub fn input(&mut self, name: &str) -> DimId {
        self.push(name.to_string(), DimData::Input)
    }

    pub fn dim(&self, d: DimId) -> &Dim {
        &self.dims[d as usize]
    }

    pub fn lookup(&self, l: LookUpId) -> &LookUp {
        &self.lookups[l as usize]
    }

    pub fn name_of(&self, d: DimId) -> &str {
        &self.dims[d as usize].name
    }

    pub fn set_name(&mut self, d: DimId, name: String) {
        self.dims[d as usize].name = name;
    }

    // ── Constructors ────────────────────────────────────────────────

    /// `x` as an expression, for the cases where Python would coerce.
    pub fn konst(&self, c: f64) -> Expr {
        Expr::scaled(self.one, c)
    }

    pub fn one_expr(&self) -> Expr {
        Expr::dim(self.one)
    }

    /// A raw `ReGLUDimension`, bypassing the memo.  `_make_multiply` and the
    /// specialized instruction decoder both need this.
    pub fn reglu_dim(&mut self, a: Expr, b: Expr, name: Option<&str>) -> DimId {
        let id = self.dims.len() as DimId;
        let name = name.map(str::to_string).unwrap_or_else(|| format!("reglu_{id}"));
        self.push(name, DimData::ReGlu { a, b })
    }

    /// `relu(b) * a` — one FFN neuron, memoized on the pair of operands.
    pub fn reglu(&mut self, a: &Expr, b: &Expr) -> Expr {
        let key = (a.key(), b.key());
        if let Some(&r) = self.reglu_cache.get(&key) {
            return Expr::dim(r);
        }
        let r = self.reglu_dim(a.clone(), b.clone(), None);
        self.reglu_cache.insert(key, r);
        Expr::dim(r)
    }

    /// `a * step(b >= 0)`, as `reglu(a, b+1) - reglu(a, b)` collapsed into one
    /// persist slot.  Exact for integer `b`.
    pub fn stepglu(&mut self, a: &Expr, b: &Expr) -> Expr {
        let key = (a.key(), b.key());
        if let Some(e) = self.stepglu_cache.get(&key) {
            return e.clone();
        }
        let b1 = b.add(&self.one_expr());
        let r1 = self.reglu_dim(a.clone(), b1, None);
        let r2 = self.reglu_dim(a.clone(), b.clone(), None);
        let combined = Expr::from_terms(vec![(r1, 1.0), (r2, -1.0)]);
        let result = self.persist(&combined, None);
        self.stepglu_cache.insert(key, result.clone());
        result
    }

    /// Materialize a linear expression into a dedicated residual slot.
    pub fn persist(&mut self, expr: &Expr, name: Option<&str>) -> Expr {
        let id = self.dims.len() as DimId;
        let name = name.map(str::to_string).unwrap_or_else(|| format!("persist_{id}"));
        let d = self.push(name, DimData::Persist { expr: expr.clone() });
        Expr::dim(d)
    }

    pub fn cumsum(&mut self, value: &Expr, name: Option<&str>) -> DimId {
        let id = self.dims.len() as DimId;
        let name = name.map(str::to_string).unwrap_or_else(|| format!("cumsum_{id}"));
        self.push(name, DimData::CumSum { value: value.clone() })
    }

    /// `a * b` for two general expressions: two ReGLUs straddling zero,
    /// summed into one persist slot.  Memoized on the operand pair.
    fn make_multiply(&mut self, a: &Expr, b: &Expr) -> Expr {
        let key = (a.key(), b.key());
        if let Some(e) = self.multiply_cache.get(&key) {
            return e.clone();
        }
        let neg_b = b.neg();
        let r1 = self.reglu_dim(a.clone(), b.clone(), None);
        let r2 = self.reglu_dim(a.clone(), neg_b, None);
        let combined = Expr::from_terms(vec![(r1, 1.0), (r2, -1.0)]);
        let result = self.persist(&combined, None);
        self.multiply_cache.insert(key, result.clone());
        result
    }

    // ── Attention ───────────────────────────────────────────────────

    /// Map a 1D key (plus an optional clear flag) to the parabolic 2D key
    /// `k -> (2k, -k^2)` the hull head scores against.
    fn to_2d_key(&mut self, k: &Expr, clear_key: Option<&Expr>, tie_break: TieBreak) -> [Expr; 2] {
        let one_expr = self.one_expr();
        let k_abs = if k.len() == 1 && k.contains(self.one) {
            let c = k.get(self.one);
            Expr::scaled(self.one, c * c)
        } else if k.len() == 1 && k.contains(self.position) {
            let c = k.get(self.position);
            Expr::scaled(self.position_sq, c * c)
        } else {
            self.make_multiply(k, k)
        };
        let kx = k.mul(2.0).sub(&one_expr.mul(2.0 * KEY_OFFSET));
        let mut ky = k_abs
            .neg()
            .add(&k.mul(2.0 * KEY_OFFSET))
            .sub(&one_expr.mul(KEY_OFFSET * KEY_OFFSET));
        if let Some(ck) = clear_key {
            let clear = if ck.len() == 1 {
                ck.clone()
            } else {
                let ck_key = ck.key();
                if !self.clear_key_cache.contains_key(&ck_key) {
                    let p = self.persist(ck, None);
                    self.clear_key_cache.insert(ck_key.clone(), p);
                }
                self.clear_key_cache[&ck_key].clone()
            };
            ky = ky.sub(&clear.mul(BIG));
        }
        match tie_break {
            TieBreak::Latest => ky = ky.add(&Expr::scaled(self.inv_log_pos, LATEST_ALPHA)),
            TieBreak::Average => ky = self.one_expr(),
        }
        [kx, ky]
    }

    fn to_2d_query(&self, q: &Expr) -> [Expr; 2] {
        let one_expr = self.one_expr();
        [q.sub(&one_expr.mul(KEY_OFFSET)), one_expr]
    }

    /// Hard attention: read the values written at the key that matches the
    /// query.  Returns one dimension per value channel.
    pub fn fetch(
        &mut self,
        values: &[Expr],
        query: Option<&Expr>,
        key: Option<&Expr>,
        clear_key: Option<&Expr>,
        tie_break: TieBreak,
    ) -> Vec<DimId> {
        let q = query.cloned().unwrap_or_default();
        let k = key.cloned().unwrap_or_default();
        let key_2d = self.to_2d_key(&k, clear_key, tie_break);
        let query_2d = self.to_2d_query(&q);

        let lu_id = self.lookups.len() as LookUpId;
        let mut dims = Vec::with_capacity(values.len());
        self.lookups.push(LookUp {
            id: lu_id,
            name: None,
            value_exprs: values.to_vec(),
            query_2d,
            key_2d,
            tie_break,
            dims: Vec::new(),
        });
        for i in 0..values.len() {
            let d = self.push(format!("lookup_{lu_id}_v{i}"), DimData::LookUp {
                lookup: lu_id,
                value_index: i,
            });
            dims.push(d);
        }
        self.lookups[lu_id as usize].dims = dims.clone();
        dims
    }

    /// Convenience for the single-value case.
    pub fn fetch1(
        &mut self,
        value: &Expr,
        query: Option<&Expr>,
        key: Option<&Expr>,
        clear_key: Option<&Expr>,
    ) -> DimId {
        self.fetch(std::slice::from_ref(value), query, key, clear_key, TieBreak::Latest)[0]
    }

    /// Cumulative sum via attention averaging: `avg * position`.
    ///
    /// Position 0 (the start token, `one = 0`) is excluded from the average
    /// because its `ky = 0 < 1`, so the denominator is `p`, not `p + 1`;
    /// multiplying by position recovers the exact cumulative sum.
    pub fn fetch_sum(&mut self, values: &[Expr]) -> Vec<Expr> {
        let key = Expr::scaled(self.one, KEY_OFFSET);
        let query = Expr::scaled(self.one, KEY_OFFSET);
        let avg_dims =
            self.fetch(values, Some(&query), Some(&key), None, TieBreak::Average);
        let pos = Expr::dim(self.position);
        avg_dims.into_iter().map(|d| self.reglu(&Expr::dim(d), &pos)).collect()
    }
}
