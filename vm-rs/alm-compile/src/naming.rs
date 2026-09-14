//! Naming graph nodes after the variables that hold them.
//!
//! The original calls `auto_name(locals())` once, at the end of
//! `wasm/interpreter.py::build()`, and CPython hands it the locals in
//! `co_varnames` order — plain locals in order of first binding, then the
//! cell variables (anything a nested function or a generator expression
//! captures) after them.  The resulting names are not cosmetic: `plan.yaml`
//! resolves every scheduled dimension *by name*, so `store_bytes[0]` and
//! `unsigned_branch+` have to come out of this port spelled exactly as they
//! came out of Python.
//!
//! Rust has no `locals()`, so the caller states the same sequence explicitly
//! (see `interpreter::build`), and this module is just the renaming rule.

use crate::expr::{DimId, Expr};
use crate::graph::{DimData, Graph};

/// One entry of the recorded variable sequence.
#[derive(Clone, Debug)]
pub enum Binding {
    Dim(DimId),
    Expr(Expr),
    Seq(Vec<Binding>),
}

impl From<DimId> for Binding {
    fn from(d: DimId) -> Self {
        Binding::Dim(d)
    }
}

impl From<Expr> for Binding {
    fn from(e: Expr) -> Self {
        Binding::Expr(e)
    }
}

impl From<&Expr> for Binding {
    fn from(e: &Expr) -> Self {
        Binding::Expr(e.clone())
    }
}

fn is_default_named(g: &Graph, d: DimId, prefix: &str) -> bool {
    g.name_of(d).starts_with(prefix)
}

/// `_name_expr_dims`: name the still-anonymous ReGLU, persist and lookup
/// dimensions an expression is built from.  Positive ReGLU coefficients get
/// `name+`, negative ones `name-`, both numbered from the second onwards.
fn name_expr_dims(g: &mut Graph, name: &str, expr: &Expr) {
    let (mut pos_idx, mut neg_idx, mut lu_idx, mut persist_idx) = (0usize, 0, 0, 0);
    for &(d, coeff) in expr.terms() {
        let data = g.dim(d).data.clone();
        match data {
            DimData::Persist { .. } if is_default_named(g, d, "persist_") => {
                let n = if persist_idx == 0 { name.to_string() } else { format!("{name}${persist_idx}") };
                g.set_name(d, n);
                persist_idx += 1;
            }
            DimData::ReGlu { .. } if is_default_named(g, d, "reglu_") => {
                if coeff > 0.0 {
                    let n = if pos_idx == 0 { format!("{name}+") } else { format!("{name}+{pos_idx}") };
                    g.set_name(d, n);
                    pos_idx += 1;
                } else {
                    let n = if neg_idx == 0 { format!("{name}-") } else { format!("{name}-{neg_idx}") };
                    g.set_name(d, n);
                    neg_idx += 1;
                }
            }
            DimData::LookUp { lookup, .. } if is_default_named(g, d, "lookup_") => {
                let n = if lu_idx == 0 { format!("{name}_lu") } else { format!("{name}_lu{lu_idx}") };
                g.set_name(d, n.clone());
                if g.lookup(lookup).name.is_none() {
                    g.lookups[lookup as usize].name = Some(n);
                }
                lu_idx += 1;
            }
            _ => {}
        }
    }
}

fn name_one(g: &mut Graph, name: &str, b: &Binding) {
    match b {
        Binding::Dim(d) => {
            let d = *d;
            if g.dim(d).is_input() {
                return;
            }
            g.set_name(d, name.to_string());
            if let DimData::LookUp { lookup, .. } = g.dim(d).data {
                if g.lookup(lookup).name.is_none() {
                    g.lookups[lookup as usize].name = Some(name.to_string());
                }
            }
        }
        Binding::Expr(e) => name_expr_dims(g, name, e),
        Binding::Seq(items) => {
            for (i, item) in items.iter().enumerate() {
                match item {
                    Binding::Dim(d) => {
                        let d = *d;
                        if g.dim(d).is_input() {
                            continue;
                        }
                        g.set_name(d, format!("{name}[{i}]"));
                        if let DimData::LookUp { lookup, .. } = g.dim(d).data {
                            if g.lookup(lookup).name.is_none() {
                                // The list branch names the lookup after the
                                // list, not after the element.
                                g.lookups[lookup as usize].name = Some(name.to_string());
                            }
                        }
                    }
                    Binding::Expr(e) => name_expr_dims(g, &format!("{name}[{i}]"), e),
                    Binding::Seq(_) => {}
                }
            }
        }
    }
}

/// Apply the recorded variable sequence, in order.  First name wins for
/// anonymous dimensions reached through an expression; for a dimension bound
/// directly to a variable, the last name wins, exactly as in Python.
pub fn auto_name(g: &mut Graph, bindings: &[(&str, Binding)]) {
    for (name, b) in bindings {
        name_one(g, name, b);
    }
}
