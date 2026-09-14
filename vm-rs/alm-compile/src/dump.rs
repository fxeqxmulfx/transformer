//! A canonical text dump of a built graph, so the port can be diffed against
//! the Python it came from.
//!
//! Coefficients are printed as their float64 bit pattern: the point of the
//! exercise is that the two builders agree to the last bit, and a decimal
//! rendering would hide exactly the disagreements worth finding.

use std::fmt::Write;

use crate::expr::Expr;
use crate::graph::{DimData, Graph, TieBreak};
use crate::interpreter::MachineGraph;

fn terms(g: &Graph, e: &Expr) -> String {
    let mut s = String::from("[");
    for (i, &(d, c)) in e.terms().iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        let _ = write!(s, "{}:{:016x}", g.name_of(d), c.to_bits());
    }
    s.push(']');
    s
}

pub fn dump(mg: &MachineGraph) -> String {
    let g = &mg.graph;
    let mut out = String::new();

    for dim in &g.dims {
        match &dim.data {
            DimData::Input => {
                let _ = writeln!(out, "dim {} input {}", dim.id, dim.name);
            }
            DimData::Persist { expr } => {
                let _ = writeln!(
                    out,
                    "dim {} persist {} {}",
                    dim.id,
                    dim.name,
                    terms(g, expr)
                );
            }
            DimData::ReGlu { a, b } => {
                let _ = writeln!(
                    out,
                    "dim {} reglu {} a={} b={}",
                    dim.id,
                    dim.name,
                    terms(g, a),
                    terms(g, b)
                );
            }
            DimData::LookUp { lookup, value_index } => {
                let _ = writeln!(
                    out,
                    "dim {} lookup {} lu={} v={}",
                    dim.id, dim.name, lookup, value_index
                );
            }
            DimData::CumSum { value } => {
                let _ = writeln!(
                    out,
                    "dim {} cumsum {} {}",
                    dim.id,
                    dim.name,
                    terms(g, value)
                );
            }
        }
    }

    for lu in &g.lookups {
        let tb = match lu.tie_break {
            TieBreak::Latest => "latest",
            TieBreak::Average => "average",
        };
        let _ = writeln!(
            out,
            "lookup {} {} tie={} qx={} qy={} kx={} ky={}",
            lu.id,
            lu.name.as_deref().unwrap_or("-"),
            tb,
            terms(g, &lu.query_2d[0]),
            terms(g, &lu.query_2d[1]),
            terms(g, &lu.key_2d[0]),
            terms(g, &lu.key_2d[1]),
        );
        for (i, v) in lu.value_exprs.iter().enumerate() {
            let _ = writeln!(out, "lookup {} value {} {}", lu.id, i, terms(g, v));
        }
    }

    let mut ins: Vec<&(String, Expr)> = mg.input_tokens.iter().collect();
    ins.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, e) in ins {
        let _ = writeln!(out, "in {} {}", name, terms(g, e));
    }

    let mut outs: Vec<&(String, Expr)> = mg.output_tokens.iter().collect();
    outs.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, e) in outs {
        let _ = writeln!(out, "out {} {}", name, terms(g, e));
    }

    out
}
