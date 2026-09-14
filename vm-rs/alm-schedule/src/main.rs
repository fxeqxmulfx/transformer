//! Solve the scheduling program and write the plan.
//!
//! The last piece of the Python pipeline to be ported, and the only one that
//! needs a solver: `alm-compile` builds the mixed-integer program, HiGHS finds
//! a schedule of minimal residual width, and the plan file is written from it.
//!
//!     alm-schedule [-o plan.yaml] [--layers N] [--max-ffn M]
//!                  [--time-limit SECONDS] [--threads N] [--lp FILE]

use alm_compile::interpreter;
use alm_compile::milp::{self, Cmp, Milp, VarKind};
use alm_compile::scheduler::{self, SchedGraph};
use good_lp::solvers::highs::highs;
use good_lp::{Expression, ProblemVariables, Solution, SolverModel, Variable};
use std::process::ExitCode;

struct Args {
    out: String,
    layers: Option<usize>,
    max_ffn: Option<usize>,
    time_limit: f64,
    threads: Option<u32>,
    lp: Option<String>,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        out: "plan.yaml".into(),
        layers: None,
        max_ffn: None,
        time_limit: 3600.0,
        threads: None,
        lp: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut next = |what: &str| it.next().ok_or(format!("{what} wants a value"));
        match flag.as_str() {
            "-o" | "--out" => a.out = next("-o")?,
            "--lp" => a.lp = Some(next("--lp")?),
            "--layers" => a.layers = Some(next("--layers")?.parse().map_err(|_| "bad --layers")?),
            "--max-ffn" => {
                a.max_ffn = Some(next("--max-ffn")?.parse().map_err(|_| "bad --max-ffn")?)
            }
            "--time-limit" => {
                a.time_limit = next("--time-limit")?.parse().map_err(|_| "bad --time-limit")?
            }
            "--threads" => {
                a.threads = Some(next("--threads")?.parse().map_err(|_| "bad --threads")?)
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

/// Hand the program to HiGHS and read the columns back.
fn solve(m: &Milp, args: &Args) -> Result<Vec<f64>, String> {
    let mut vars = ProblemVariables::new();
    let cols: Vec<Variable> = m
        .vars
        .iter()
        .map(|v| {
            let def = good_lp::variable().name(v.name.clone());
            let def = match v.kind {
                VarKind::Binary => def.binary(),
                VarKind::Integer => match v.hi {
                    Some(hi) => def.integer().min(v.lo).max(hi),
                    None => def.integer().min(v.lo),
                },
            };
            vars.add(def)
        })
        .collect();

    let objective = Expression::from(cols[m.d_half]);
    let mut problem = vars.minimise(objective).using(highs).set_time_limit(args.time_limit);
    if let Some(t) = args.threads {
        problem = problem.set_threads(t);
    }
    problem.set_verbose(true);
    for row in &m.rows {
        let mut lhs = Expression::with_capacity(row.terms.len());
        for &(v, c) in &row.terms {
            lhs.add_mul(c, cols[v]);
        }
        problem = problem.with(match row.op {
            Cmp::Le => lhs.leq(row.rhs),
            Cmp::Ge => lhs.geq(row.rhs),
            Cmp::Eq => lhs.eq(row.rhs),
        });
    }
    let solution = problem.solve().map_err(|e| format!("HiGHS: {e}"))?;
    Ok(cols.iter().map(|&c| solution.value(c)).collect())
}

fn run() -> Result<(), String> {
    let args = parse()?;
    let mg = interpreter::build();
    let sg = SchedGraph::build(&mg.graph);
    let layers = args.layers.unwrap_or_else(|| scheduler::min_layers(&sg));
    let m = milp::build(&mg, &sg, layers, args.max_ffn, None);
    println!(
        "{} gates, {} dims, {} layers: {} variables, {} rows",
        sg.ops.len(),
        sg.all_dims.len(),
        layers,
        m.vars.len(),
        m.rows.len()
    );

    if let Some(path) = &args.lp {
        std::fs::write(path, m.to_lp()).map_err(|e| format!("{path}: {e}"))?;
        println!("program written to {path}");
        return Ok(());
    }

    let values = solve(&m, &args)?;
    let broken = m.violations(&values);
    if !broken.is_empty() {
        return Err(format!("the solution breaks {} rows", broken.len()));
    }

    let pa = m.schedule(&sg, &values);
    let a = scheduler::analyze(&mg, &sg, &pa);
    println!(
        "{} layers, d_model={}, max_dep={}, max_lin={}, max_exp_lin={}",
        a.num_layers, a.d_model, a.max_dep, a.max_lin, a.max_exp_lin
    );
    for l in 0..a.num_layers {
        let [attn, p1, ffn, p2] = &a.layers[l];
        let (c1, c3) = (4 * l as i64 + 1, 4 * l as i64 + 3);
        let w = |c: i64| {
            (
                a.alive_after.get(&c).map_or(0, |v| v.len()),
                a.lin_widths.get(&c).copied().unwrap_or(0),
                a.exp_lin_widths.get(&c).copied().unwrap_or(0),
            )
        };
        println!(
            "  L{l}: A[{}] P1[{}] F[{}] P2[{}]  after_attn={:?} after_ffn={:?}",
            attn.len(),
            p1.len(),
            ffn.len(),
            p2.len(),
            w(c1),
            w(c3)
        );
    }

    let text = scheduler::write_plan(&mg.graph, &sg, &a);
    std::fs::write(&args.out, text).map_err(|e| format!("{}: {e}", args.out))?;
    println!("schedule written to {}", args.out);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("alm-schedule: {e}");
            ExitCode::FAILURE
        }
    }
}
