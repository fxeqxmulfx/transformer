//! CM34 original equations, two shift gauges and all independent directions.
//! No raw arithmetic, readout, held-out label or proposed answer is an input.
use crate::*;
use serde_json::Value;
mod audit;
mod plan;
pub use audit::{BasisAudit, check, check_basis};
pub use plan::Code;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    /// All observations must fit; this cap never selects an equation prefix.
    pub equations: usize,
    pub elimination: finite_system::Policy,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            equations: 8190,
            elimination: finite_system::Policy {
                nullity: 8,
                ..Default::default()
            },
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub n: usize,
    pub vocabulary: usize,
    /// Cell=row+n*column; the names have no arithmetic answer meaning.
    pub observations: Vec<[usize; 2]>,
    pub policy: Policy,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub input: Input,
    pub prime: u16,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Names {
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
    pub classes: Vec<usize>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub field: finite_system::Work,
    pub code_coordinates: usize,
    pub observed_vector_checks: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub request: Request,
    pub names: Names,
    pub buffer_preflight_bytes: u64,
    pub field: Option<finite_system::Outcome>,
    pub field_audit: Option<Value>,
    pub basis_audit: Option<BasisAudit>,
    pub code: Option<Code>,
    pub reason: String,
    pub raw_label_identification: bool,
    pub target_readout_compatibility_proved: bool,
    pub work: Work,
}
pub fn run(request: Request) -> Result<Report> {
    let names = request.names()?;
    let mut out = Report {
        request,
        names,
        buffer_preflight_bytes: 0,
        field: None,
        field_audit: None,
        basis_audit: None,
        code: None,
        reason: String::new(),
        raw_label_identification: false,
        target_readout_compatibility_proved: false,
        work: Work::default(),
    };
    if let Some(reason) = out.request.refusal(&out.names) {
        out.reason = reason.into();
        return Ok(out);
    }
    let problem = out.request.problem(&out.names)?;
    out.buffer_preflight_bytes = 4 * problem.validate()? + 512 * 1024 * 1024;
    if out.buffer_preflight_bytes > 8 * 1024 * 1024 * 1024 {
        out.reason = "discovery_memory_cap".into();
        return Ok(out);
    }
    let field = finite_system::solve(problem.clone())?;
    out.field_audit = Some(finite_system::check(&problem, &field)?);
    out.work.field = field.work.clone();
    out.reason = field.reason.clone();
    if let Some(space) = &field.space {
        out.basis_audit = Some(audit::basis_after_field_check(
            &problem,
            &field,
            &space.directions,
        )?);
        let code = Code::from_directions(out.names.clone(), out.request.prime, &space.directions)?;
        out.work.code_coordinates = space.directions.len() * problem.system.variables;
        for &[cell, label] in &out.request.input.observations {
            out.work.observed_vector_checks += 1;
            if code.query(cell % out.request.input.n, cell / out.request.input.n)
                != code.label(label).map(<[u16]>::to_vec)
            {
                return Err("complete basis disagrees with an original observation".into());
            }
        }
        out.code = Some(code);
        out.reason = "complete_kernel_vector_code".into();
    }
    out.field = Some(field);
    Ok(out)
}
pub fn replay(request: &Request, report: &Report) -> Result<()> {
    check(request, report)?;
    if run(request.clone())? != *report {
        return Err("complete-kernel construction or charged work does not replay".into());
    }
    Ok(())
}
