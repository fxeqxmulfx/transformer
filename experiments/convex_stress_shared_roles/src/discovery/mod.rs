//! CM40 SharedRoles/{Basic,Identification} at ec5f839d. This is a new
//! augmented-matrix certificate, not a CM35 certificate with replaced codes.
//! Compaction and readout expansion port readout_kernel/discovery.rs at
//! 21e86e7c. The independent finite-field and full-basis checkers are unchanged.
use crate::*;
use kernel::discovery::{BasisAudit, Code, Input, Names};
use serde_json::Value;
mod plan;
use plan::Plan;
pub use plan::problem;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub input: Input,
    /// Declared answer for every raw vocabulary label, including unseen ones.
    pub readout: Vec<usize>,
    pub answers: usize,
    pub prime: u16,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub field: finite_system::Work,
    pub matrix_entries: usize,
    pub role_rows: usize,
    pub code_coordinates: usize,
    pub group_pair_comparisons: usize,
    pub expanded_coordinates: usize,
    pub observed_vector_checks: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub request: Request,
    pub names: Names,
    /// Row/column dictionary indices sharing exactly the same token ID.
    pub alignments: Vec<[usize; 2]>,
    pub matrix_shape: [usize; 2],
    pub buffer_preflight_bytes: u64,
    pub field: Option<finite_system::Outcome>,
    pub field_audit: Option<Value>,
    pub basis_audit: Option<BasisAudit>,
    pub grouped_code: Option<Code>,
    pub raw_code: Option<Code>,
    pub reason: String,
    pub work: Work,
    pub raw_label_identification: bool,
    pub target_readout_compatibility_proved: bool,
    pub shared_role_target_proved: bool,
}
fn certificate(request: Request, field: Option<finite_system::Outcome>) -> Result<Report> {
    let plan = Plan::new(&request)?;
    let mut out = Report {
        request,
        names: plan.names.clone(),
        alignments: plan.alignments.clone(),
        matrix_shape: plan.shape,
        buffer_preflight_bytes: 0,
        field: None,
        field_audit: None,
        basis_audit: None,
        grouped_code: None,
        raw_code: None,
        reason: String::new(),
        work: Work::default(),
        raw_label_identification: false,
        target_readout_compatibility_proved: false,
        shared_role_target_proved: false,
    };
    if let Some(reason) = plan.refusal(&out.request) {
        if field.is_some() {
            return Err("refused shared-role matrix has a field outcome".into());
        }
        out.reason = reason.into();
        return Ok(out);
    }
    out.buffer_preflight_bytes = plan.buffer_bytes(&out.request);
    let matrix = plan.matrix(&out.request)?;
    let field = field.ok_or("missing shared-role field outcome")?;
    out.field_audit = Some(finite_system::check(&matrix, &field)?);
    out.work.field = field.work.clone();
    out.work.matrix_entries = plan.shape[0] * plan.shape[1];
    out.work.role_rows = plan.alignments.len();
    out.reason = field.reason.clone();
    if let Some(space) = &field.space {
        out.basis_audit = Some(kernel::discovery::check_basis(
            &matrix,
            &field,
            &space.directions,
        )?);
        let code = plan.code(&out.request, &space.directions)?;
        out.work.code_coordinates = code.dimension * plan.shape[1];
        let mut collision = false;
        for (i, a) in code.classes.iter().enumerate() {
            for b in &code.classes[i + 1..] {
                out.work.group_pair_comparisons += 1;
                collision |= a == b;
            }
        }
        if collision {
            out.reason = "incompatible_shared_group_codes".into();
        } else {
            let raw = plan.expand(&out.request, &code)?;
            out.work.expanded_coordinates = raw.dimension * raw.classes.len();
            for &[cell, label] in &out.request.input.observations {
                out.work.observed_vector_checks += 1;
                if raw
                    .query(cell % out.request.input.n, cell / out.request.input.n)
                    .as_deref()
                    != raw.label(label)
                {
                    return Err("shared code fails an original raw observation".into());
                }
            }
            out.raw_code = Some(raw);
            out.reason = "complete_shared_role_kernel".into();
        }
        out.grouped_code = Some(code);
    }
    out.field = Some(field);
    Ok(out)
}
pub fn run(request: Request) -> Result<Report> {
    let plan = Plan::new(&request)?;
    let field = if plan.refusal(&request).is_none() {
        Some(finite_system::solve(plan.matrix(&request)?)?)
    } else {
        None
    };
    certificate(request, field)
}
/// Rebuild the actual matrix and independently replay the saved row program.
/// No elimination solver or self-declared completeness flag authorizes state.
pub fn check(request: &Request, report: &Report) -> Result<()> {
    if &certificate(request.clone(), report.field.clone())? != report {
        return Err("shared-role certificate, complete codes, scope or costs changed".into());
    }
    Ok(())
}
pub fn replay(request: &Request, report: &Report) -> Result<()> {
    check(request, report)?;
    if run(request.clone())? != *report {
        return Err("shared-role discovery does not replay".into());
    }
    Ok(())
}
