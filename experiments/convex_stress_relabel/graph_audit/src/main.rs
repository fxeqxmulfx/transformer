use convex_relabel_graph_audit::{audit, audit_all, fingerprint};
use convex_stress_budgets::Result;
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["fingerprint"] => println!("{}", fingerprint()),
        ["audit", root, id] => println!("{}", audit(Path::new(root), id)?),
        ["audit-all", root, output] => {
            let summary = audit_all(Path::new(root), Path::new(output))?;
            println!(
                "{}",
                serde_json::json!({"audited_cells":summary["cells"],"auditor_fingerprint":summary["auditor_fingerprint"]})
            );
        }
        _ => {
            return Err(
                "usage: convex-relabel-graph-audit fingerprint | audit COHORT_ROOT CELL_ID | audit-all COHORT_ROOT FRESH_OUTPUT".into(),
            );
        }
    }
    Ok(())
}
