//! Fixed-weight controls for any trained associative read count. A zero
//! budget bypasses the local read in SlotLayer while keeping its parameters.
//! ReadHorizon and TypedLookupPaths describe the positive-budget transitions.
use super::*;
use crate::head::{PathRead, PeerRead};

pub(super) fn controls(read: PeerRead, requested: &str) -> Result<Vec<(String, PeerRead)>> {
    let path = read
        .path
        .ok_or("checkpoint has no repeated associative read")?;
    let mut seen = std::collections::BTreeSet::new();
    let mut controls = Vec::new();
    for word in requested.split(',') {
        let passes: usize = word.parse()?;
        if !seen.insert(passes) {
            return Err("duplicate evaluation read budget".into());
        }
        controls.push((
            format!("reads-{passes}"),
            PeerRead {
                path: Some(PathRead { passes, ..path }),
                ..read
            },
        ));
    }
    Ok(controls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabling_or_extending_reads_preserves_every_other_setting_and_parameter() -> Result<()> {
        for read in [
            PeerRead::AGREEMENT_LOOKUP,
            PeerRead::ORDERED_LOOKUP,
            PeerRead::SOFT_ADDRESS,
        ] {
            let original = read.path.unwrap();
            for (name, control) in controls(read, "0,1,5,10,20,40")? {
                let path = control.path.unwrap();
                assert_eq!(name, format!("reads-{}", path.passes));
                assert_eq!(path.parameters(64), original.parameters(64));
                assert_eq!(
                    PeerRead {
                        path: Some(PathRead {
                            passes: original.passes,
                            ..path
                        }),
                        ..control
                    },
                    read
                );
            }
        }
        assert!(controls(PeerRead::CHART_INDUCTION, "1").is_err());
        for bad in ["", "1,1", "-1", "1,"] {
            assert!(controls(PeerRead::AGREEMENT_LOOKUP, bad).is_err());
        }
        Ok(())
    }
}
