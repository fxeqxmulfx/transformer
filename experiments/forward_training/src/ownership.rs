//! NVIDIA ownership snapshots include graphics processes as well as compute.
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

fn pids(xml: &str) -> Option<Vec<u32>> {
    let mut found = false;
    let mut out = vec![];
    for section in xml.split("<processes>").skip(1) {
        found = true;
        let section = section.split_once("</processes>")?.0;
        for process in section.split("<process_info>").skip(1) {
            let process = process.split_once("</process_info>")?.0;
            out.push(
                process
                    .split_once("<pid>")?
                    .1
                    .split_once("</pid>")?
                    .0
                    .trim()
                    .parse()
                    .ok()?,
            );
        }
    }
    found.then_some(out)
}
pub fn snapshot(directory: &Path, label: &str) -> Value {
    match Command::new("nvidia-smi").args(["-q", "-x"]).output() {
        Ok(result) => {
            let xml = String::from_utf8_lossy(&result.stdout);
            let owners = pids(&xml);
            let path = directory.join(format!("gpu-{label}.xml"));
            let saved = fs::write(&path, result.stdout.as_slice()).is_ok();
            let clean = result.status.success()
                && saved
                && owners
                    .as_ref()
                    .is_some_and(|p| p.iter().all(|&id| id == std::process::id()));
            json!({"label":label,"path":path.file_name(),"pids":owners,"own_pid":std::process::id(),
                "clean_ownership":clean,"snapshot_saved":saved,"query_success":result.status.success(),
                "stderr":String::from_utf8_lossy(&result.stderr)})
        }
        Err(e) => json!({"label":label,"clean_ownership":false,"error":e.to_string()}),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ownership_includes_graphics_and_rejects_unknown_processes() {
        assert_eq!(pids("<processes></processes>"), Some(vec![]));
        assert_eq!(
            pids("<processes><process_info><pid>12</pid><type>G</type></process_info></processes>"),
            Some(vec![12])
        );
        assert_eq!(
            pids("<processes><process_info><pid>N/A</pid></process_info></processes>"),
            None
        );
        assert_eq!(pids("error"), None);
    }
}
