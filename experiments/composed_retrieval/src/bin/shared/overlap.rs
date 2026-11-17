//! Verify duplicate fits before counting overlapping original cases once.
use composed_retrieval::{Result, data};
use learned_retrieval::cohort::{raw_file, read};
use std::{fs, path::Path};

pub fn check_overlap(a: &Path, b: &Path, id: &str, p: &data::Protocol) -> Result<usize> {
    let mut raw_files = 0;
    for arm in ["original", "repeat"] {
        let (a, b) = (
            a.join(arm).join("run").join(id),
            b.join(arm).join("run").join(id),
        );
        let (ra, rb) = (
            read(&a.join("scores/report.json"))?,
            read(&b.join("scores/report.json"))?,
        );
        if ra["cells"] != rb["cells"]
            || ra["metric"] != rb["metric"]
            || ra["target"] != rb["target"]
            || ra["selection"] != rb["selection"]
        {
            return Err("overlap original metrics or criteria differ".into());
        }
        for mode in &p.modes {
            for seed in &p.seeds {
                let cell = format!("{mode}-{seed}");
                let (x, y) = (
                    read(&a.join(&cell).join("receipt.json"))?,
                    read(&b.join(&cell).join("receipt.json"))?,
                );
                if x["data_fingerprints"] != y["data_fingerprints"]
                    || x["training"] != y["training"]
                    || fs::read(a.join(&cell).join("model.json"))?
                        != fs::read(b.join(&cell).join("model.json"))?
                {
                    return Err("overlap model, optimizer or source differs".into());
                }
                for phase in ["initial", "final"] {
                    let key = format!("{cell}-{phase}");
                    let splits = ra["cells"][&key]["splits"]
                        .as_object()
                        .ok_or("overlap split roster")?;
                    for split in splits.keys() {
                        if fs::read(raw_file(&a.join("scores"), &key, split))?
                            != fs::read(raw_file(&b.join("scores"), &key, split))?
                        {
                            return Err("overlap raw logits differ".into());
                        }
                        raw_files += 1;
                    }
                }
            }
        }
    }
    Ok(raw_files)
}
