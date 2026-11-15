//! Preserve source and compact findings before user-authorized removal of old
//! experiment outputs. This tool never deletes anything. Copies are hashed;
//! source contents are deduplicated, and every omitted payload is identified.
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufReader, BufWriter, Read, Write},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
mod compact;
mod streaming;

fn digest(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(64);
    for b in Sha256::digest(bytes) {
        write!(out, "{b:02x}").unwrap();
    }
    out
}
fn hash(path: &Path) -> Result<String> {
    use std::fmt::Write;
    let mut file = fs::File::open(path)?;
    let mut buffer = [0u8; 65536];
    let mut hash = Sha256::new();
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let mut out = String::with_capacity(64);
    for b in hash.finalize() {
        write!(out, "{b:02x}").unwrap();
    }
    Ok(out)
}
fn code(path: &Path) -> bool {
    let extension = path.extension().and_then(|x| x.to_str()).unwrap_or("");
    let name = path.file_name().and_then(|x| x.to_str()).unwrap_or("");
    matches!(
        extension,
        "rs" | "lean"
            | "py"
            | "sh"
            | "toml"
            | "lock"
            | "md"
            | "yaml"
            | "yml"
            | "js"
            | "ts"
            | "tsx"
            | "css"
            | "html"
            | "c"
            | "h"
            | "cpp"
            | "hpp"
    ) || matches!(
        name,
        "Makefile" | "Dockerfile" | "lean-toolchain" | "requirements.txt" | ".gitignore"
    ) || (extension == "json"
        && path.components().any(|p| p.as_os_str() == "source")
        && !path.components().any(|p| {
            matches!(
                p.as_os_str().to_str(),
                Some("runs" | "target" | "results" | "outputs")
            )
        }))
        || matches!(
            name,
            "protocol.json" | "suite.json" | "plan.json" | "cases.json"
        )
}
fn summary(path: &Path) -> bool {
    let name = path.file_name().and_then(|x| x.to_str()).unwrap_or("");
    name.ends_with(".json")
        && [
            "summary",
            "report",
            "result",
            "metrics",
            "verified",
            "verification",
            "measurements",
            "inventory",
            "metadata",
            "replay",
            "original35",
            "baseline",
            "history",
        ]
        .iter()
        .any(|key| name.contains(key))
}
fn provenance(path: &Path) -> bool {
    let name = path.file_name().and_then(|x| x.to_str()).unwrap_or("");
    name.ends_with(".sha256")
        || name.ends_with(".commit")
        || name.ends_with(".resources")
        || name.ends_with(".exit")
        || name.ends_with(".version")
        || name == "manifest.json"
}
struct Archive {
    output: PathBuf,
    index: BufWriter<fs::File>,
    files: u64,
    source_files: u64,
    summaries: u64,
    skipped: u64,
    summary_cache: std::collections::BTreeMap<String, String>,
}
impl Archive {
    fn put(
        &mut self,
        path: &Path,
        kind: &str,
        bytes: &[u8],
        original_hash: String,
        original_bytes: u64,
        mode: u32,
    ) -> Result<()> {
        let sha = digest(bytes);
        let blob = self.output.join("objects").join(&sha);
        if blob.exists() {
            if hash(&blob)? != sha {
                return Err("retained object hash mismatch".into());
            }
        } else {
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&blob)?;
            f.write_all(bytes)?;
            f.sync_all()?;
        }
        if kind != "source" {
            let view = self.output.join("summaries").join(path.strip_prefix("/")?);
            fs::create_dir_all(view.parent().ok_or("retained summary has no parent")?)?;
            if !view.exists() {
                fs::hard_link(&blob, &view)?;
            }
        }
        serde_json::to_writer(
            &mut self.index,
            &json!({"original_path":path,"kind":kind,"original_sha256":original_hash,
            "original_bytes":original_bytes,"unix_mode":mode,"retained_sha256":sha,"retained_bytes":bytes.len()}),
        )?;
        self.index.write_all(b"\n")?;
        self.files += 1;
        Ok(())
    }
    fn visit(&mut self, path: &Path) -> Result<()> {
        let metadata = fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() {
            serde_json::to_writer(
                &mut self.index,
                &json!({"original_path":path,"kind":"symlink_not_followed","target":fs::read_link(path)?}),
            )?;
            self.index.write_all(b"\n")?;
            return Ok(());
        }
        if metadata.is_dir() {
            if matches!(
                path.file_name().and_then(|x| x.to_str()),
                Some(
                    "target"
                        | ".git"
                        | ".lake"
                        | ".discrtree"
                        | "__pycache__"
                        | ".venv"
                        | "node_modules"
                )
            ) {
                return Ok(());
            }
            let mut entries: Vec<_> = fs::read_dir(path)?
                .map(|e| e.map(|x| x.path()))
                .collect::<std::io::Result<_>>()?;
            entries.sort();
            for child in entries {
                self.visit(&child)?;
            }
            return Ok(());
        }
        if !metadata.is_file() {
            return Err("unsupported non-regular experiment entry".into());
        }
        let size = metadata.len();
        let mode = metadata.permissions().mode();
        if code(path) {
            if size > 64 * 1024 * 1024 {
                return Err(format!(
                    "source file too large for safe retention: {}",
                    path.display()
                )
                .into());
            }
            let bytes = fs::read(path)?;
            let sha = digest(&bytes);
            self.put(path, "source", &bytes, sha, size, mode)?;
            self.source_files += 1;
        } else if provenance(path) && size <= 8 * 1024 * 1024 {
            let bytes = fs::read(path)?;
            let sha = digest(&bytes);
            self.put(path, "historical_provenance", &bytes, sha, size, mode)?;
        } else if summary(path) {
            if size > 1024 * 1024 * 1024 {
                return Err(format!("summary source too large: {}", path.display()).into());
            }
            let sha = hash(path)?;
            if let Some(saved) = self.summary_cache.get(&sha) {
                let bytes = fs::read(self.output.join("objects").join(saved))?;
                self.put(path, "compact_summary", &bytes, sha, size, mode)?;
                self.summaries += 1;
                return Ok(());
            }
            let value = streaming::read(BufReader::new(fs::File::open(path)?))?;
            let retained = json!({"format":"compact_experiment_summary_v1","source_sha256":sha,
                "raw_replay_evidence_will_be_removed":true,"summary":value});
            let bytes = serde_json::to_vec(&retained)?;
            self.summary_cache.insert(sha.clone(), digest(&bytes));
            self.put(path, "compact_summary", &bytes, sha, size, mode)?;
            self.summaries += 1;
        } else {
            self.skipped += 1;
        }
        Ok(())
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [list, output] = args.as_slice() else {
        return Err("usage: retain-experiment-summaries ROOT_LIST NEW_ARCHIVE".into());
    };
    let output = PathBuf::from(output);
    fs::create_dir(&output)?;
    fs::create_dir(output.join("objects"))?;
    let index = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("index.jsonl"))?,
    );
    let mut archive = Archive {
        output: output.clone(),
        index,
        files: 0,
        source_files: 0,
        summaries: 0,
        skipped: 0,
        summary_cache: std::collections::BTreeMap::new(),
    };
    for line in fs::read_to_string(list)?.lines() {
        let path = PathBuf::from(line).canonicalize()?;
        eprintln!("retaining {}", path.display());
        if output.canonicalize()?.starts_with(&path) {
            return Err("retention destination lies inside an input tree".into());
        }
        archive.visit(&path)?;
    }
    archive.index.flush()?;
    let report = json!({"retained_files":archive.files,"source_files":archive.source_files,"summaries":archive.summaries,
        "skipped_generated_files":archive.skipped,"index_sha256":hash(&output.join("index.jsonl"))?,"deletions_performed":0});
    fs::write(
        output.join("archive-summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
