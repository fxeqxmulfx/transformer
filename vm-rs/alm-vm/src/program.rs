//! Reading a token file, and the reference it is checked against.
//!
//! Ported from the argument handling of `model/transformer.cpp:215-275`.

use alm_model::RawModel;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// A program: the token ids the model is primed with, and the reference run.
pub struct Program {
    pub name: String,
    pub ids: Vec<usize>,
    pub reference: Option<Vec<usize>>,
}

/// The file `prog.txt` keeps its reference run in `prog_ref.txt`.
fn reference_path(path: &Path) -> PathBuf {
    let mut with_ref = path.to_path_buf();
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = path.extension().map(|s| s.to_string_lossy().into_owned());
    with_ref.set_file_name(match ext {
        Some(e) => format!("{stem}_ref.{e}"),
        None => format!("{stem}_ref"),
    });
    with_ref
}

/// Split on whitespace and look every token up, failing on the first unknown
/// one — an unknown token is a compiler bug, not an input to be tolerated.
fn ids_of(model: &RawModel, text: &str, whence: &Path) -> io::Result<Vec<usize>> {
    text.split_whitespace()
        .map(|t| {
            let t = if t == "<sp>" { " " } else { t };
            model.token_id(t).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown token {t:?} in {}", whence.display()),
                )
            })
        })
        .collect()
}

impl Program {
    /// `args` is the C++ `--args=STR`: everything after the last `}` is
    /// dropped and the string is re-appended one character per token, with a
    /// `00` terminator.
    pub fn load(model: &RawModel, path: &Path, args: Option<&str>) -> io::Result<Program> {
        let mut ids = ids_of(model, &fs::read_to_string(path)?, path)?;

        if let Some(args) = args {
            let close = model.token_id("}").and_then(|b| ids.iter().rposition(|&i| i == b));
            if let Some(at) = close {
                ids.truncate(at + 1);
            }
            for ch in args.chars() {
                match model.token_id(&ch.to_string()) {
                    Some(id) => ids.push(id),
                    None => eprintln!("warning: char {ch:?} is not in the vocabulary"),
                }
            }
            if let Some(nul) = model.token_id("00") {
                ids.push(nul);
            }
        }

        let rp = reference_path(path);
        let reference = match fs::read_to_string(&rp) {
            Ok(text) => Some(ids_of(model, &text, &rp)?),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };

        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        Ok(Program { name, ids, reference })
    }
}
