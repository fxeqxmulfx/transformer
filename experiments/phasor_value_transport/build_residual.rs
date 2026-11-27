//! Extend the pinned phase-score engine with its ordinary PeriNorm residual.
use std::{
    fs,
    path::{Path, PathBuf},
};

const WIDE_HEAD_KERNELS: &[&str] = &[
    "attention",
    "centered_attention",
    "ordinary_attention",
    "ordinary_forward",
    "phase_score",
    "phase_score_forward",
    "stable_attention",
];

fn replace(text: &mut String, old: &str, new: &str, count: usize) {
    assert_eq!(
        text.matches(old).count(),
        count,
        "pinned source changed: {old}"
    );
    *text = text.replace(old, new);
}

pub fn wide_head(kernel: &mut String) {
    if cfg!(feature = "v1-residual") {
        let old = "#[instances(16, 32, 128)]";
        let count = kernel.matches(old).count();
        assert!(count > 0, "pinned kernel has no head instances");
        replace(kernel, old, "#[instances(16, 32, 64, 128)]", count);
    }
}

pub fn engine(engine: &mut String) {
    if !cfg!(feature = "v1-residual") {
        return;
    }
    replace(
        engine,
        "                \"complex-phase-score-lerope\",",
        "                \"complex-phase-score-lerope\",\n                crate::architecture::V1_RESIDUAL_ARCH,",
        1,
    );
    replace(
        engine,
        "![16, 32, 128].contains",
        "![16, 32, 64, 128].contains",
        1,
    );
    replace(
        engine,
        "            32 => dispatch!(32),",
        "            32 => dispatch!(32),\n            64 => dispatch!(64),",
        2,
    );
}

pub fn layout(layout: &mut String) {
    if !cfg!(feature = "v1-residual") {
        return;
    }
    let phase_score = "            | \"complex-phase-score-lerope\"";
    for next in ["\npub fn is_mhc", "pub fn log_positions"] {
        let old = format!("{phase_score}\n    )\n}}\n{next}");
        let new = format!(
            "{phase_score}\n            | crate::architecture::V1_RESIDUAL_ARCH\n    )\n}}\n{next}"
        );
        replace(layout, &old, &new, 1);
    }
    replace(
        layout,
        "cfg.architecture == \"complex-phase-score-lerope\"",
        "matches!(cfg.architecture.as_str(), \"complex-phase-score-lerope\" | crate::architecture::V1_RESIDUAL_ARCH)",
        1,
    );
}

pub fn module(dir: &Path, upstream: &Path, out: &Path, name: &str) -> PathBuf {
    let file = dir.join(format!("{name}.rs"));
    if !cfg!(feature = "v1-residual") || dir != upstream {
        return file;
    }
    if name != "optimizer" && !WIDE_HEAD_KERNELS.contains(&name) {
        return file;
    }
    println!("cargo:rerun-if-changed={}", file.display());
    let mut source = fs::read_to_string(&file).unwrap();
    if name == "optimizer" {
        replace(
            &mut source,
            "|| layout::is_mhc(&e.config)",
            "|| layout::is_mhc(&e.config) || e.config.architecture == crate::architecture::V1_RESIDUAL_ARCH",
            1,
        );
    } else {
        wide_head(&mut source);
    }
    let generated = out.join(format!("{name}.rs"));
    fs::write(&generated, source).unwrap();
    generated
}
