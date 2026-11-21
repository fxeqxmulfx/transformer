//! Compile the upstream trainer at the same revision as the Phasor CPU oracle.
use std::{env, fs, path::Path, process::Command};

fn pin(dir: &Path, revision: &str, tree: &str) {
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .unwrap()
    };
    assert_eq!(
        String::from_utf8(git(&["rev-parse", "HEAD"]).stdout)
            .unwrap()
            .trim(),
        revision
    );
    assert!(
        git(&["diff", "--quiet", revision, "--", tree])
            .status
            .success()
    );
    println!("cargo:rerun-if-changed={}", dir.join(".git/HEAD").display());
    println!("cargo:rerun-if-changed={}", dir.join(tree).display());
}

fn main() {
    let gpt = Path::new("../../.lake/reproduction/gpt")
        .canonicalize()
        .unwrap();
    let kgen = Path::new("../../.lake/reproduction/kgen")
        .canonicalize()
        .unwrap();
    pin(&gpt, "5a37c515639f23037d191b0659a5a0853984328d", "research");
    pin(&kgen, "a1b6bc2c956e1a5f35960e26b209e1b6c213096c", "crates");
    let out = std::path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let src = gpt.join("research/main_model/phase_score_native/src");
    let mut engine = fs::read_to_string(src.join("engine.rs")).unwrap();
    // Only the FP32 reference dispatches entirely to the upstream SGEMM.
    // Do not load unused Ampere tensor modules on a Pascal GPU in that mode.
    for (from, to) in [
        (
            "tensor: crate::tensor::kernels::Kernels,",
            "tensor: Option<crate::tensor::kernels::Kernels>,",
        ),
        (
            "pipeline: crate::pipeline_host::Pipeline,",
            "pipeline: Option<crate::pipeline_host::Pipeline>,",
        ),
        (
            "tensor: crate::tensor::kernels::Kernels::load(&cuda)?,",
            "tensor: (!crate::precision::FP32_AFFINE_REFERENCE).then(|| crate::tensor::kernels::Kernels::load(&cuda)).transpose()?,",
        ),
        (
            "pipeline: {",
            "pipeline: if crate::precision::FP32_AFFINE_REFERENCE { None } else {",
        ),
        (
            "crate::pipeline_host::Pipeline::new(&cuda, operand, operand, partial)?",
            "Some(crate::pipeline_host::Pipeline::new(&cuda, operand, operand, partial)?)",
        ),
        (
            ".multiply_tf32(&self.tensor,",
            ".as_ref().unwrap().multiply_tf32(self.tensor.as_ref().unwrap(),",
        ),
        (
            ".multiply::<{ crate::precision::BF16 }>",
            ".as_ref().unwrap().multiply::<{ crate::precision::BF16 }>",
        ),
    ] {
        assert_eq!(
            engine.matches(from).count(),
            1,
            "unexpected upstream source for {from}"
        );
        engine = engine.replace(from, to);
    }
    let mhc = src.join("engine_mhc.rs");
    engine = engine.replace(
        "include!(\"engine_mhc.rs\");",
        &format!("include!({mhc:?});"),
    );
    let extension = Path::new("src/value_layer.rs").canonicalize().unwrap();
    engine.push_str(&format!("\ninclude!({extension:?});\n"));
    println!("cargo:rerun-if-changed=src/value_layer.rs");
    fs::write(out.join("engine.rs"), engine).unwrap();
    let geometry = gpt.join("research/complex_geometry_suite");
    let mut mods = String::new();
    for (dir, names) in [
        (
            src,
            "attention base_ops centered_attention embedding_backward layout mhc optimizer optimizer_kernels ordinary_attention ordinary_forward phase_mhc phase_score phase_score_forward pipeline pipeline_host placement precision rotary stable_attention stable_chain stable_loss tensor tiled_transpose",
        ),
        (
            geometry.join("native/src"),
            "complex gemm random sequence_ops",
        ),
        (geometry.join("diagnostic/src"), "plain_product"),
        (
            gpt.join("research/main_model/wsd_suite_native/src"),
            "schedule",
        ),
        (out.clone(), "engine"),
    ] {
        for name in names.split_whitespace() {
            let path = dir.join(format!("{name}.rs"));
            mods.push_str(&format!(
                "#[allow(dead_code, clippy::all)]\n#[path={path:?}] mod {name};\n"
            ));
        }
    }
    fs::write(out.join("upstream.rs"), mods).unwrap();
}
