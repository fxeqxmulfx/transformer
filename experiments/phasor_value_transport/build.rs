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

// Kernite 0.6 makes device pointer offsets fallible. The pinned engine only
// takes offsets into its parameter, gradient and optimizer-state buffers.
fn check_device_offsets(mut source: String) -> String {
    let offsets = source.matches(".at(").count();
    let checked = ".ok_or(\"CUDA parameter layout exceeds buffer\")?";
    for argument in [
        "offset",
        "layer.score_offset",
        "layer.offset + 3 * w * w",
        "layout::frequency_offset(&self.config)",
        "router.offset",
        "router.phase_offset",
    ] {
        let call = format!(".at({argument})");
        source = source.replace(&call, &format!("{call}{checked}"));
    }
    assert_eq!(source.matches(checked).count(), offsets);
    source
}

fn main() {
    let lock = include_str!("Cargo.lock");
    let version = lock
        .split_once("name = \"kernite-cuda\"\nversion = \"")
        .and_then(|(_, rest)| rest.split_once('"'))
        .expect("missing kernite-cuda version in Cargo.lock")
        .0;
    println!("cargo:rustc-env=KERNITE_CUDA_VERSION={version}");
    let gpt = Path::new("../../.lake/reproduction/gpt")
        .canonicalize()
        .unwrap();
    pin(&gpt, "5a37c515639f23037d191b0659a5a0853984328d", "research");
    let out = std::path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let src = gpt.join("research/main_model/phase_score_native/src");
    let mut engine = check_device_offsets(fs::read_to_string(src.join("engine.rs")).unwrap());
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
    // The ordinary parameter group includes the new hidden tables. Keep the
    // optimizer, cross-entropy and schedule at their pinned implementations.
    for (from, to) in [
        (
            "pub phase_optimizer: bool,",
            "pub phase_optimizer: bool,\n    pub shared_values: bool,",
        ),
        (
            "pub struct Engine {",
            "pub struct Engine {\n    shared_layer: Option<ValueLayer>,",
        ),
        (
            "let mut engine = Self {",
            "let mut engine = Self {\n            shared_layer: None,",
        ),
        (
            "        Ok(engine)",
            "        engine.enable_shared_values()?;\n        Ok(engine)",
        ),
        (
            "        let normalizer = labels",
            "        self.project_shared_values(tokens, batch, length)?;\n        let normalizer = labels",
        ),
        (
            "pub fn backward(&self, batch: usize, length: usize)",
            "fn backward_original(&self, batch: usize, length: usize)",
        ),
    ] {
        assert_eq!(
            engine.matches(from).count(),
            1,
            "unexpected engine source for {from}"
        );
        engine = engine.replace(from, to);
    }
    let layout = fs::read_to_string(src.join("layout.rs")).unwrap();
    let initialize = "p.resize(total_count(cfg), 0.);";
    assert_eq!(layout.matches(initialize).count(), 1);
    let layout = layout.replace(
        initialize,
        "p.resize(total_count(cfg), 0.);\n    crate::pair_memory::initialize_gains(cfg, &mut p);",
    );
    let offset = "score_offset(cfg) + score_count(cfg)";
    assert_eq!(layout.matches(offset).count(), 1);
    fs::write(
        out.join("layout.rs"),
        layout.replace(
            offset,
            "crate::pair_memory::start(cfg) + crate::pair_memory::parameters(cfg)",
        ),
    )
    .unwrap();
    fs::write(
        out.join("engine_mhc.rs"),
        check_device_offsets(fs::read_to_string(src.join("engine_mhc.rs")).unwrap()),
    )
    .unwrap();
    let extension = Path::new("src/value_layer.rs").canonicalize().unwrap();
    engine.push_str(&format!("\ninclude!({extension:?});\n"));
    println!("cargo:rerun-if-changed=src/value_layer.rs");
    fs::write(out.join("engine.rs"), engine).unwrap();
    let geometry = gpt.join("research/complex_geometry_suite");
    let mut mods = String::new();
    for (dir, names) in [
        (
            src,
            "attention base_ops centered_attention embedding_backward mhc optimizer optimizer_kernels ordinary_attention ordinary_forward phase_mhc phase_score phase_score_forward pipeline pipeline_host placement precision rotary stable_attention stable_chain stable_loss tensor tiled_transpose",
        ),
        (
            geometry.join("native/src"),
            "blake complex data gemm metrics pcg random sequence_ops",
        ),
        (geometry.join("diagnostic/src"), "plain_product"),
        (
            gpt.join("research/main_model/wsd_suite_native/src"),
            "schedule",
        ),
        (out.clone(), "engine layout"),
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
