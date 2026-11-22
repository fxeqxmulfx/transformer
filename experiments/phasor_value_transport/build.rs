//! Compile the upstream trainer at the same revision as the Phasor CPU oracle.
mod build_compact;
mod build_embedding;
use std::{collections::BTreeSet, env, fs, path::Path, process::Command};

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

// The pinned trainer lists only the budgets in its 35-task cohort. The
// larger authenticated GPT catalog declares additional horizons for the
// same update rule. Keep the pinned rate function and percentage cutoffs.
fn catalog_schedule(gpt: &Path, out: &Path) {
    let plan = fs::read(gpt.join("research/current_grokking/plan.json")).unwrap();
    let plan: serde_json::Value = serde_json::from_slice(&plan).unwrap();
    let budgets: BTreeSet<_> = plan["cases"]
        .as_array()
        .expect("missing GPT catalog cases")
        .iter()
        .map(|case| case["max_updates"].as_u64().expect("missing GPT budget"))
        .collect();
    assert!(
        [20_000, 30_000, 100_000]
            .iter()
            .all(|b| budgets.contains(b))
    );
    let budgets: Vec<_> = budgets.into_iter().collect();
    let source =
        fs::read_to_string(gpt.join("research/main_model/wsd_suite_native/src/schedule.rs"))
            .unwrap();
    let original = "[20_000, 30_000, 100_000]";
    assert_eq!(
        source.matches(original).count(),
        1,
        "unexpected pinned budget guard"
    );
    fs::write(
        out.join("schedule.rs"),
        source.replace(original, &format!("{budgets:?}")),
    )
    .unwrap();
}

// Only replace the two final stream operations in the pinned four-stream network.
fn stream_engine(src: &Path) -> String {
    let mut source = check_device_offsets(fs::read_to_string(src.join("engine_mhc.rs")).unwrap());
    let forward = "unsafe {\n            k.collapse::<false>(grid(r * w), self.shared.ptr(), x, w as u32, (r * w) as u32)?;\n        }";
    let backward = "unsafe {\n            k.collapse::<true>(\n                grid(r * w),\n                self.stream_grad.ptr(),\n                self.dshared.ptr(),\n                w as u32,\n                (r * w) as u32,\n            )?;\n        }";
    for (from, to) in [
        (forward, "self.collapse_streams(x, r)?;"),
        (backward, "self.collapse_streams_backward(r)?;"),
    ] {
        assert_eq!(
            source.matches(from).count(),
            1,
            "unexpected final stream operation"
        );
        source = source.replace(from, to);
    }
    build_embedding::separate_input_gradient(&mut source);
    build_compact::hidden_dimensions(&mut source, false);
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
    catalog_schedule(&gpt, &out);
    let src = gpt.join("research/main_model/phase_score_native/src");
    let mut engine = check_device_offsets(fs::read_to_string(src.join("engine.rs")).unwrap());
    build_compact::hidden_dimensions(&mut engine, true);
    let embedding_helpers = build_embedding::extract(&mut engine);
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
    // New trainable coordinates go in the original ordinary AdamW group.
    for (from, to) in [
        (
            "pub phase_optimizer: bool,",
            "pub phase_optimizer: bool,\n    pub head: crate::head::Head,",
        ),
        (
            "pub struct Engine {",
            "pub struct Engine {\n    shared_layer: Option<ValueLayer>,\n    hidden_layer: Option<hidden_layer::HiddenLayer>,\n    context_rotation: Option<context_rotation_layer::ContextRotation>,\n    input_product: Option<product_layer::ProductLayer>,\n    stream_read: Option<crate::learned_read::kernels::Kernels>,\n    supervised_shape: Option<(usize, usize)>,",
        ),
        (
            "let mut engine = Self {",
            "let mut engine = Self {\n            shared_layer: None,\n            hidden_layer: None,\n            context_rotation: None,\n            input_product: None,\n            stream_read: None,\n            supervised_shape: None,",
        ),
        (
            "        Ok(engine)",
            "        engine.enable_output_head()?;\n        Ok(engine)",
        ),
        (
            "self.linear(\n            self.logits.ptr(),\n            self.last.ptr(),\n            0,\n            r,\n            self.config.vocab,\n            w,\n        )?;",
            "self.project_output(tokens, batch, length)?;",
        ),
        (
            "self.linear_back(\n            self.dlast.ptr(),\n            self.dlogits.ptr(),\n            self.last.ptr(),\n            0,\n            r,\n            self.config.vocab,\n            w,\n        )?;",
            "self.output_backward(r, length)?;",
        ),
        (
            "pub fn backward(&self, batch: usize, length: usize)",
            "fn backward_original(&self, batch: usize, length: usize)",
        ),
        // The pinned non-mHC models use a prefix mask. Phasor's four-stream
        // backbone never reads it, and the raw-token model never computes it.
        (
            "            self.seq.prefix(\n                grid(r),\n                self.flags.ptr(),",
            "            if !self.uses_embedding_backbone() { self.seq.prefix(\n                grid(r),\n                self.flags.ptr(),",
        ),
        (
            "            self.ops.embed(",
            "            }\n            self.ops.embed(",
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
    let layout = build_compact::layout(layout);
    let initialize = "p.resize(total_count(cfg), 0.);";
    assert_eq!(layout.matches(initialize).count(), 1);
    let layout = layout.replace(
        initialize,
        "p.resize(total_count(cfg), 0.);\n    cfg.head.initialize(cfg, seed, &mut p);",
    );
    let offset = "score_offset(cfg) + score_count(cfg)";
    assert_eq!(layout.matches(offset).count(), 1);
    fs::write(
        out.join("layout.rs"),
        layout.replace(
            offset,
            "crate::architecture::start(cfg) + crate::architecture::parameters(cfg)",
        ),
    )
    .unwrap();
    fs::write(out.join("engine_mhc.rs"), stream_engine(&src)).unwrap();
    engine.push_str(&embedding_helpers);
    for file in [
        "src/value_layer.rs",
        "src/stream_layer.rs",
        "src/embedding_input.rs",
    ] {
        let extension = Path::new(file).canonicalize().unwrap();
        engine.push_str(&format!("\ninclude!({extension:?});\n"));
        println!("cargo:rerun-if-changed={file}");
    }
    for name in [
        "hidden_layer",
        "shared_bilinear_layer",
        "product_layer",
        "context_rotation_layer",
        "associative_layer",
    ] {
        let file = format!("src/{name}.rs");
        let path = Path::new(&file).canonicalize().unwrap();
        engine.push_str(&format!("#[path={path:?}] mod {name};\n"));
        println!("cargo:rerun-if-changed={file}");
    }
    fs::write(out.join("engine.rs"), engine).unwrap();
    let geometry = gpt.join("research/complex_geometry_suite");
    let mut mods = String::new();
    // The pinned stable-loss tests use the scalar oracle declared inline in
    // their binary. Include that same oracle when compiling native test targets.
    let main = src.join("main.rs");
    println!("cargo:rerun-if-changed={}", main.display());
    let text = fs::read_to_string(main).unwrap();
    let marker = "#[cfg(test)]\nmod loss_precision {";
    let (_, tail) = text.split_once(marker).expect("missing pinned loss oracle");
    let (body, _) = tail
        .split_once("\n}\n")
        .expect("unterminated pinned loss oracle");
    mods.push_str(&format!("{marker}{body}\n}}\n"));
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
        (out.clone(), "engine layout schedule"),
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
