//! Generate the frozen row-tiled specialization from the pinned Kernite kernel.
//! All scalar score, normalization and value instructions come from the source.
use std::{fs, path::Path};

fn replace_once(text: &mut String, source: &str, replacement: &str) {
    assert_eq!(
        text.matches(source).count(),
        1,
        "pinned phase forward changed: {source}"
    );
    *text = text.replace(source, replacement);
}

pub fn kernel(source: &Path, out: &Path) {
    let source = source.join("phase_score_forward.rs");
    let mut kernel = fs::read_to_string(&source).unwrap();
    crate::build_residual::wide_head(&mut kernel);
    for (old, new) in [
        ("        imaginary: *mut f32,\n", ""),
        (
            "        head_count: u32,",
            "        head_count: u32,\n        first_head: u32,",
        ),
        (
            "        rem.u32(h, id, heads);",
            "        mul.lo.u32(scorebase, id, length);\n        add.u32(id, id, first_head);\n        rem.u32(h, id, heads);",
        ),
        (
            "        mad.lo.u32(scorebase, b, heads, h);\n        mad.lo.u32(scorebase, scorebase, length, t);\n        mul.lo.u32(scorebase, scorebase, length);\n",
            "",
        ),
        ("            store(imaginary, at, im);\n", ""),
        (
            "        if lead {\n            store(prob, at, p);\n        }\n",
            "",
        ),
    ] {
        replace_once(&mut kernel, old, new);
    }
    fs::write(out.join("phase_score_tiles.rs"), kernel).unwrap();
    println!("cargo:rerun-if-changed={}", source.display());
}

pub fn dispatch(engine: &mut String) {
    replace_once(
        engine,
        "    score_kernels: Option<(",
        "    score_tiles: Option<crate::phase_score_tiles::kernels::Kernels>,\n    score_kernels: Option<(",
    );
    replace_once(
        engine,
        "            score_kernels: if",
        "            score_tiles: if scoring_only && layout::phase_score(&config) { Some(crate::phase_score_tiles::kernels::Kernels::load(&cuda)?) } else { None },\n            score_kernels: if",
    );
    let start = "                        forward.forward::<$d>(";
    let end = "                    } else if crate::placement::FUSED_ATTENTION {";
    let range = engine
        .split_once(start)
        .unwrap()
        .1
        .split_once(end)
        .unwrap()
        .0;
    let retained = format!("{start}{range}");
    let tiled = r#"                        if let Some(tiles) = &self.score_tiles {
                            for first in (0..r * h).step_by(SCORE_TILE_ROWS) {
                                let count = (r * h - first).min(SCORE_TILE_ROWS);
                                tiles.forward::<$d>(
                                    Grid::new(count.div_ceil(if $d == 16 { 8 } else { 4 }) as u32, 128),
                                    layer.att.ptr(), layer.prob.ptr(), layer.qkr.ptr(), layer.score_table.ptr(),
                                    h as u32, length as u32, count as u32, first as u32,
                                )?;
                            }
                        } else {
"#;
    replace_once(
        engine,
        &retained,
        &format!("{tiled}{retained}                        }}\n"),
    );
}
