//! Use a single configured feed-forward width in all pinned offsets, allocations
//! and kernels. Preserve the pinned initializer as the source of retained weights.
fn replace(source: &mut String, old: &str, new: &str, count: usize) {
    assert_eq!(
        source.matches(old).count(),
        count,
        "unexpected upstream source for {old}"
    );
    *source = source.replace(old, new);
}

pub fn hidden_dimensions(source: &mut String, engine: bool) {
    replace(
        source,
        "layout::hidden(w, \"\")",
        "crate::compact::hidden(&self.config)",
        2,
    );
    if engine {
        replace(
            source,
            "layout::hidden(w, &config.architecture)",
            "crate::compact::hidden(&config)",
            1,
        );
        replace(
            source,
            "layout::hidden(w, &self.config.architecture)",
            "crate::compact::hidden(&self.config)",
            1,
        );
    }
}

pub fn layout(mut source: String) -> String {
    replace(
        &mut source,
        "count(cfg.width, cfg.layers, cfg.vocab)",
        "crate::compact::base_count(cfg)",
        2,
    );
    replace(
        &mut source,
        "let h = hidden(w, &cfg.architecture);",
        "let h = crate::compact::hidden(cfg);",
        1,
    );
    let init = "    let mut p = initialize(cfg.width, cfg.layers, cfg.vocab, seed);";
    replace(
        &mut source,
        init,
        &format!(
            "    if cfg.head == crate::head::Head::CompactStream {{ return crate::compact::initialize(cfg, seed); }}\n{init}"
        ),
        1,
    );
    source
}
