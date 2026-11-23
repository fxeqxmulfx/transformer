//! Release the buffers exclusive to backward while scoring frozen weights.
//! The pinned forward passes some derivative buffers to specialized kernels.
//! Every buffer named by a forward call keeps its full original allocation.
const LAYER: &[&str] = &[
    "score_parts",
    "dqnorm",
    "dqkr",
    "gd",
    "dv",
    "dqkv",
    "datt",
    "dap",
    "dmid",
    "dfc",
    "dhidden",
    "dfp",
    "dx",
    "branch",
];
const ROUTER: &[&str] = &["dlogits", "dprojected", "dnormalized", "dinput"];

fn once(s: &mut String, old: &str, new: &str) {
    assert_eq!(
        s.matches(old).count(),
        1,
        "pinned allocation changed: {old}"
    );
    *s = s.replace(old, new);
}

fn between<'a>(s: &'a str, begin: &str, end: &str) -> &'a str {
    s.split_once(begin)
        .expect(begin)
        .1
        .split_once(end)
        .expect(end)
        .0
}

fn no_use(body: &str, owner: &str, fields: &[&str]) {
    let identifier = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    for (at, _) in body.match_indices(owner) {
        if at > 0 && identifier(body.as_bytes()[at - 1]) {
            continue;
        }
        let after = body[at + owner.len()..].trim_start();
        let Some(field) = after.strip_prefix('.') else {
            continue;
        };
        let field = field.trim_start();
        let length = field.bytes().take_while(|&b| identifier(b)).count();
        let name = &field[..length];
        assert!(
            !fields.contains(&name),
            "pinned forward accesses {owner}.{name}"
        );
    }
}
fn prune(text: &mut String, fields: &[&str], frozen: &str) {
    for &name in fields {
        let start = format!("            {name}: cuda.alloc(");
        let size = between(text, &start, ")?,").to_owned();
        let allocation = format!("{start}{size})?,");
        let smaller = format!("{start}if {frozen} {{1}} else {{{size}}})?,");
        once(text, &allocation, &smaller);
    }
}

pub fn prepare(engine: &mut String, router: &mut String) {
    let attention = between(
        engine,
        "    fn attention_forward(",
        "    fn attention_backward(",
    );
    let forward = between(engine, "    pub fn forward(", "    fn backward_original(");
    let mhc = between(
        router,
        "    fn forward_mhc(",
        "    fn backward_embedding_layers(",
    );
    for section in [attention, forward, mhc] {
        no_use(section, "layer", LAYER);
    }
    for owner in ["a", "f"] {
        no_use(mhc, owner, ROUTER);
    }
    for (from, to) in [
        ("    fn write_route(", "    fn write_route_backward("),
        ("    fn route(", "    fn route_backward("),
    ] {
        no_use(between(router, from, to), "router", ROUTER);
    }
    once(
        router,
        "cfg: &Config, sublayer: usize) -> Result<Self>",
        "cfg: &Config, sublayer: usize, frozen: bool) -> Result<Self>",
    );
    prune(router, ROUTER, "frozen");
    let (layers, rest) = engine.split_once("pub struct Engine {").unwrap();
    let mut layers = layers.to_owned();
    prune(&mut layers, LAYER, "shared.is_some()");
    for index in ["2 * index", "2 * index + 1"] {
        once(
            &mut layers,
            &format!("Router::new(cuda, cfg, {index})?"),
            &format!("Router::new(cuda, cfg, {index}, shared.is_some())?"),
        );
    }
    *engine = format!("{layers}pub struct Engine {{{rest}");
    let layer = LAYER
        .iter()
        .map(|v| format!("layer.{v}.len()"))
        .collect::<Vec<_>>()
        .join(" + ");
    let router = ROUTER
        .iter()
        .map(|v| format!("route.{v}.len()"))
        .collect::<Vec<_>>()
        .join(" + ");
    let marker = "    #[cfg(test)]\n    pub(crate) fn attention_score_storage";
    let inspect = format!(
        "    #[cfg(test)]\n    pub(crate) fn adjoint_storage(&self) -> (usize,usize) {{
        let mut cells=0;
        let mut arrays=0;
        for layer in &self.layers {{
            cells+={layer};
            arrays+={};
            if let Some(routes)=&layer.routing {{for route in routes {{
                cells+={router}; arrays+={};
            }}}}
        }}
        (cells,arrays)
    }}\n\n{marker}",
        LAYER.len(),
        ROUTER.len()
    );
    once(engine, marker, &inspect);
}
