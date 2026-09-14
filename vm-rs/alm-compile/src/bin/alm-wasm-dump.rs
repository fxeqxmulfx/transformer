//! Print the canonical form of a decoded `.wasm`, to diff against
//! `tests/wasmdump.py`.  With `--lower`, print the body each function is
//! lowered to instead, to diff against `tests/lowerdump.py`.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let lower = args.iter().any(|a| a == "--lower");
    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .expect("usage: alm-wasm-dump [--lower] FILE.wasm");

    let data = std::fs::read(path).expect("read the module");
    let mut m = alm_compile::decoder::decode(&data).expect("decode the module");

    if lower {
        for fi in 0..m.functions.len() {
            let ty = &m.types[m.func_type_indices[fi] as usize];
            let num_params = ty.params.len() as u32;
            m.functions[fi] = alm_compile::lower::lower_hard_ops(&m.functions[fi], num_params);
        }
    }

    print!("{}", alm_compile::decoder::dump(&m));

    if lower {
        for (fi, f) in m.functions.iter().enumerate() {
            for (name, count) in alm_compile::lower::check_basic_only(f) {
                println!("unsupported func {fi} {name} {count}");
            }
        }
    }
}
