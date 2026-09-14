//! Print the canonical form of a decoded `.wasm`, to diff against
//! `tests/wasmdump.py`.

fn main() {
    let path = std::env::args().nth(1).expect("usage: alm-wasm-dump FILE.wasm");
    let data = std::fs::read(&path).expect("read the module");
    let m = alm_compile::decoder::decode(&data).expect("decode the module");
    print!("{}", alm_compile::decoder::dump(&m));
}
