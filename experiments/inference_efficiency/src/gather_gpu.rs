//! Exact lookup of old executed features by operand pair; no labels or answers.
#[kernite_macros::kernels]
pub mod kernels {
    pub unsafe extern "ptx-kernel" fn gather(
        out: *mut f32,
        old: *const f32,
        pairs: *const u32,
        count: u32,
    ) {
        let (block, thread, size, i): (u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(thread, tid.x);
        mov.u32(size, ntid.x);
        mad.lo.u32(i, block, size, thread);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (q, k, j, a, b, row, at): (u32, u32, u32, u32, u32, u32, u32);
        let (p, src, dst): (u64, u64, u64);
        div.u32(q, i, 128);
        rem.u32(k, i, 128);
        mul.lo.u32(j, q, 2);
        mad.wide.u32(p, j, 4, pairs);
        ld.global.u32(a, [p]);
        ld.global.u32(b, [p + 4]);
        mad.lo.u32(row, a, 97, b);
        mad.lo.u32(at, row, 128, k);
        mad.wide.u32(src, at, 4, old);
        mad.wide.u32(dst, i, 4, out);
        let value: f32;
        ld.global.f32(value, [src]);
        st.global.f32([dst], value);
    }
}
