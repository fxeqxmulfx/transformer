//! Copy dense subranges across the Kernite device/host boundary.
#[kernite_macros::kernels]
pub mod kernels {
    pub unsafe extern "ptx-kernel" fn copy(out: *mut f32, input: *const f32, count: u32) {
        let (i, b, t, n): (u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
        let skip: pred;
        setp.ge.u32(skip, i, count);
        if skip {
            bra(done);
        }
        let (src, dst): (u64, u64);
        mad.wide.u32(src, i, 4, input);
        mad.wide.u32(dst, i, 4, out);
        let val: f32;
        ld.global.f32(val, [src]);
        st.global.f32([dst], val);
        label(done);
    }
}
