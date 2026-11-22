//! Copy dense subranges across the Kernite device/host boundary.
#[kernite_macros::kernels]
pub mod kernels {
    pub unsafe extern "ptx-kernel" fn clear(out: *mut f32, count: u32) {
        let (i, block, lane, size): (u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        mov.u32(size, ntid.x);
        mad.lo.u32(i, block, size, lane);
        let stop: pred;
        setp.ge.u32(stop, i, count);
        if stop {
            ret();
        }
        let (address, zero): (u64, f32);
        mad.wide.u32(address, i, 4, out);
        mov.f32(zero, 0.0);
        st.global.f32([address], zero);
    }

    /// Every index occurs once; a dense row has already accumulated all visits.
    pub unsafe extern "ptx-kernel" fn scatter_rows(
        out: *mut f32,
        input: *const f32,
        rows: *const u32,
        width: u32,
        count: u32,
    ) {
        let (i, block, lane, size): (u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        mov.u32(size, ntid.x);
        mad.lo.u32(i, block, size, lane);
        let stop: pred;
        setp.ge.u32(stop, i, count);
        if stop {
            ret();
        }
        let (row, col, target, offset): (u32, u32, u32, u32);
        div.u32(row, i, width);
        rem.u32(col, i, width);
        let (address, from, to): (u64, u64, u64);
        mad.wide.u32(address, row, 4, rows);
        ld.global.u32(target, [address]);
        mad.lo.u32(offset, target, width, col);
        mad.wide.u32(from, i, 4, input);
        mad.wide.u32(to, offset, 4, out);
        let value: f32;
        ld.global.f32(value, [from]);
        st.global.f32([to], value);
    }

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
