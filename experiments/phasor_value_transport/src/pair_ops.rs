//! Add selected memory embeddings before RMS. In the transpose each parameter
//! has one writer and accumulates its examples in a deterministic order.
#[kernite_macros::kernels]
pub mod kernels {
    fn word(v: u32, ptr: u64, i: u32) {
        let address: u64;
        mad.wide.u32(address, i, 4, ptr);
        ld.global.u32(v, [address]);
    }
    pub unsafe extern "ptx-kernel" fn read(
        hidden: *mut f32,
        values: *const f32,
        keys: *const u32,
        width: u32,
        tables: u32,
    ) {
        let (row, col, at, ix, end, key): (u32, u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(col, tid.x);
        mad.lo.u32(at, row, width, col);
        let (dest, ptr): (u64, u64);
        let (sum, val): (f32, f32);
        mad.wide.u32(dest, at, 4, hidden);
        ld.global.f32(sum, [dest]);
        mul.lo.u32(ix, row, tables);
        add.u32(end, ix, tables);
        let more: pred;
        label(next);
        word(key, keys, ix);
        mad.lo.u32(at, key, width, col);
        mad.wide.u32(ptr, at, 4, values);
        ld.global.f32(val, [ptr]);
        add.rn.f32(sum, sum, val);
        add.u32(ix, ix, 1);
        setp.lt.u32(more, ix, end);
        if more {
            bra(next);
        }
        st.global.f32([dest], sum);
    }
    pub unsafe extern "ptx-kernel" fn backward(
        gradients: *mut f32,
        dhidden: *const f32,
        offsets: *const u32,
        owners: *const u32,
        width: u32,
    ) {
        let (key, col, ix, end, row, at): (u32, u32, u32, u32, u32, u32);
        mov.u32(key, ctaid.x);
        mov.u32(col, tid.x);
        let (sum, val): (f32, f32);
        let ptr: u64;
        mov.f32(sum, 0.0);

        word(ix, offsets, key);
        add.u32(at, key, 1);
        word(end, offsets, at);
        let done: pred;
        label(next);
        setp.ge.u32(done, ix, end);
        if done {
            bra(write);
        }
        word(row, owners, ix);
        mad.lo.u32(at, row, width, col);
        mad.wide.u32(ptr, at, 4, dhidden);
        ld.global.f32(val, [ptr]);
        add.rn.f32(sum, sum, val);
        add.u32(ix, ix, 1);
        bra(next);
        label(write);
        mad.lo.u32(at, key, width, col);
        mad.wide.u32(ptr, at, 4, gradients);
        st.global.f32([ptr], sum);
    }
}
