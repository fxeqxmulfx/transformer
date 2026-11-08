//! Auxiliary RMS, including partial warps at widths 341/682. Source:
//! rotate90_complex ops.rs, pinned SHA 14cff3b5, separated radial/tangent
//! derivative; changed forward reduction uses FP64 as frozen in PROTOCOL.md.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        ld.global.f32(x, [p]);
    }
    fn store(base: u64, i: u32, x: f32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        st.global.f32([p], x);
    }
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    fn sum64(x: f64) {
        let (lo, hi, ol, oh): (u32, u32, u32, u32);
        let v: f64;
        for j in 0..5 {
            mov.b64((lo, hi), x);
            shfl.sync.bfly.b32(ol, lo, 1 << j, 31, 0xffffffff);
            shfl.sync.bfly.b32(oh, hi, 1 << j, 31, 0xffffffff);
            mov.b64(v, (ol, oh));
            add.rn.f64(x, x, v);
        }
    }
    pub unsafe extern "ptx-kernel" fn copy(out: *mut f32, x: *const f32, count: u32) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let v: f32;
        load(v, x, i);
        store(out, i, v);
    }
    #[instances(128, 341, 682)]
    pub unsafe extern "ptx-kernel" fn rms<const WIDTH: usize>(
        out: *mut f32,
        x: *const f32,
        rows: u32,
    ) {
        let (i, row, lane, k, at): (u32, u32, u32, u32, u32);
        index(i);
        shr.u32(row, i, 5);
        and.b32(lane, i, 31);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let v: [f64; (WIDTH + 31) / 32];
        let (energy, temp, inv): (f64, f64, f64);
        let z: f32;
        mov.f64(energy, 0.0f64);
        for j in 0..(WIDTH + 31) / 32 {
            add.u32(k, lane, 32 * j);
            mad.lo.u32(at, row, WIDTH, k);
            mov.f32(z, 0.0);
            setp.lt.u32(past, k, WIDTH);
            if past {
                load(z, x, at);
            }
            cvt.f64.f32(v[j], z);
            mul.rn.f64(temp, v[j], v[j]);
            add.rn.f64(energy, energy, temp);
        }
        sum64(energy);
        mul.rn.f64(energy, energy, 1.0f64 / (WIDTH as f64));
        add.rn.f64(inv, energy, 0.00000011920928955078125f64);
        sqrt.rn.f64(inv, inv);
        div.rn.f64(inv, 1.0f64, inv);
        for j in 0..(WIDTH + 31) / 32 {
            add.u32(k, lane, 32 * j);
            mad.lo.u32(at, row, WIDTH, k);
            mul.rn.f64(temp, v[j], inv);
            cvt.rn.f32.f64(z, temp);
            setp.lt.u32(past, k, WIDTH);
            if past {
                store(out, at, z);
            }
        }
    }
    #[instances(128, 341, 682)]
    pub unsafe extern "ptx-kernel" fn derivative<const WIDTH: usize>(
        out: *mut f32,
        x: *const f32,
        dy: *const f32,
        rows: u32,
    ) {
        let (i, row, lane, k, at): (u32, u32, u32, u32, u32);
        index(i);
        shr.u32(row, i, 5);
        and.b32(lane, i, 31);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (v, g): ([f64; (WIDTH + 31) / 32], [f64; (WIDTH + 31) / 32]);
        let (energy, dot, temp, inv, den, radial, parallel, tangent): (
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
        );
        let z: f32;
        mov.f64(energy, 0.0f64);
        mov.f64(dot, 0.0f64);
        for j in 0..(WIDTH + 31) / 32 {
            add.u32(k, lane, 32 * j);
            mad.lo.u32(at, row, WIDTH, k);
            setp.lt.u32(past, k, WIDTH);
            mov.f32(z, 0.0);
            if past {
                load(z, x, at);
            }
            cvt.f64.f32(v[j], z);
            mov.f32(z, 0.0);
            if past {
                load(z, dy, at);
            }
            cvt.f64.f32(g[j], z);
            mul.rn.f64(temp, v[j], v[j]);
            add.rn.f64(energy, energy, temp);
            mul.rn.f64(temp, v[j], g[j]);
            add.rn.f64(dot, dot, temp);
        }
        sum64(energy);
        sum64(dot);
        mul.rn.f64(energy, energy, 1.0f64 / (WIDTH as f64));
        mul.rn.f64(dot, dot, 1.0f64 / (WIDTH as f64));
        let positive: pred;
        setp.gt.f64(positive, energy, 0.0f64);
        selp.f64(den, energy, 1.0f64, positive);
        add.rn.f64(inv, energy, 0.00000011920928955078125f64);
        sqrt.rn.f64(inv, inv);
        div.rn.f64(inv, 1.0f64, inv);
        mul.rn.f64(radial, inv, inv);
        mul.rn.f64(radial, radial, inv);
        mul.rn.f64(radial, radial, 0.00000011920928955078125f64);
        for j in 0..(WIDTH + 31) / 32 {
            mul.rn.f64(parallel, v[j], dot);
            div.rn.f64(parallel, parallel, den);
            sub.rn.f64(tangent, g[j], parallel);
            mul.rn.f64(tangent, tangent, inv);
            mul.rn.f64(parallel, parallel, radial);
            add.rn.f64(tangent, tangent, parallel);
            cvt.rn.f32.f64(z, tangent);
            add.u32(k, lane, 32 * j);
            mad.lo.u32(at, row, WIDTH, k);
            setp.lt.u32(past, k, WIDTH);
            if past {
                store(out, at, z);
            }
        }
    }
}
