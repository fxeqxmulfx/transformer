//! Compare the warp's confidence and tie decisions with the scalar classifier.
use super::*;
use crate::{slot_memory::trusted_slot, slot_orbits::ABSENT};

pub(super) fn run() -> Result<()> {
    let cuda = kernite_cuda::Cuda::new()?;
    let kernels = crate::slot_addresses::kernels::Kernels::load(&cuda)?;
    let mut rng = random::Random::new(8909);
    let cut = std::f32::consts::FRAC_1_SQRT_2;
    for slots in [1, 7, 32, 63, 64, 97, 128, 256] {
        let cells: usize = 79;
        let mut inverse: Vec<_> = (0..slots).map(|_| rng.gauss(1., 0.15) as f32).collect();
        let mut scales: Vec<_> = (0..cells).map(|_| rng.gauss(1., 0.3) as f32).collect();
        let mut products: Vec<_> = (0..cells * slots)
            .map(|_| rng.gauss(0., 1.) as f32)
            .collect();
        let middle = 32.min(slots - 1);
        let last = slots - 1;
        for i in [0, middle, last] {
            inverse[i] = 1.;
        }
        scales[..9].fill(1.);
        scales[0] = 0.;
        products[..9 * slots].fill(0.);
        products[0] = 20.;
        products[slots] = cut;
        products[2 * slots] = cut.next_up();
        products[3 * slots] = 1.;
        products[3 * slots + middle] = 1.; // Tie in the same lane when middle is 32.
        products[4 * slots] = 1.;
        products[4 * slots + last] = 1.; // A different lane can tie the peak.
        products[5 * slots] = 1.;
        products[5 * slots + middle] = 1.;
        products[5 * slots + last] = 2.;
        products[6 * slots] = cut.next_down();
        products[7 * slots + last] = f32::NAN;
        products[8 * slots] = f32::INFINITY;
        let expected: Vec<_> = products
            .chunks_exact(slots)
            .zip(&scales)
            .map(|(row, &scale)| trusted_slot(row, &inverse, scale))
            .collect();
        if expected[..3] != [ABSENT, ABSENT, 0]
            || expected[5] != last as u32
            || slots > 1 && expected[3..5] != [ABSENT; 2]
        {
            return Err("trusted-bank fixture missed a confidence boundary or tie".into());
        }
        let products = cuda.upload(&products)?;
        let inverse = cuda.upload(&inverse)?;
        let scales = cuda.upload(&scales)?;
        let bank = cuda.alloc::<u32>(cells)?;
        unsafe {
            kernels.trusted_bank(
                kernite_cuda::Grid::new(cells.div_ceil(4) as u32, 128),
                bank.ptr(),
                products.ptr(),
                inverse.ptr(),
                scales.ptr(),
                slots as u32,
                cells as u32,
                cut,
            )?;
        }
        if bank.read()? != expected {
            return Err(
                format!("parallel confidence bank differs at dictionary size {slots}").into(),
            );
        }
        println!("parallel trusted bank slots={slots} scalar_matches={cells}");
    }
    Ok(())
}
