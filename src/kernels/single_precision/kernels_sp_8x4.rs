#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m256, _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_maskstore_ps,
    _mm256_set_epi32, _mm256_set_ps, _mm256_storeu_ps,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_maskstore_ps,
    _mm256_set_epi32, _mm256_set_ps, _mm256_storeu_ps,
};

use crate::{MR, NR};

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "sse",
    target_feature = "sse2"
))]
#[target_feature(
    enable = "avx",
    enable = "avx2",
    enable = "fma",
    enable = "sse",
    enable = "sse2"
)]
pub unsafe fn kernel_sp_8x1(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    mr: usize,
    _nr: usize,
    kc: usize,
    _m: usize,
) {
    unsafe {
        let mut c_01234567_0 = match mr {
            1 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[0]),
            2 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[1], c[0]),
            3 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, c[2], c[1], c[0]),
            4 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, c[3], c[2], c[1], c[0]),
            5 => _mm256_set_ps(0.0, 0.0, 0.0, c[4], c[3], c[2], c[1], c[0]),
            6 => _mm256_set_ps(0.0, 0.0, c[5], c[4], c[3], c[2], c[1], c[0]),
            7 => _mm256_set_ps(0.0, c[6], c[5], c[4], c[3], c[2], c[1], c[0]),
            8 => _mm256_loadu_ps(&c[0]),
            _ => todo!(),
        };

        // Declare vector register for load/broadcasting B_pj
        let mut b_pj: __m256;

        for p in 0..kc {
            // Declare a vector register to hold the current column of A and load
            // it with the four elements of that column.
            let a_01234567_p = _mm256_loadu_ps(&a[p * MR]);

            // Load/broadcast B_p0
            b_pj = _mm256_broadcast_ss(&b[p * NR]);

            // Update the first column of C with the current column of A time  B_p0
            c_01234567_0 = _mm256_fmadd_ps(a_01234567_p, b_pj, c_01234567_0);
        }

        match mr {
            1 => {
                // Mask to store only the first  element
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
            }
            2 => {
                // Mask to store only the first two elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
            }
            3 => {
                // Mask to store only the first three elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
            }
            4 => {
                // Mask to store only the first four elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
            }
            5 => {
                // Mask to store only the first five elements
                let mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
            }
            6 => {
                // Mask to store only the first six elements
                let mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
            }
            7 => {
                // Mask to store only the first seven elements
                let mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
            }
            8 => _mm256_storeu_ps(&mut c[0], c_01234567_0),
            _ => todo!(),
        }
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "sse",
    target_feature = "sse2"
))]
#[target_feature(
    enable = "avx",
    enable = "avx2",
    enable = "fma",
    enable = "sse",
    enable = "sse2"
)]
pub unsafe fn kernel_sp_8x2(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    mr: usize,
    _nr: usize,
    kc: usize,
    m: usize,
) {
    unsafe {
        let mut c_01234567_0 = match mr {
            1 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[0]),
            2 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[1], c[0]),
            3 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, c[2], c[1], c[0]),
            4 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, c[3], c[2], c[1], c[0]),
            5 => _mm256_set_ps(0.0, 0.0, 0.0, c[4], c[3], c[2], c[1], c[0]),
            6 => _mm256_set_ps(0.0, 0.0, c[5], c[4], c[3], c[2], c[1], c[0]),
            7 => _mm256_set_ps(0.0, c[6], c[5], c[4], c[3], c[2], c[1], c[0]),
            8 => _mm256_loadu_ps(&c[0]),
            _ => todo!(),
        };
        let mut c_01234567_1 = match mr {
            1 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[m]),
            2 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[m + 1], c[m]),
            3 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, c[m + 2], c[m + 1], c[m]),
            4 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, c[m + 3], c[m + 2], c[m + 1], c[m]),
            5 => _mm256_set_ps(0.0, 0.0, 0.0, c[m + 4], c[m + 3], c[m + 2], c[m + 1], c[m]),
            6 => _mm256_set_ps(
                0.0,
                0.0,
                c[m + 5],
                c[m + 4],
                c[m + 3],
                c[m + 2],
                c[m + 1],
                c[m],
            ),
            7 => _mm256_set_ps(
                0.0,
                c[m + 6],
                c[m + 5],
                c[m + 4],
                c[m + 3],
                c[m + 2],
                c[m + 1],
                c[m],
            ),
            8 => _mm256_loadu_ps(&c[m]),
            _ => todo!(),
        };

        // Declare vector register for load/broadcasting B_pj
        let mut b_pj: __m256;

        for p in 0..kc {
            // Declare a vector register to hold the current column of A and load
            // it with the four elements of that column.
            let a_01234567_p = _mm256_loadu_ps(&a[p * MR]);

            // Load/broadcast B_p0
            b_pj = _mm256_broadcast_ss(&b[p * NR]);

            // Update the first column of C with the current column of A time  B_p0
            c_01234567_0 = _mm256_fmadd_ps(a_01234567_p, b_pj, c_01234567_0);

            // Load/broadcast B_p1
            b_pj = _mm256_broadcast_ss(&b[p * NR + 1]);

            // Update the second column of C with the current column of A time  B_p1
            c_01234567_1 = _mm256_fmadd_ps(a_01234567_p, b_pj, c_01234567_1);
        }

        match mr {
            1 => {
                // Mask to store only the first  element
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
            }
            2 => {
                // Mask to store only the first two elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
            }
            3 => {
                // Mask to store only the first three elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
            }
            4 => {
                // Mask to store only the first four elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
            }
            5 => {
                // Mask to store only the first five elements
                let mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
            }
            6 => {
                // Mask to store only the first six elements
                let mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
            }
            7 => {
                // Mask to store only the first seven elements
                let mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
            }
            8 => {
                _mm256_storeu_ps(&mut c[0], c_01234567_0);
                _mm256_storeu_ps(&mut c[m], c_01234567_1);
            }
            _ => todo!(),
        }
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "sse",
    target_feature = "sse2"
))]
#[target_feature(
    enable = "avx",
    enable = "avx2",
    enable = "fma",
    enable = "sse",
    enable = "sse2"
)]
pub unsafe fn kernel_sp_8x3(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    mr: usize,
    _nr: usize,
    kc: usize,
    m: usize,
) {
    unsafe {
        let mut c_01234567_0 = match mr {
            1 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[0]),
            2 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[1], c[0]),
            3 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, c[2], c[1], c[0]),
            4 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, c[3], c[2], c[1], c[0]),
            5 => _mm256_set_ps(0.0, 0.0, 0.0, c[4], c[3], c[2], c[1], c[0]),
            6 => _mm256_set_ps(0.0, 0.0, c[5], c[4], c[3], c[2], c[1], c[0]),
            7 => _mm256_set_ps(0.0, c[6], c[5], c[4], c[3], c[2], c[1], c[0]),
            8 => _mm256_loadu_ps(&c[0]),
            _ => todo!(),
        };

        let mut c_01234567_1 = match mr {
            1 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[m]),
            2 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[m + 1], c[m]),
            3 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, c[m + 2], c[m + 1], c[m]),
            4 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, c[m + 3], c[m + 2], c[m + 1], c[m]),
            5 => _mm256_set_ps(0.0, 0.0, 0.0, c[m + 4], c[m + 3], c[m + 2], c[m + 1], c[m]),
            6 => _mm256_set_ps(
                0.0,
                0.0,
                c[m + 5],
                c[m + 4],
                c[m + 3],
                c[m + 2],
                c[m + 1],
                c[m],
            ),
            7 => _mm256_set_ps(
                0.0,
                c[m + 6],
                c[m + 5],
                c[m + 4],
                c[m + 3],
                c[m + 2],
                c[m + 1],
                c[m],
            ),
            8 => _mm256_loadu_ps(&c[m]),
            _ => todo!(),
        };

        let mut c_01234567_2 = match mr {
            1 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[2 * m]),
            2 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[2 * m + 1], c[2 * m]),
            3 => _mm256_set_ps(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                c[2 * m + 2],
                c[2 * m + 1],
                c[2 * m],
            ),
            4 => _mm256_set_ps(
                0.0,
                0.0,
                0.0,
                0.0,
                c[2 * m + 3],
                c[2 * m + 2],
                c[2 * m + 1],
                c[2 * m],
            ),
            5 => _mm256_set_ps(
                0.0,
                0.0,
                0.0,
                c[2 * m + 4],
                c[2 * m + 3],
                c[2 * m + 2],
                c[2 * m + 1],
                c[2 * m],
            ),
            6 => _mm256_set_ps(
                0.0,
                0.0,
                c[2 * m + 5],
                c[2 * m + 4],
                c[2 * m + 3],
                c[2 * m + 2],
                c[2 * m + 1],
                c[2 * m],
            ),
            7 => _mm256_set_ps(
                0.0,
                c[2 * m + 6],
                c[2 * m + 5],
                c[2 * m + 4],
                c[2 * m + 3],
                c[2 * m + 2],
                c[2 * m + 1],
                c[2 * m],
            ),
            8 => _mm256_loadu_ps(&c[2 * m]),
            _ => todo!(),
        };

        // Declare vector register for load/broadcasting B_pj
        let mut b_pj: __m256;

        for p in 0..kc {
            // Declare a vector register to hold the current column of A and load
            // it with the four elements of that column.
            let a_01234567_p = _mm256_loadu_ps(&a[p * MR]);

            // Load/broadcast B_p0
            b_pj = _mm256_broadcast_ss(&b[p * NR]);

            // Update the first column of C with the current column of A time  B_p0
            c_01234567_0 = _mm256_fmadd_ps(a_01234567_p, b_pj, c_01234567_0);

            // Load/broadcast B_p1
            b_pj = _mm256_broadcast_ss(&b[p * NR + 1]);

            // Update the second column of C with the current column of A time  B_p1
            c_01234567_1 = _mm256_fmadd_ps(a_01234567_p, b_pj, c_01234567_1);

            // Load/broadcast B_p2
            b_pj = _mm256_broadcast_ss(&b[p * NR + 2]);

            // Update the third column of C with the current column of A time  B_p2
            c_01234567_2 = _mm256_fmadd_ps(a_01234567_p, b_pj, c_01234567_2);
        }

        match mr {
            1 => {
                // Mask to store only the first  element
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
            }
            2 => {
                // Mask to store only the first two elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
            }
            3 => {
                // Mask to store only the first three elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
            }
            4 => {
                // Mask to store only the first four elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
            }
            5 => {
                // Mask to store only the first five elements
                let mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
            }
            6 => {
                // Mask to store only the first six elements
                let mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
            }
            7 => {
                // Mask to store only the first seven elements
                let mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
            }
            8 => {
                _mm256_storeu_ps(&mut c[0], c_01234567_0);
                _mm256_storeu_ps(&mut c[m], c_01234567_1);
                _mm256_storeu_ps(&mut c[2 * m], c_01234567_2);
            }
            _ => todo!(),
        }
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "sse",
    target_feature = "sse2"
))]
#[target_feature(
    enable = "avx",
    enable = "avx2",
    enable = "fma",
    enable = "sse",
    enable = "sse2"
)]
pub unsafe fn kernel_sp_8x4(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    mr: usize,
    _nr: usize,
    kc: usize,
    m: usize,
) {
    unsafe {
        let mut c_01234567_0 = match mr {
            1 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[0]),
            2 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[1], c[0]),
            3 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, c[2], c[1], c[0]),
            4 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, c[3], c[2], c[1], c[0]),
            5 => _mm256_set_ps(0.0, 0.0, 0.0, c[4], c[3], c[2], c[1], c[0]),
            6 => _mm256_set_ps(0.0, 0.0, c[5], c[4], c[3], c[2], c[1], c[0]),
            7 => _mm256_set_ps(0.0, c[6], c[5], c[4], c[3], c[2], c[1], c[0]),
            8 => _mm256_loadu_ps(&c[0]),
            _ => todo!(),
        };

        let mut c_01234567_1 = match mr {
            1 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[m]),
            2 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[m + 1], c[m]),
            3 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, c[m + 2], c[m + 1], c[m]),
            4 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, c[m + 3], c[m + 2], c[m + 1], c[m]),
            5 => _mm256_set_ps(0.0, 0.0, 0.0, c[m + 4], c[m + 3], c[m + 2], c[m + 1], c[m]),
            6 => _mm256_set_ps(
                0.0,
                0.0,
                c[m + 5],
                c[m + 4],
                c[m + 3],
                c[m + 2],
                c[m + 1],
                c[m],
            ),
            7 => _mm256_set_ps(
                0.0,
                c[m + 6],
                c[m + 5],
                c[m + 4],
                c[m + 3],
                c[m + 2],
                c[m + 1],
                c[m],
            ),
            8 => _mm256_loadu_ps(&c[m]),
            _ => todo!(),
        };

        let mut c_01234567_2 = match mr {
            1 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[2 * m]),
            2 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[2 * m + 1], c[2 * m]),
            3 => _mm256_set_ps(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                c[2 * m + 2],
                c[2 * m + 1],
                c[2 * m],
            ),
            4 => _mm256_set_ps(
                0.0,
                0.0,
                0.0,
                0.0,
                c[2 * m + 3],
                c[2 * m + 2],
                c[2 * m + 1],
                c[2 * m],
            ),
            5 => _mm256_set_ps(
                0.0,
                0.0,
                0.0,
                c[2 * m + 4],
                c[2 * m + 3],
                c[2 * m + 2],
                c[2 * m + 1],
                c[2 * m],
            ),
            6 => _mm256_set_ps(
                0.0,
                0.0,
                c[2 * m + 5],
                c[2 * m + 4],
                c[2 * m + 3],
                c[2 * m + 2],
                c[2 * m + 1],
                c[2 * m],
            ),
            7 => _mm256_set_ps(
                0.0,
                c[2 * m + 6],
                c[2 * m + 5],
                c[2 * m + 4],
                c[2 * m + 3],
                c[2 * m + 2],
                c[2 * m + 1],
                c[2 * m],
            ),
            8 => _mm256_loadu_ps(&c[2 * m]),
            _ => todo!(),
        };

        let mut c_01234567_3 = match mr {
            1 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[3 * m]),
            2 => _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[3 * m + 1], c[3 * m]),
            3 => _mm256_set_ps(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                c[3 * m + 2],
                c[3 * m + 1],
                c[3 * m],
            ),
            4 => _mm256_set_ps(
                0.0,
                0.0,
                0.0,
                0.0,
                c[3 * m + 3],
                c[3 * m + 2],
                c[3 * m + 1],
                c[3 * m],
            ),
            5 => _mm256_set_ps(
                0.0,
                0.0,
                0.0,
                c[3 * m + 4],
                c[3 * m + 3],
                c[3 * m + 2],
                c[3 * m + 1],
                c[3 * m],
            ),
            6 => _mm256_set_ps(
                0.0,
                0.0,
                c[3 * m + 5],
                c[3 * m + 4],
                c[3 * m + 3],
                c[3 * m + 2],
                c[3 * m + 1],
                c[3 * m],
            ),
            7 => _mm256_set_ps(
                0.0,
                c[3 * m + 6],
                c[3 * m + 5],
                c[3 * m + 4],
                c[3 * m + 3],
                c[3 * m + 2],
                c[3 * m + 1],
                c[3 * m],
            ),
            8 => _mm256_loadu_ps(&c[3 * m]),
            _ => todo!(),
        };

        // Declare vector register for load/broadcasting B_pj
        let mut b_pj: __m256;

        for p in 0..kc {
            // Declare a vector register to hold the current column of A and load
            // it with the four elements of that column.
            let a_0123_p = _mm256_loadu_ps(&a[p * MR]);

            // Load/broadcast B_p0
            b_pj = _mm256_broadcast_ss(&b[p * NR]);

            // Update the first column of C with the current column of A time  B_p0
            c_01234567_0 = _mm256_fmadd_ps(a_0123_p, b_pj, c_01234567_0);

            // Load/broadcast B_p1
            b_pj = _mm256_broadcast_ss(&b[p * NR + 1]);

            // Update the second column of C with the current column of A time  B_p1
            c_01234567_1 = _mm256_fmadd_ps(a_0123_p, b_pj, c_01234567_1);

            // Load/broadcast B_p2
            b_pj = _mm256_broadcast_ss(&b[p * NR + 2]);

            // Update the third column of C with the current column of A time  B_p2
            c_01234567_2 = _mm256_fmadd_ps(a_0123_p, b_pj, c_01234567_2);

            // Load/broadcast B_p3
            b_pj = _mm256_broadcast_ss(&b[p * NR + 3]);

            // Update the fourth column of C with the current column of A time  B_p3
            c_01234567_3 = _mm256_fmadd_ps(a_0123_p, b_pj, c_01234567_3);
        }

        match mr {
            1 => {
                // Mask to store only the first  element
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
                _mm256_maskstore_ps(&mut c[3 * m], mask, c_01234567_3);
            }
            2 => {
                // Mask to store only the first two elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
                _mm256_maskstore_ps(&mut c[3 * m], mask, c_01234567_3);
            }
            3 => {
                // Mask to store only the first three elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
                _mm256_maskstore_ps(&mut c[3 * m], mask, c_01234567_3);
            }
            4 => {
                // Mask to store only the first four elements
                let mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
                _mm256_maskstore_ps(&mut c[3 * m], mask, c_01234567_3);
            }
            5 => {
                // Mask to store only the first five elements
                let mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
                _mm256_maskstore_ps(&mut c[3 * m], mask, c_01234567_3);
            }
            6 => {
                // Mask to store only the first six elements
                let mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
                _mm256_maskstore_ps(&mut c[3 * m], mask, c_01234567_3);
            }
            7 => {
                // Mask to store only the first seven elements
                let mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
                _mm256_maskstore_ps(&mut c[0], mask, c_01234567_0);
                _mm256_maskstore_ps(&mut c[m], mask, c_01234567_1);
                _mm256_maskstore_ps(&mut c[2 * m], mask, c_01234567_2);
                _mm256_maskstore_ps(&mut c[3 * m], mask, c_01234567_3);
            }
            8 => {
                _mm256_storeu_ps(&mut c[0], c_01234567_0);
                _mm256_storeu_ps(&mut c[m], c_01234567_1);
                _mm256_storeu_ps(&mut c[2 * m], c_01234567_2);
                _mm256_storeu_ps(&mut c[3 * m], c_01234567_2);
            }
            _ => todo!(),
        }
    }
}
