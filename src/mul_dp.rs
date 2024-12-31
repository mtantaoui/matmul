use std::{cmp::min, sync::Mutex};

use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::kernels::kernels_dp_4x4::{kernel_4x1, kernel_4x2, kernel_4x3, kernel_4x4};

const NUM_THREADS: usize = 8;

pub const MR: usize = 4;
pub const NR: usize = 4;

const MC: usize = MR * NUM_THREADS;
const NC: usize = NR * NUM_THREADS;
const KC: usize = NUM_THREADS;

pub fn par_matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    c.par_chunks_mut(m * NC)
        .enumerate()
        .for_each(|(j, c_chunk)| {
            let jc = j * NC;
            let nc = min(NC, n - jc);

            let c_chunk_mutex = Mutex::new(c_chunk);

            a.par_chunks(m * KC).enumerate().for_each(|(p, a_chunk)| {
                let pc = p * KC;
                let kc = min(KC, k - pc);

                let block_b_packed = pack_block_b(&b[(jc * k + pc)..], nc, kc, k);

                for ic in (0..m).step_by(MC) {
                    let mc = min(MC, m - ic);

                    let block_a_packed = pack_block_a(&a_chunk[ic..], mc, kc, m);

                    block_b_packed
                        .par_chunks(kc * NR)
                        .enumerate()
                        .for_each(|(jr, b_panel)| {
                            block_a_packed
                                .chunks(MR * kc)
                                .enumerate()
                                .for_each(|(ir, a_panel)| {
                                    let nr = min(NR, nc - jr * NR);
                                    let mr = min(MR, mc - ir * MR);

                                    let mut c_chunk = c_chunk_mutex.lock().unwrap();

                                    let mut c_micropanel =
                                        &mut c_chunk[(jr * NR * m + (ic + ir * MR))..];

                                    match nr {
                                        1 => unsafe {
                                            kernel_4x1(
                                                a_panel,
                                                b_panel,
                                                &mut c_micropanel,
                                                mr,
                                                nr,
                                                kc,
                                                m,
                                            )
                                        },
                                        2 => unsafe {
                                            kernel_4x2(
                                                a_panel,
                                                b_panel,
                                                &mut c_micropanel,
                                                mr,
                                                nr,
                                                kc,
                                                m,
                                            )
                                        },
                                        3 => unsafe {
                                            kernel_4x3(
                                                a_panel,
                                                b_panel,
                                                &mut c_micropanel,
                                                mr,
                                                nr,
                                                kc,
                                                m,
                                            );
                                        },
                                        4 => unsafe {
                                            kernel_4x4(
                                                a_panel,
                                                b_panel,
                                                &mut c_micropanel,
                                                mr,
                                                nr,
                                                kc,
                                                m,
                                            )
                                        },
                                        _ => todo!(),
                                    }
                                });
                        });
                }
            });
        });
}

fn pack_panel_b(b: &[f64], nr: usize, kc: usize, k: usize) -> Vec<f64> {
    let mut panel_b_packed = Vec::new();

    // let mut panel_b_packed = aligned_vec_f64(kc * NR, 32);

    for p in 0..kc {
        for j in 0..nr {
            panel_b_packed.push(b[j * k + p]);
        }
        panel_b_packed.extend(vec![0.0; NR - nr]);
    }

    return panel_b_packed;
}

fn pack_block_b(b: &[f64], nc: usize, kc: usize, k: usize) -> Vec<f64> {
    let mut block_b_packed = Vec::new();

    for j in (0..nc).step_by(NR) {
        let nr = min(NR, nc - j);
        let panel_b_packed = pack_panel_b(&b[j * k..], nr, kc, k);
        block_b_packed.extend(panel_b_packed);
    }

    return block_b_packed;
}

fn pack_panel_a(a: &[f64], mr: usize, kc: usize, m: usize) -> Vec<f64> {
    let mut panel_a_packed = Vec::new();

    // let mut panel_a_packed = aligned_vec_f64(kc * MR, 32);
    for p in 0..kc {
        for i in 0..mr {
            panel_a_packed.push(a[p * m + i]);
        }
        panel_a_packed.extend(vec![0.0; MR - mr]);
    }

    return panel_a_packed;
}

fn pack_block_a(a: &[f64], mc: usize, kc: usize, m: usize) -> Vec<f64> {
    let mut block_a_packed = Vec::new();

    for i in (0..mc).step_by(MR) {
        let mr = min(MR, mc - i);
        let panel_a_packed = pack_panel_a(&a[i..], mr, kc, m);
        block_a_packed.extend(panel_a_packed);
    }

    return block_a_packed;
}

pub fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    c.chunks_mut(m * NC).enumerate().for_each(|(j, c_chunk)| {
        let jc = j * NC;
        let nc = min(NC, n - jc);

        a.chunks(m * KC).enumerate().for_each(|(p, a_chunk)| {
            let pc = p * KC;
            let kc = min(KC, k - pc);

            let block_b_packed = pack_block_b(&b[(jc * k + pc)..], nc, kc, k);

            for ic in (0..m).step_by(MC) {
                let mc = min(MC, m - ic);

                let block_a_packed = pack_block_a(&a_chunk[ic..], mc, kc, m);

                block_b_packed
                    .chunks(kc * NR)
                    .enumerate()
                    .for_each(|(jr, b_panel)| {
                        block_a_packed
                            .chunks(MR * kc)
                            .enumerate()
                            .for_each(|(ir, a_panel)| {
                                let nr = min(NR, nc - jr * NR);
                                let mr = min(MR, mc - ir * MR);

                                let mut c_micropanel =
                                    &mut c_chunk[(jr * NR * m + (ic + ir * MR))..];

                                match nr {
                                    1 => unsafe {
                                        kernel_4x1(
                                            a_panel,
                                            b_panel,
                                            &mut c_micropanel,
                                            mr,
                                            nr,
                                            kc,
                                            m,
                                        )
                                    },
                                    2 => unsafe {
                                        kernel_4x2(
                                            a_panel,
                                            b_panel,
                                            &mut c_micropanel,
                                            mr,
                                            nr,
                                            kc,
                                            m,
                                        )
                                    },
                                    3 => unsafe {
                                        kernel_4x3(
                                            a_panel,
                                            b_panel,
                                            &mut c_micropanel,
                                            mr,
                                            nr,
                                            kc,
                                            m,
                                        );
                                    },
                                    4 => unsafe {
                                        kernel_4x4(
                                            a_panel,
                                            b_panel,
                                            &mut c_micropanel,
                                            mr,
                                            nr,
                                            kc,
                                            m,
                                        )
                                    },
                                    _ => todo!(),
                                }
                            });
                    });
            }
        });
    });
}

pub fn naive_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &[f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    fn at(i: usize, j: usize, ld: usize) -> usize {
        (j * ld) + i
    }

    for j in 0..n {
        for p in 0..k {
            for i in 0..m {
                c[at(i, j, ld_c)] += a[at(i, p, ld_a)] * b[at(p, j, ld_b)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        // defining matrices as vectors (column major matrices)
        let m = 100; //number of rows of A
        let n = 100; // number of columns of B
        let k = 100; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|i| i as f64).collect();
        let ld_a = m; // leading dimension of A (number of rows)

        let b: Vec<f64> = (1..=(k * n)).map(|i| i as f64).collect();
        let ld_b = k; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|_| 0.0).collect();
        let mut c_ref: Vec<f64> = (1..=(m * n)).map(|_| 0.0).collect();
        let ld_c = m; // leading dimension of C (number of rows)

        // computing C:=AB + C
        println!("MR: {}", MR);
        println!("NR: {}", NR);

        matmul(a.as_slice(), b.as_slice(), c.as_mut_slice(), m, n, k);

        naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

        assert_eq!(c, c_ref);
    }

    #[test]
    fn test_par_matmul() {
        // defining matrices as vectors (column major matrices)
        let m = 100; //number of rows of A
        let n = 100; // number of columns of B
        let k = 100; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|i| i as f64).collect();
        let ld_a = m; // leading dimension of A (number of rows)

        let b: Vec<f64> = (1..=(k * n)).map(|i| i as f64).collect();
        let ld_b = k; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|_| 0.0).collect();
        let mut c_ref: Vec<f64> = (1..=(m * n)).map(|_| 0.0).collect();
        let ld_c = m; // leading dimension of C (number of rows)

        // computing C:=AB + C
        par_matmul(a.as_slice(), b.as_slice(), c.as_mut_slice(), m, n, k);

        naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

        assert_eq!(c, c_ref);
    }
}
