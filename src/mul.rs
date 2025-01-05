use std::{cmp::min, sync::Mutex};

use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::{
    KC, MC, MR, NC, NR,
    kernels::single_precision::{
        kernels_sp_4x4::{kernel_sp_4x1, kernel_sp_4x2, kernel_sp_4x3, kernel_sp_4x4},
        kernels_sp_8x4::{kernel_sp_8x1, kernel_sp_8x2, kernel_sp_8x3, kernel_sp_8x4},
    },
};

pub fn display(m: usize, n: usize, ld: usize, a: &[f32]) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", a[at(i, j, ld)]);
        }
        println!()
    }
}

pub fn par_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
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

                                    let c_micropanel =
                                        &mut c_chunk[(jr * NR * m + (ic + ir * MR))..];

                                    match nr {
                                        1 => unsafe {
                                            kernel_sp_4x1(
                                                a_panel,
                                                b_panel,
                                                c_micropanel,
                                                mr,
                                                nr,
                                                kc,
                                                m,
                                            )
                                        },
                                        2 => unsafe {
                                            kernel_sp_4x2(
                                                a_panel,
                                                b_panel,
                                                c_micropanel,
                                                mr,
                                                nr,
                                                kc,
                                                m,
                                            )
                                        },
                                        3 => unsafe {
                                            kernel_sp_4x3(
                                                a_panel,
                                                b_panel,
                                                c_micropanel,
                                                mr,
                                                nr,
                                                kc,
                                                m,
                                            );
                                        },
                                        4 => unsafe {
                                            kernel_sp_4x4(
                                                a_panel,
                                                b_panel,
                                                c_micropanel,
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

fn pack_panel_b(b: &[f32], nr: usize, kc: usize, k: usize) -> Vec<f32> {
    let mut panel_b_packed = Vec::new();

    // let mut panel_b_packed = aligned_vec_f32(kc * NR, 32);

    for p in 0..kc {
        for j in 0..nr {
            panel_b_packed.push(b[j * k + p]);
        }
        panel_b_packed.extend(vec![0.0; NR - nr]);
    }

    panel_b_packed
}

fn pack_block_b(b: &[f32], nc: usize, kc: usize, k: usize) -> Vec<f32> {
    let mut block_b_packed = Vec::new();

    for j in (0..nc).step_by(NR) {
        let nr = min(NR, nc - j);
        let panel_b_packed = pack_panel_b(&b[j * k..], nr, kc, k);
        block_b_packed.extend(panel_b_packed);
    }

    block_b_packed
}

fn pack_panel_a(a: &[f32], mr: usize, kc: usize, m: usize) -> Vec<f32> {
    let mut panel_a_packed = Vec::new();

    // let mut panel_a_packed = aligned_vec_f32(kc * MR, 32);
    for p in 0..kc {
        for i in 0..mr {
            panel_a_packed.push(a[p * m + i]);
        }
        panel_a_packed.extend(vec![0.0; MR - mr]);
    }

    panel_a_packed
}

fn pack_block_a(a: &[f32], mc: usize, kc: usize, m: usize) -> Vec<f32> {
    let mut block_a_packed = Vec::new();

    for i in (0..mc).step_by(MR) {
        let mr = min(MR, mc - i);
        let panel_a_packed = pack_panel_a(&a[i..], mr, kc, m);
        block_a_packed.extend(panel_a_packed);
    }

    block_a_packed
}

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.par_chunks_mut(m * NC)
        .enumerate()
        .for_each(|(j, c_chunk)| {
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

                                    let c_micropanel =
                                        &mut c_chunk[(jr * NR * m + (ic + ir * MR))..];

                                    match nr {
                                        1 => unsafe {
                                            kernel_sp_4x1(
                                                a_panel,
                                                b_panel,
                                                c_micropanel,
                                                mr,
                                                nr,
                                                kc,
                                                m,
                                            )
                                        },
                                        2 => unsafe {
                                            kernel_sp_4x2(
                                                a_panel,
                                                b_panel,
                                                c_micropanel,
                                                mr,
                                                nr,
                                                kc,
                                                m,
                                            )
                                        },
                                        3 => unsafe {
                                            kernel_sp_4x3(
                                                a_panel,
                                                b_panel,
                                                c_micropanel,
                                                mr,
                                                nr,
                                                kc,
                                                m,
                                            );
                                        },
                                        4 => unsafe {
                                            kernel_sp_4x4(
                                                a_panel,
                                                b_panel,
                                                c_micropanel,
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

fn at(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
}

pub fn naive_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    ld_a: usize,
    b: &[f32],
    ld_b: usize,
    c: &mut [f32],
    ld_c: usize,
) {
    for j in 0..n {
        for p in 0..k {
            for i in 0..m {
                c[at(i, j, ld_c)] = c[at(i, j, ld_c)] + a[at(i, p, ld_a)] * b[at(p, j, ld_b)];
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_gemm_5_loops() {
        // defining matrices as vectors (column major matrices)
        let m = 5; //number of rows of A
        let n = 5; // number of columns of B
        let k = 10_000; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f32> = (1..=(m * k)).map(|_| 1.0).collect();
        let ld_a = m; // leading dimension of A (number of rows)

        let b: Vec<f32> = (1..=(k * n)).map(|_| 1.0).collect();
        let ld_b = k; // leading dimension of B (number of rows)

        let mut c: Vec<f32> = (1..=(m * n)).map(|_| 0.0).collect();
        let mut c_ref: Vec<f32> = (1..=(m * n)).map(|_| 0.0).collect();
        let ld_c = m; // leading dimension of C (number of rows)

        // computing C:=AB + C
        matmul(a.as_slice(), b.as_slice(), c.as_mut_slice(), m, n, k);

        naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

        c.clone()
            .into_iter()
            .zip(c_ref.clone())
            .enumerate()
            .for_each(|(_, (c, c_ref))| {
                println!(
                    "{:?} {:?} --> {}",
                    c_ref,
                    c,
                    (c - c_ref).abs() <= f32::EPSILON
                )
            });

        assert_eq!(c, c_ref);
    }
}
