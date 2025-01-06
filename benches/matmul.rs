use criterion::{Criterion, black_box, criterion_group, criterion_main};
use faer::{Parallelism, mat};
use matmul::{mul, mul_dp};

fn matmul_benchmarks_f64(crit: &mut Criterion) {
    let m = 1000; //number of rows of A
    let n = 1000; // number of columns of B
    let k = 1000; // number of columns of A and number of rows of B , they must be equal!!!!!

    let a: Vec<f64> = (1..=(m * k)).map(|i| i as f64).collect();

    let b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();

    let mut benchmarks = crit.benchmark_group("Matrix multiplication");

    benchmarks.sample_size(20);

    // let mut c: Vec<f64> = (1..=(m * n)).map(|_| 0.0).collect();

    // benchmarks.bench_function("par_matmul-f64", |bencher| {
    //     bencher.iter(|| {
    //         mul_dp::par_matmul(
    //             black_box(a.as_slice()),
    //             black_box(b.as_slice()),
    //             black_box(c.as_mut_slice()),
    //             black_box(m),
    //             black_box(n),
    //             black_box(k),
    //         )
    //     })
    // });

    let mut c: Vec<f64> = (1..=(m * n)).map(|_| 0.0).collect();

    benchmarks.bench_function("matmul-f64", |bencher| {
        bencher.iter(|| {
            mul_dp::matmul(
                black_box(a.as_slice()),
                black_box(b.as_slice()),
                black_box(c.as_mut_slice()),
                black_box(m),
                black_box(n),
                black_box(k),
            )
        })
    });

    let mut c: Vec<f64> = (1..=(m * n)).map(|_| 0.0).collect();

    let a = mat::from_column_major_slice(&a, m, k);
    let b = mat::from_column_major_slice(&b, k, n);

    benchmarks.bench_function("faer-f64", |bencher| {
        bencher.iter(|| {
            let c = mat::from_column_major_slice_mut(c.as_mut_slice(), m, n);
            faer::linalg::matmul::matmul(
                black_box(c),
                black_box(a),
                black_box(b),
                black_box(Some(1.0)),
                black_box(1.0),
                black_box(Parallelism::None),
            );
        })
    });

    benchmarks.finish();
}

fn matmul_benchmarks_f32(crit: &mut Criterion) {
    let m = 100; //number of rows of A
    let n = 100; // number of columns of B
    let k = 100; // number of columns of A and number of rows of B , they must be equal!!!!!

    let a: Vec<f32> = (1..=(m * k)).map(|i| i as f32).collect();

    let b: Vec<f32> = (1..=(k * n)).map(|a| a as f32).collect();

    let mut benchmarks = crit.benchmark_group("Matrix multiplication");

    benchmarks.sample_size(20);

    // let mut c: Vec<f32> = (1..=(m * n)).map(|_| 0.0).collect();

    // benchmarks.bench_function("par_matmul-f32", |bencher| {
    //     bencher.iter(|| {
    //         mul::par_matmul(
    //             black_box(a.as_slice()),
    //             black_box(b.as_slice()),
    //             black_box(c.as_mut_slice()),
    //             black_box(m),
    //             black_box(n),
    //             black_box(k),
    //         )
    //     })
    // });

    let mut c: Vec<f32> = (1..=(m * n)).map(|_| 0.0).collect();

    benchmarks.bench_function("matmul-f32", |bencher| {
        bencher.iter(|| {
            mul::matmul(
                black_box(a.as_slice()),
                black_box(b.as_slice()),
                black_box(c.as_mut_slice()),
                black_box(m),
                black_box(n),
                black_box(k),
            )
        })
    });

    let mut c: Vec<f32> = (1..=(m * n)).map(|_| 0.0).collect();

    let a = mat::from_column_major_slice(&a, m, k);
    let b = mat::from_column_major_slice(&b, k, m);

    benchmarks.bench_function("faer-f32", |bencher| {
        bencher.iter(|| {
            let c = mat::from_column_major_slice_mut(c.as_mut_slice(), m, n);
            faer::linalg::matmul::matmul(
                black_box(c),
                black_box(a),
                black_box(b),
                black_box(Some(1.0)),
                black_box(1.0),
                black_box(Parallelism::None),
            );
        })
    });

    benchmarks.finish();
}

criterion_group!(benches, matmul_benchmarks_f32, matmul_benchmarks_f64);
criterion_main!(benches);
