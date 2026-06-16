use ndarray::linalg::general_mat_mul;
use ndarray::{ArrayView2, ArrayViewMut2, Axis};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::hint::black_box;
use std::time::{Duration, Instant};

const MUTATIONS: usize = 2458;
const CLUSTERS: usize = 40;
const CONTRACTION: usize = 300;
const THREADS: usize = 16;
const ITERS: usize = 80;

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn max_abs_diff(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn checksum(values: &[f64]) -> f64 {
    values.iter().step_by(97).sum()
}

fn time_it<F>(mut f: F) -> (Duration, f64)
where
    F: FnMut() -> f64,
{
    let mut timings = Vec::with_capacity(ITERS);
    let mut last = 0.0;
    for i in 0..(ITERS + 10) {
        let started = Instant::now();
        last = f();
        let elapsed = started.elapsed();
        if i >= 10 {
            timings.push(elapsed);
        }
    }
    (median(timings), last)
}

fn update_z_hand(data: &[f64], theta: &[f64], out: &mut [f64]) {
    out.par_chunks_mut(CLUSTERS)
        .enumerate()
        .for_each(|(mutation_index, row)| {
            row.fill(0.0);
            let data_row = &data[mutation_index * CONTRACTION..(mutation_index + 1) * CONTRACTION];
            for contraction_index in 0..CONTRACTION {
                let data_value = data_row[contraction_index];
                let theta_row =
                    &theta[contraction_index * CLUSTERS..(contraction_index + 1) * CLUSTERS];
                for cluster_index in 0..CLUSTERS {
                    row[cluster_index] += data_value * theta_row[cluster_index];
                }
            }
        });
}

fn update_z_gemm_row_split(data: &[f64], theta: &[f64], out: &mut [f64]) {
    let chunk_rows = MUTATIONS.div_ceil(THREADS);
    let theta = ArrayView2::from_shape((CONTRACTION, CLUSTERS), theta).unwrap();
    out.par_chunks_mut(chunk_rows * CLUSTERS)
        .enumerate()
        .for_each(|(chunk_index, out_chunk)| {
            let row_start = chunk_index * chunk_rows;
            let rows = out_chunk.len() / CLUSTERS;
            let data_chunk = ArrayView2::from_shape(
                (rows, CONTRACTION),
                &data[row_start * CONTRACTION..(row_start + rows) * CONTRACTION],
            )
            .unwrap();
            let mut out_view = ArrayViewMut2::from_shape((rows, CLUSTERS), out_chunk).unwrap();
            general_mat_mul(1.0, &data_chunk, &theta, 0.0, &mut out_view);
        });
}

fn update_theta_hand(z_t: &[f64], data: &[f64], out: &mut [f64]) {
    out.par_chunks_mut(CONTRACTION)
        .enumerate()
        .for_each(|(cluster_index, row)| {
            row.fill(0.0);
            let z_row = &z_t[cluster_index * MUTATIONS..(cluster_index + 1) * MUTATIONS];
            for mutation_index in 0..MUTATIONS {
                let weight = z_row[mutation_index];
                let data_row =
                    &data[mutation_index * CONTRACTION..(mutation_index + 1) * CONTRACTION];
                for contraction_index in 0..CONTRACTION {
                    row[contraction_index] += weight * data_row[contraction_index];
                }
            }
        });
}

// This is the tempting but slow update_theta split: each GEMM has only a few
// cluster rows, so the matrixmultiply micro-kernel cannot amortize its setup.
fn update_theta_gemm_cluster_split(z_t: &[f64], data: &[f64], out: &mut [f64]) {
    let chunk_clusters = CLUSTERS.div_ceil(THREADS);
    let data = ArrayView2::from_shape((MUTATIONS, CONTRACTION), data).unwrap();
    out.par_chunks_mut(chunk_clusters * CONTRACTION)
        .enumerate()
        .for_each(|(chunk_index, out_chunk)| {
            let cluster_start = chunk_index * chunk_clusters;
            let rows = out_chunk.len() / CONTRACTION;
            let z_chunk = ArrayView2::from_shape(
                (rows, MUTATIONS),
                &z_t[cluster_start * MUTATIONS..(cluster_start + rows) * MUTATIONS],
            )
            .unwrap();
            let mut out_view = ArrayViewMut2::from_shape((rows, CONTRACTION), out_chunk).unwrap();
            general_mat_mul(1.0, &z_chunk, &data, 0.0, &mut out_view);
        });
}

// This matches the production update_theta path: keep all cluster rows together
// and split the wider contraction axis into strided output column blocks.
fn update_theta_gemm_column_split(z_t: &[f64], data: &[f64], out: &mut [f64]) {
    let z_t = ArrayView2::from_shape((CLUSTERS, MUTATIONS), z_t).unwrap();
    let data = ArrayView2::from_shape((MUTATIONS, CONTRACTION), data).unwrap();
    let mut out_view = ArrayViewMut2::from_shape((CLUSTERS, CONTRACTION), out).unwrap();
    let chunk_cols = CONTRACTION.div_ceil(THREADS);

    out_view
        .axis_chunks_iter_mut(Axis(1), chunk_cols)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_index, mut out_block)| {
            let col_start = chunk_index * chunk_cols;
            let cols = out_block.ncols();
            let data_block = data.slice(ndarray::s![.., col_start..col_start + cols]);
            general_mat_mul(1.0, &z_t, &data_block, 0.0, &mut out_block);
        });
}

fn main() {
    ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build_global()
        .unwrap();

    let mut rng = StdRng::seed_from_u64(0x5eed);
    let data = (0..MUTATIONS * CONTRACTION)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<_>>();
    let theta = (0..CONTRACTION * CLUSTERS)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<_>>();
    let z_t = (0..CLUSTERS * MUTATIONS)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<_>>();

    let mut z_ref = vec![0.0; MUTATIONS * CLUSTERS];
    let mut z_out = vec![0.0; MUTATIONS * CLUSTERS];
    let mut theta_ref = vec![0.0; CLUSTERS * CONTRACTION];
    let mut theta_out = vec![0.0; CLUSTERS * CONTRACTION];

    update_z_hand(&data, &theta, &mut z_ref);
    update_theta_hand(&z_t, &data, &mut theta_ref);

    let mut run_z = |name: &str, f: fn(&[f64], &[f64], &mut [f64])| {
        let (elapsed, sum) = time_it(|| {
            f(black_box(&data), black_box(&theta), black_box(&mut z_out));
            black_box(checksum(&z_out))
        });
        println!(
            "{name:34} {:8.3} ms diff={:.3e} checksum={:.6}",
            elapsed.as_secs_f64() * 1_000.0,
            max_abs_diff(&z_ref, &z_out),
            sum
        );
    };

    let mut run_theta = |name: &str, f: fn(&[f64], &[f64], &mut [f64])| {
        let (elapsed, sum) = time_it(|| {
            f(black_box(&z_t), black_box(&data), black_box(&mut theta_out));
            black_box(checksum(&theta_out))
        });
        println!(
            "{name:34} {:8.3} ms diff={:.3e} checksum={:.6}",
            elapsed.as_secs_f64() * 1_000.0,
            max_abs_diff(&theta_ref, &theta_out),
            sum
        );
    };

    println!(
        "mutations={MUTATIONS} clusters={CLUSTERS} contraction={CONTRACTION} threads={THREADS} median_of={ITERS}"
    );
    run_z("update_z hand row split", update_z_hand);
    run_z("update_z gemm row split", update_z_gemm_row_split);
    run_theta("update_theta hand cluster split", update_theta_hand);
    run_theta(
        "update_theta gemm cluster split",
        update_theta_gemm_cluster_split,
    );
    run_theta(
        "update_theta gemm column split",
        update_theta_gemm_column_split,
    );
}
