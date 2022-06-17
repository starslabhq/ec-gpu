use std::sync::Arc;

use blstrs::Bls12;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ec_gpu_gen::multiexp::MultiexpKernel;
use ec_gpu_gen::multiexp_cpu::SourceBuilder;
use ec_gpu_gen::threadpool::Worker;
use ff::{Field, PrimeField};
use group::{Curve, Group};
use pairing::Engine;
use rust_gpu_tools::Device;

fn bench_multiexp(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("multiexp");
    // The difference between runs is so little, hence a low sample size is OK.
    group.sample_size(10);

    let devices = Device::all();
    let mut kern = MultiexpKernel::<Bls12>::create(&devices).expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = rand::thread_rng();

    let num_elements = (10..20).map(|shift| 1 << shift).collect::<Vec<_>>();
    for num in num_elements {
        group.bench_with_input(BenchmarkId::from_parameter(num), &num, |bencher, &num| {
            let bases_raw = (0..num)
                .map(|_| <Bls12 as Engine>::G1::random(&mut rng).to_affine())
                .collect::<Vec<_>>();

            let exponents_raw = (0..num)
                .map(|_| <Bls12 as Engine>::Fr::random(&mut rng).to_repr())
                .collect::<Vec<_>>();
            let exponents = Arc::new(exponents_raw);

            let (bases, skip) = SourceBuilder::get((Arc::new(bases_raw), 0));

            bencher.iter(|| {
                black_box(
                    kern.multiexp(&pool, bases.clone(), exponents.clone(), skip)
                        .unwrap(),
                );
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_multiexp);
criterion_main!(benches);
