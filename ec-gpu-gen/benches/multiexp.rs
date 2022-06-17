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

fn gpu_multiexp(num_elements: usize) {
    let devices = Device::all();
    let mut kern = MultiexpKernel::<Bls12>::create(&devices).expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = rand::thread_rng();

    let bases = (0..num_elements)
        .map(|_| <Bls12 as Engine>::G1::random(&mut rng).to_affine())
        .collect::<Vec<_>>();

    let exponents = (0..num_elements)
        .map(|_| <Bls12 as Engine>::Fr::random(&mut rng).to_repr())
        .collect::<Vec<_>>();

    let (raw_bases, skip) = SourceBuilder::get((Arc::new(bases), 0));
    kern.multiexp(&pool, raw_bases, Arc::new(exponents), skip)
        .unwrap();
}

fn bench_multiexp(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("multiexp");
    let num_elements = (10..20).map(|shift| 1 << shift).collect::<Vec<_>>();
    for num in num_elements {
        group.bench_with_input(BenchmarkId::from_parameter(num), &num, |bencher, &num| {
            bencher.iter(|| {
                black_box(gpu_multiexp(num));
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_multiexp);
criterion_main!(benches);
