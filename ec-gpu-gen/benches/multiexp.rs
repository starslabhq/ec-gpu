use criterion::{black_box, criterion_group, criterion_main, Criterion};

    use std::time::Instant;
use std::sync::{Arc, RwLock};

    use blstrs::Bls12;
    use ff::Field;
    use group::Curve;

use ec_gpu_gen::multiexp_cpu::FullDensity;
 use ec_gpu_gen::threadpool::Worker;
 use ec_gpu_gen::multiexp::MultiexpKernel;
use rust_gpu_tools::Device;
use ec_gpu_gen::EcError;
use ec_gpu_gen::multiexp_cpu::SourceBuilder;
use ec_gpu::GpuEngine;
use group::prime::PrimeCurveAffine;
use pairing::Engine;
use ec_gpu_gen::multiexp_cpu::QueryDensity;
use group::Group;
 use ff::PrimeField;

    //use crate::multiexp_cpu::{multiexp_cpu, FullDensity, QueryDensity, SourceBuilder};

    fn multiexp_gpu<Q, D, G, E, S>(
        pool: &Worker,
        bases: S,
        density_map: D,
        exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
        kern: &mut MultiexpKernel<E>,
    ) -> Result<<G as PrimeCurveAffine>::Curve, EcError>
    where
        for<'a> &'a Q: QueryDensity,
        D: Send + Sync + 'static + Clone + AsRef<Q>,
        G: PrimeCurveAffine,
        E: GpuEngine,
        E: Engine<Fr = G::Scalar>,
        S: SourceBuilder<G>,
    {
        let exps = density_map.as_ref().generate_exps::<E>(exponents);
        let (bss, skip) = bases.get();
        kern.multiexp(pool, bss, exps, skip).map_err(Into::into)
    }

    fn gpu_multiexp_consistency(num_elements: usize) {
        let devices = Device::all();
        let mut kern =
            MultiexpKernel::<Bls12>::create(&devices).expect("Cannot initialize kernel!");
        let pool = Worker::new();

        let mut rng = rand::thread_rng();

        let bases = (0..num_elements)
            .map(|_| <Bls12 as Engine>::G1::random(&mut rng).to_affine())
            .collect::<Vec<_>>();

        let exponents = (0..num_elements)
            .map(|_| <Bls12 as Engine>::Fr::random(&mut rng).to_repr())
            .collect::<Vec<_>>();

        //multiexp_gpu(&pool, (bases, 0), FullDensity, exponents, &mut kern).unwrap();
        //let exps = density_map.as_ref().generate_exps::<E>(exponents);
        let (raw_bases, skip) = SourceBuilder::get((Arc::new(bases), 0));
        kern.multiexp(&pool, raw_bases, Arc::new(exponents), skip);
    }

pub fn criterion_benchmark(c: &mut Criterion) {
   //c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
   c.bench_function("multiexp", |bencher| {
       bencher.iter(|| {
           black_box(
               gpu_multiexp_consistency(1024)
           );
       })
   });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
