use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

use ec_gpu::GpuFieldName;
use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Curve, Group};
use log::{debug, error, info};
use rust_gpu_tools::{program_closures, Device, Program};
use yastl::Scope;

use crate::{
    error::{EcError, EcResult},
    program,
    threadpool::Worker,
};

// TODO vmx 2022-05-24: document what MAX_WINDOW_SIZE is about.
const MAX_WINDOW_SIZE: usize = 10;
//const LOCAL_WORK_SIZE: usize = 256;
const LOCAL_WORK_SIZE: usize = 128;
//const MEMORY_PADDING: f64 = 0.2f64; // Let 20% of GPU memory be free
const MEMORY_PADDING: f64 = 0.1f64; // Let 10% of GPU memory be free, this is an arbitrary value
                                    // The number of units the work is split into. One unit will result in one CUDA thread.
const NUM_WORK_UNITS: usize = 8192;

/// Divide and ceil to the next value.
const fn div_ceil(a: usize, b: usize) -> usize {
    if a % b == 0 {
        a / b
    } else {
        (a / b) + 1
    }
}

/// Multiexp kernel for a single GPU.
pub struct SingleMultiexpKernel<'a, G>
where
    G: PrimeCurveAffine,
{
    program: Program,
    //core_count: usize,
    /// The number of exponentiations the GPU can handle in a single execution of the kernel.
    n: usize,
    /// An optional function which will be called at places where it is possible to abort the
    /// multiexp calculations. If it returns true, the calculation will be aborted with an
    /// [`EcError::Aborted`].
    maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,

    _phantom: std::marker::PhantomData<G::Scalar>,
}

/// Calculates the window size, based on the given number of terms.
///
/// For best performance, the window size is reduced, so that maximum parallelism is possible. If
/// you e.g. have put only a subset of the terms into the GPU memory, then a smaller window size
/// leads to more windows, hence more units to work on, as we split the work into `num_windows *
/// num_groups`.
fn calc_window_size(num_terms: usize) -> usize {
    let window_size = (div_ceil(num_terms, NUM_WORK_UNITS) as f64).log2() as usize;
    debug!(
        "vmx: multiexp: vmx_calc_window_size: window_size: {}",
        window_size
    );
    std::cmp::min(window_size, MAX_WINDOW_SIZE)
}

/// Calculates the maximum number of terms that can be put onto the GPU memory.
fn calc_chunk_size<G>(mem: u64) -> usize
where
    G: PrimeCurveAffine,
    G::Scalar: PrimeField,
{
    // TODO vmx 2022-05-25: double check with actual numbers if that's really correct.
    let aff_size = std::mem::size_of::<G>();
    debug!("vmx: multiexp: calc_chunk_size: aff_size: {}", aff_size);
    let exp_size = exp_size::<G::Scalar>();
    debug!("vmx: multiexp: calc_chunk_size: exp_size: {}", exp_size);
    // TODO vmx 2022-05-25: double check with actual numbers if that's really correct.
    let proj_size = std::mem::size_of::<G::Curve>();
    debug!("vmx: multiexp: calc_chunk_size: proj_size: {}", proj_size);

    // Leave `MEMORY_PADDING` percent of the memory free.
    let max_memory = ((mem as f64) * (1f64 - MEMORY_PADDING)) as usize;
    // The amount of memory (in bytes) of a single term.
    let term_size = aff_size + exp_size;
    // The number of buckets needed for one work unit is `2^window_size - 1`.
    // TODO vmx 2022-06-07: Check why the global buffer allocation is not using the `- 1`.
    let max_buckets_per_work_unit = 1 << MAX_WINDOW_SIZE;
    // The amount of memory (in bytes) we need for the intermediate steps (buckets).
    let buckets_size = NUM_WORK_UNITS * max_buckets_per_work_unit * proj_size;
    // The amount of memory (in bytes) we need for the results.
    let results_size = NUM_WORK_UNITS * proj_size;

    (max_memory - buckets_size - results_size) / term_size
}

/// The size of the exponent in bytes.
///
/// It's the actual bytes size it needs in memory, not it's theoratical bit size.
fn exp_size<F: PrimeField>() -> usize {
    std::mem::size_of::<F::Repr>()
}

impl<'a, G> SingleMultiexpKernel<'a, G>
where
    G: PrimeCurveAffine,
{
    /// Create a new kernel for a device.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create(
        device: &Device,
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let mem = device.memory();
        let chunk_size = calc_chunk_size::<G>(mem);
        debug!(
            "vmx: multiexp: create: max chunk size for GPU: {}",
            chunk_size
        );

        let program = program::program(device)?;

        Ok(SingleMultiexpKernel {
            program,
            n: chunk_size,
            maybe_abort,
            _phantom: std::marker::PhantomData,
        })
    }

    // TODO vmx 2022-05-25: `n` is not needed, we can just use `bases.len()` instead (and perhaps
    // assert that `exps.len()` is the same.
    /// Run the actual multiexp computation on the GPU.
    ///
    /// The number of `bases` and `exponents` are determined by [`SingleMultiexpKernel::n`], this
    /// means that it is guaranteed that this amount of calculations fit on the GPU this kernel is
    /// running on.
    pub fn multiexp(
        &self,
        bases: &[G],
        exponents: &[<G::Scalar as PrimeField>::Repr],
        n: usize,
    ) -> EcResult<G::Curve>
    where
        G::Scalar: GpuFieldName,
        <G::Curve as Curve>::Base: GpuFieldName,
    {
        if let Some(maybe_abort) = &self.maybe_abort {
            if maybe_abort() {
                return Err(EcError::Aborted);
            }
        }
        debug!(
            "vmx: multiexp: number of exponentations on this GPU ({:?}): {}",
            std::any::type_name::<G>(),
            exponents.len()
        );

        let window_size = calc_window_size(bases.len());
        debug!("vmx: multiexp: window_size: {}", window_size);
        // windows_size * num_windows needs to be >= 256 in order for the kernel to work correctly.
        let num_windows = div_ceil(256, window_size);
        debug!("vmx: multiexp: num_windows: {}", num_windows);
        let num_groups = NUM_WORK_UNITS / num_windows;
        debug!("vmx: multiexp: num_groups: {}", num_groups);
        debug!(
            "vmx: multiexp: elements per groups: {}",
            div_ceil(exponents.len(), num_groups)
        );
        let bucket_len = 1 << window_size;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<G::Curve>> {
            let base_buffer = program.create_buffer_from_slice(bases)?;
            debug!(
                "vmx: multiexp: program: base buffer mem size: {}",
                std::mem::size_of::<G>() * bases.len()
            );
            let exp_buffer = program.create_buffer_from_slice(exponents)?;
            debug!(
                "vmx: multiexp: program: exp buffer mem size: {}",
                std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * exponents.len()
            );

            // It is safe as the GPU will initialize that buffer
            let bucket_buffer =
                unsafe { program.create_buffer::<G::Curve>(NUM_WORK_UNITS * bucket_len)? };
            debug!(
                "vmx: multiexp: program: bucket buffer mem size: {}",
                std::mem::size_of::<G::Curve>() * NUM_WORK_UNITS * bucket_len
            );
            // It is safe as the GPU will initialize that buffer
            let result_buffer = unsafe { program.create_buffer::<G::Curve>(NUM_WORK_UNITS)? };
            debug!(
                "vmx: multiexp: program: result buffer mem size: {}",
                std::mem::size_of::<G::Curve>() * NUM_WORK_UNITS
            );

            debug!(
                "vmx: multiexp: program: total alloced size: {}",
                std::mem::size_of::<G>() * bases.len()
                    + std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * exponents.len()
                    + std::mem::size_of::<G::Curve>() * NUM_WORK_UNITS * bucket_len
                    + std::mem::size_of::<G::Curve>() * NUM_WORK_UNITS
            );

            // The global work size follows CUDA's definition and is the number of
            // `LOCAL_WORK_SIZE` sized thread groups.
            let global_work_size = div_ceil(num_windows * num_groups, LOCAL_WORK_SIZE);
            debug!(
                "vmx: multiexp: program: global work size: {}",
                global_work_size
            );

            let kernel_name = format!(
                "{}_{}_bellman_multiexp",
                <G::Curve as Curve>::Base::name(),
                G::Scalar::name()
            );
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&base_buffer)
                .arg(&bucket_buffer)
                .arg(&result_buffer)
                .arg(&exp_buffer)
                .arg(&(n as u32))
                .arg(&(num_groups as u32))
                .arg(&(num_windows as u32))
                .arg(&(window_size as u32))
                .run()?;

            // TODO vmx 2022-06-09: I'm pretty sure this should be `num_windows * num_groups`
            // instead as it's the total number of threads
            let mut results = vec![G::Curve::identity(); NUM_WORK_UNITS];
            program.read_into_buffer(&result_buffer, &mut results)?;

            Ok(results)
        });

        let results = self.program.run(closures, ())?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = G::Curve::identity();
        let mut bits = 0;
        let exp_bits = exp_size::<G::Scalar>() * 8;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc = acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }
}

/// A struct that containts several multiexp kernels for different devices.
pub struct MultiexpKernel<'a, G>
where
    G: PrimeCurveAffine,
{
    kernels: Vec<SingleMultiexpKernel<'a, G>>,
}

impl<'a, G> MultiexpKernel<'a, G>
where
    G: PrimeCurveAffine,
    G::Scalar: GpuFieldName,
    <G::Curve as Curve>::Base: GpuFieldName,
{
    /// Create new kernels, one for each given device.
    pub fn create(devices: &[&Device]) -> EcResult<Self> {
        Self::create_optional_abort(devices, None)
    }

    /// Create new kernels, one for each given device, with early abort hook.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create_with_abort(
        devices: &[&Device],
        maybe_abort: &'a (dyn Fn() -> bool + Send + Sync),
    ) -> EcResult<Self> {
        Self::create_optional_abort(devices, Some(maybe_abort))
    }

    fn create_optional_abort(
        devices: &[&Device],
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let kernels: Vec<_> = devices
            .iter()
            .filter_map(|device| {
                let kernel = SingleMultiexpKernel::create(device, maybe_abort);
                if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name(),
                        e
                    );
                }
                kernel.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(EcError::Simple("No working GPUs found!"));
        }
        info!("Multiexp: {} working device(s) selected.", kernels.len());
        for (i, k) in kernels.iter().enumerate() {
            info!(
                "Multiexp: Device {}: {} (Chunk-size: {})",
                i,
                k.program.device_name(),
                k.n
            );
        }
        Ok(MultiexpKernel { kernels })
    }

    /// Calculate multiexp on all available GPUs.
    ///
    /// It needs to run within a [`yastl::Scope`]. This method usually isn't called directly, use
    /// [`MultiexpKernel::multiexp`] instead.
    pub fn parallel_multiexp<'s>(
        &'s mut self,
        scope: &Scope<'s>,
        bases: &'s [G],
        exps: &'s [<G::Scalar as PrimeField>::Repr],
        results: &'s mut [G::Curve],
        error: Arc<RwLock<EcResult<()>>>,
    ) {
        let num_devices = self.kernels.len();
        let num_exps = exps.len();
        // The maximum number of exponentiations per device.
        let chunk_size = ((num_exps as f64) / (num_devices as f64)).ceil() as usize;
        debug!(
            "vmx: multiexp: total number of exps2({:?}): {}",
            std::any::type_name::<G>(),
            num_exps
        );
        debug!(
            "vmx: multiexp: total number of exps per GPU ({:?}): {}",
            std::any::type_name::<G>(),
            chunk_size
        );

        for (((bases, exps), kern), result) in bases
            .chunks(chunk_size)
            .zip(exps.chunks(chunk_size))
            // NOTE vmx 2021-11-17: This doesn't need to be a mutable iterator. But when it isn't
            // there will be errors that the OpenCL CommandQueue cannot be shared between threads
            // safely.
            .zip(self.kernels.iter_mut())
            .zip(results.iter_mut())
        {
            let error = error.clone();
            scope.execute(move || {
                let mut acc = G::Curve::identity();
                debug!(
                    "vmx: multiexp: number of terms per kernel ({:?}): {}",
                    std::any::type_name::<G>(),
                    kern.n
                );
                for (bases, exps) in bases.chunks(kern.n).zip(exps.chunks(kern.n)) {
                    if error.read().unwrap().is_err() {
                        break;
                    }
                    match kern.multiexp(bases, exps, bases.len()) {
                        Ok(result) => acc.add_assign(&result),
                        Err(e) => {
                            *error.write().unwrap() = Err(e);
                            break;
                        }
                    }
                }
                if error.read().unwrap().is_ok() {
                    *result = acc;
                }
            });
        }
    }

    /// Calculate multiexp.
    ///
    /// This is the main entry point.
    pub fn multiexp(
        &mut self,
        pool: &Worker,
        bases_arc: Arc<Vec<G>>,
        exps: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
        skip: usize,
    ) -> EcResult<G::Curve> {
        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases_arc[skip..(skip + exps.len())];
        let exps = &exps[..];
        debug!(
            "vmx: multiexp: total number of based and exps ({}): {} {}",
            std::any::type_name::<G>(),
            &bases_arc.len(),
            &exps.len()
        );

        let mut results = Vec::new();
        let error = Arc::new(RwLock::new(Ok(())));

        pool.scoped(|s| {
            results = vec![G::Curve::identity(); self.kernels.len()];
            self.parallel_multiexp(s, bases, exps, &mut results, error.clone());
        });

        Arc::try_unwrap(error)
            .expect("only one ref left")
            .into_inner()
            .unwrap()?;

        let mut acc = G::Curve::identity();
        for r in results {
            acc.add_assign(&r);
        }

        Ok(acc)
    }

    /// Returns the number of kernels (one per device).
    pub fn num_kernels(&self) -> usize {
        self.kernels.len()
    }
}

#[cfg(feature = "bls12")]
#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Instant;

    use blstrs::Bls12;
    use ff::Field;
    use group::Curve;
    use pairing::Engine;

    use crate::multiexp_cpu::{multiexp_cpu, FullDensity, QueryDensity, SourceBuilder};

    fn multiexp_gpu<Q, D, G, S>(
        pool: &Worker,
        bases: S,
        density_map: D,
        exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
        kern: &mut MultiexpKernel<G>,
    ) -> Result<G::Curve, EcError>
    where
        for<'a> &'a Q: QueryDensity,
        D: Send + Sync + 'static + Clone + AsRef<Q>,
        G: PrimeCurveAffine,
        G::Scalar: GpuFieldName,
        <G::Curve as group::Curve>::Base: GpuFieldName,
        S: SourceBuilder<G>,
    {
        let exps = density_map.as_ref().generate_exps::<G::Scalar>(exponents);
        let (bss, skip) = bases.get();
        kern.multiexp(pool, bss, exps, skip).map_err(Into::into)
    }

    #[test]
    fn gpu_multiexp_consistency() {
        fil_logger::maybe_init();
        const MAX_LOG_D: usize = 28;
        const START_LOG_D: usize = 16;
        let devices = Device::all();
        let mut kern = MultiexpKernel::<<Bls12 as Engine>::G1Affine>::create(&devices)
            .expect("Cannot initialize kernel!");
        let pool = Worker::new();

        let mut rng = rand::thread_rng();

        let mut bases = (0..(1 << START_LOG_D))
            .map(|_| <Bls12 as Engine>::G1::random(&mut rng).to_affine())
            .collect::<Vec<_>>();

        for log_d in START_LOG_D..=MAX_LOG_D {
            let g = Arc::new(bases.clone());

            let samples = 1 << log_d;
            println!("Testing Multiexp for {} elements...", samples);

            let v = Arc::new(
                (0..samples)
                    .map(|_| <Bls12 as Engine>::Fr::random(&mut rng).to_repr())
                    .collect::<Vec<_>>(),
            );

            let mut now = Instant::now();
            let gpu =
                multiexp_gpu(&pool, (g.clone(), 0), FullDensity, v.clone(), &mut kern).unwrap();
            let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("GPU took {}ms.", gpu_dur);
            /*
                        now = Instant::now();
                        let cpu = multiexp_cpu(&pool, (g.clone(), 0), FullDensity, v.clone())
                            .wait()
                            .unwrap();
                        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
                        println!("CPU took {}ms.", cpu_dur);

                        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

                        assert_eq!(cpu, gpu);
            */
            println!("============================");

            bases = [bases.clone(), bases.clone()].concat();
        }
    }
}
