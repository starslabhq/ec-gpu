use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

use ec_gpu::GpuFieldName;
use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Curve, Group};
use log::{debug, error, info, warn};
use rust_gpu_tools::{program_closures, Device, Program, CUDA_CORES};
use yastl::Scope;

use crate::{
    error::{EcError, EcResult},
    program,
    threadpool::Worker,
};

// TODO vmx 2022-05-24: document what MAX_WINDOW_SIZE is about.
const MAX_WINDOW_SIZE: usize = 10;
const LOCAL_WORK_SIZE: usize = 256;
const MEMORY_PADDING: f64 = 0.2f64; // Let 20% of GPU memory be free
const DEFAULT_CUDA_CORES: usize = 2560;

/// Divide and ceil to the next value.
const fn div_ceil(a: usize, b: usize) -> usize {
    if a % b == 0 {
        a / b
    } else {
        (a / b) + 1
    }
}

fn get_cuda_cores_count(name: &str) -> usize {
    *CUDA_CORES.get(name).unwrap_or_else(|| {
        warn!(
            "Number of CUDA cores for your device ({}) is unknown! Best performance is only \
            achieved when the number of CUDA cores is known! You can find the instructions on \
            how to support custom GPUs here: https://docs.rs/rust-gpu-tools",
            name
        );
        &DEFAULT_CUDA_CORES
    })
}

/// Multiexp kernel for a single GPU.
pub struct SingleMultiexpKernel<'a, G>
where
    G: PrimeCurveAffine,
{
    program: Program,
    core_count: usize,
    /// The number of exponentiations the GPU can handle in a single execution of the kernel.
    n: usize,
    /// An optional function which will be called at places where it is possible to abort the
    /// multiexp calculations. If it returns true, the calculation will be aborted with an
    /// [`EcError::Aborted`].
    maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,

    _phantom: std::marker::PhantomData<G::Scalar>,
}

fn calc_num_groups(core_count: usize, num_windows: usize) -> usize {
    // Observations show that we get the best performance when num_groups * num_windows ~= 2 * CUDA_CORES
    2 * core_count / num_windows
}

/// Calculates the window size, based on the given number of terms.
///
/// For best performance, the window size is reduced, so that maximum parallelism is possible. If
/// you e.g. have put only a subset of the terms into the GPU memory, then a smaller window size
/// leads to more windows, hence more units to work on, as we split the work into `num_windows *
/// num_groups`.
// TODO vmx 2022-05-30: change from `core_count` to `work_units`, which is `2 * core_count`.
fn calc_window_size(n: usize, exp_bits: usize, core_count: usize) -> usize {
    // window_size = ln(n / num_groups)
    // num_windows = exp_bits / window_size
    // num_groups = 2 * core_count / num_windows = 2 * core_count * window_size / exp_bits
    // window_size = ln(n / num_groups) = ln(n * exp_bits / (2 * core_count * window_size))
    // window_size = ln(exp_bits * n / (2 * core_count)) - ln(window_size)
    //
    // Thus we need to solve the following equation:
    // window_size + ln(window_size) = ln(exp_bits * n / (2 * core_count))
    let exp_bits_per_work_unit = div_ceil(exp_bits * n, 2 * core_count);
    let lower_bound = (exp_bits_per_work_unit as f64).ln();
    for w in 0..MAX_WINDOW_SIZE {
        if (w as f64) + (w as f64).ln() > lower_bound {
            return w;
        }
    }

    MAX_WINDOW_SIZE
}

/// Calculates the number of terms that could optimally be calculated on the GPU, if the GPU had
/// an unlimited amount of memory.
// TODO vmx 2022-05-30: change from `core_count` to `work_units`, which is `2 * core_count`.
// TODO vmx 2022-06-07: Try to find out why `e^window_size` is optimal.
fn calc_best_chunk_size(max_window_size: usize, core_count: usize, exp_bits: usize) -> usize {
    // Best chunk-size (N) can also be calculated using the same logic as calc_window_size:
    // n = e^window_size * window_size * 2 * core_count / exp_bits
    //let e_window_size = (max_window_size as f64).exp().ceil() as usize;
    //let work_units = 2 * core_count;
    //div_ceil(
    //    e_window_size * max_window_size * work_units,
    //    exp_bits,
    //)
    (((max_window_size as f64).exp() as f64)
       * (max_window_size as f64)
       * 2f64
       * (core_count as f64)
       / (exp_bits as f64))
       .ceil() as usize
}

/// Calculates the maximum number of terms that can be put onto the GPU memory.
// The core count is needed as it determines how much auxiliary space we need.
// TODO vmx 2022-05-30: change from `core_count` to `work_units`, which is `2 * core_count`.
// TODO vmx 2022-05-30: rename to `calc_max_chunk_size`.
fn calc_chunk_size<G>(mem: u64, core_count: usize) -> usize
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
    let work_units = 2 * core_count;
    // The amount of memory (in bytes) we need for the intermediate steps (buckets).
    let buckets_size = work_units * max_buckets_per_work_unit * proj_size;
    // The amount of memory (in bytes) we need for the results.
    let results_size = work_units * proj_size;

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
        let exp_bits = exp_size::<G::Scalar>() * 8;
        let core_count = get_cuda_cores_count(&device.name());
        let mem = device.memory();
        let max_n = calc_chunk_size::<G>(mem, core_count);
        debug!("vmx: multiexp: create: max chunk size for GPU: {}", max_n);
        let best_n = calc_best_chunk_size(MAX_WINDOW_SIZE, core_count, exp_bits);
        debug!(
            "vmx: multiexp: create: best chunk size possible: {}",
            best_n
        );
        let n = std::cmp::min(max_n, best_n);
        debug!("vmx: multiexp: create: actual chunk size: {}", n);

        let program = program::program(device)?;

        Ok(SingleMultiexpKernel {
            program,
            core_count,
            n,
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

        let exp_bits = exp_size::<G::Scalar>() * 8;
        let window_size = calc_window_size(n as usize, exp_bits, self.core_count);
        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = calc_num_groups(self.core_count, num_windows);
        let bucket_len = 1 << window_size;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<G::Curve>> {
            let base_buffer = program.create_buffer_from_slice(bases)?;
            debug!("vmx: multiexp: program: base buffer mem size: {}", std::mem::size_of::<G>() * bases.len());
            let exp_buffer = program.create_buffer_from_slice(exponents)?;
            debug!("vmx: multiexp: program: exp buffer mem size: {}", std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * exponents.len());

            // It is safe as the GPU will initialize that buffer
            let bucket_buffer =
                unsafe { program.create_buffer::<G::Curve>(2 * self.core_count * bucket_len)? };
            debug!("vmx: multiexp: program: bucket buffer mem size: {}", std::mem::size_of::<G::Curve>() * 2 * self.core_count * bucket_len);
            // It is safe as the GPU will initialize that buffer
            let result_buffer = unsafe { program.create_buffer::<G::Curve>(2 * self.core_count)? };
            debug!("vmx: multiexp: program: result buffer mem size: {}", std::mem::size_of::<G::Curve>() * 2 * self.core_count);

            debug!("vmx: multiexp: program: total alloced size: {}", std::mem::size_of::<G>() * bases.len() + std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * exponents.len() + std::mem::size_of::<G::Curve>() * 2 * self.core_count * bucket_len +  std::mem::size_of::<G::Curve>() * 2 * self.core_count);

            // The global work size follows CUDA's definition and is the number of
            // `LOCAL_WORK_SIZE` sized thread groups.
            let global_work_size = div_ceil(num_windows * num_groups, LOCAL_WORK_SIZE);

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

            let mut results = vec![G::Curve::identity(); 2 * self.core_count];
            program.read_into_buffer(&result_buffer, &mut results)?;

            Ok(results)
        });

        let results = self.program.run(closures, ())?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = G::Curve::identity();
        let mut bits = 0;
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

    fn num_work_units(&self) -> usize {
        self.core_count * 2
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
        const MAX_LOG_D: usize = 16;
fil_logger::maybe_init();
        const START_LOG_D: usize = 10;
        let devices = Device::all();
        let mut kern = MultiexpKernel::<<Bls12 as Engine>::G1Affine>::create(&devices)
            .expect("Cannot initialize kernel!");
        let pool = Worker::new();

        let mut rng = rand::thread_rng();

        let mut bases = (0..(1 << 10))
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

            now = Instant::now();
            let cpu = multiexp_cpu(&pool, (g.clone(), 0), FullDensity, v.clone())
                .wait()
                .unwrap();
            let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("CPU took {}ms.", cpu_dur);

            println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

            assert_eq!(cpu, gpu);

            println!("============================");

            bases = [bases.clone(), bases.clone()].concat();
        }
    }
}
