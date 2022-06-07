use std::collections::HashSet;
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;

use ec_gpu::{GpuField, GpuFieldName};

static COMMON_SRC: &str = include_str!("cl/common.cl");
static FIELD_SRC: &str = include_str!("cl/field.cl");
static FIELD2_SRC: &str = include_str!("cl/field2.cl");
static EC_SRC: &str = include_str!("cl/ec.cl");
static FFT_SRC: &str = include_str!("cl/fft.cl");
static MULTIEXP_SRC: &str = include_str!("cl/multiexp.cl");

/// This trait is used to uniquely identify items by some identifier (`name`) and to return the GPU
/// source code they produce.
trait NameAndSource<L: Limb> {
    /// The name to identify the item.
    fn name(&self) -> String;
    /// The GPU source code that is generated.
    fn source(&self) -> String;
}

impl<L: Limb> PartialEq for dyn NameAndSource<L> {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl<L: Limb> Eq for dyn NameAndSource<L> {}

impl<L: Limb> Hash for dyn NameAndSource<L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

/// Prints the name by default, the source code in the alternate mode via `{:#?}`.
impl<L: Limb> fmt::Debug for dyn NameAndSource<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            f.debug_map()
                .entries(vec![("name", self.name()), ("source", self.source())])
                .finish()
        } else {
            write!(f, "{:?}", self.name())
        }
    }
}

/// A field that might even be an extension field.
///
/// If it is an extension field, it's generic over the sub-field, the extension field is stored
/// as a string only.
// Storing the extension field as a string is a bit of a hack around Rust's type system. If we
// would store the generic type, then the enum would need to be generic over two fields, even in
// the case when no extension field is used. This would make the API harder to use.
#[derive(Debug)]
pub enum Field<F: GpuField> {
    // A field.
    Field(PhantomData<F>),
    /// An extension field with the given name.
    ExtensionField(String),
}

impl<F: GpuField> Clone for Field<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Field(_) => Self::Field(PhantomData),
            Self::ExtensionField(name) => Self::ExtensionField(name.to_string()),
        }
    }
}

impl<F: GpuField> Field<F> {
    /// Create a new field for the given generic type.
    pub fn new() -> Self {
        Self::Field(PhantomData)
    }

    /// Create an extension field.
    ///
    /// The sub-field is specified by generic on the field, the extension field by the generic on
    /// this function.
    pub fn quadratic_extension<E: GpuFieldName>() -> Self {
        Self::ExtensionField(E::name())
    }

    /// If it is an extension field, the extension field's identifier is used as name.
    pub fn name(&self) -> String {
        match self {
            Self::Field(_) => F::name(),
            Self::ExtensionField(name) => name.to_string(),
        }
    }
}

impl<F: GpuField, L: Limb> NameAndSource<L> for Field<F> {
    fn name(&self) -> String {
        self.name()
    }

    fn source(&self) -> String {
        match self {
            Self::Field(_) => [
                params::<F, L>(),
                field_add_sub_nvidia::<F, L>().expect("preallocated"),
                String::from(FIELD_SRC),
            ]
            .join("\n")
            .replace("FIELD", &F::name()),
            Self::ExtensionField(name) => String::from(FIELD2_SRC)
                .replace("FIELD2", name)
                .replace("FIELD", &F::name()),
        }
    }
}

/// Struct that generates FFT GPU source code.
struct Fft<F: GpuField> {
    /// The field the FFT operates on.
    field: Field<F>,
}

impl<F: GpuField, L: Limb> NameAndSource<L> for Fft<F> {
    fn name(&self) -> String {
        self.field.name()
    }

    fn source(&self) -> String {
        String::from(FFT_SRC).replace("FIELD", &self.field.name())
    }
}

/// Struct that generates multiexp GPU source code.
struct Multiexp<F: GpuField, Exp: GpuField> {
    /// Base field to use, may also be an extension field.
    field: Field<F>,
    /// The scalar field that is used for the exponent.
    exponent: Field<Exp>,
}

impl<F: GpuField, Exp: GpuField, L: Limb> NameAndSource<L> for Multiexp<F, Exp> {
    /// Multiexp depends on the base as well as the scalar field, hence use both as identifier.
    fn name(&self) -> String {
        format!("{}_{}", self.field.name(), self.exponent.name())
    }

    fn source(&self) -> String {
        let ec = String::from(EC_SRC).replace("FIELD", &self.field.name());
        let multiexp = String::from(MULTIEXP_SRC)
            .replace("FIELD", &self.field.name())
            .replace("EXPONENT", &self.exponent.name());
        [ec, multiexp].concat()
    }
}

/// Configuration to create the source code of a GPU kernel.
///
/// # Example
///
/// ```
/// use blstrs::{Fp, Fp2, Scalar};
///
/// let source = Config::new()
///     .add_fft(Field::<Scalar>::new())
///     // G1
///     .add_multiexp(Field::<Fp>::new(), Field::<Scalar>::new())
///     // G2
///     .add_multiexp(
///         Field::<Fp>::quadratic_extension::<Fp2>(),
///         Field::<Scalar>::new(),
///     )
///     .gen_source();
///```
// In the `HashSet`s the concrete types cannot be used, as each item of the set should be able to
// have its own (different) generic type.
pub struct Config<L: Limb> {
    /// The [`Field`]s that are used in this kernel.
    fields: HashSet<Box<dyn NameAndSource<L>>>,
    /// The extension [`Field`]s that are used in this kernel.
    extension_fields: HashSet<Box<dyn NameAndSource<L>>>,
    /// The extension-[`Fft`]s that are used in this kernel.
    ffts: HashSet<Box<dyn NameAndSource<L>>>,
    /// The [`Multiexp`]s that are used in this kernel.
    multiexps: HashSet<Box<dyn NameAndSource<L>>>,
}

impl<L: Limb> Config<L> {
    /// Create a new configuration to generation a GPU kernel.
    pub fn new() -> Self {
        Self {
            fields: HashSet::new(),
            extension_fields: HashSet::new(),
            ffts: HashSet::new(),
            multiexps: HashSet::new(),
        }
    }

    /// Add a field to the configuration.
    ///
    /// If it is an extension field, then the extension field *and* the sub-field is added.
    pub fn add_field<F>(mut self, field: Field<F>) -> Self
    where
        F: GpuField + 'static,
    {
        match field {
            Field::Field(_) => {
                self.fields.insert(Box::new(field));
            }
            Field::ExtensionField(_) => {
                self.extension_fields.insert(Box::new(field));
                // Also add the sub-field.
                let sub_field = Field::<F>::new();
                self.fields.insert(Box::new(sub_field));
            }
        }
        self
    }

    /// Add an FFT kernel function to the configuration.
    pub fn add_fft<F>(self, field: Field<F>) -> Self
    where
        F: GpuField + 'static,
    {
        let mut config = self.add_field(field.clone());
        let fft = Fft { field };
        config.ffts.insert(Box::new(fft));
        config
    }

    /// Add an Multiexp kernel function to the configuration.
    pub fn add_multiexp<F, Exp>(self, field: Field<F>, exponent: Field<Exp>) -> Self
    where
        F: GpuField + 'static,
        Exp: GpuField + 'static,
    {
        let mut config = self.add_field(field.clone());
        let multiexp = Multiexp { field, exponent };
        config.multiexps.insert(Box::new(multiexp));
        config
    }

    /// Generate the GPU kernel source code based on the current configuration.
    pub fn gen_source(&self) -> String {
        let fields = self.fields.iter().map(|field| field.source()).collect();
        let extension_fields = self
            .extension_fields
            .iter()
            .map(|field| field.source())
            .collect();
        let ffts = self.ffts.iter().map(|fft| fft.source()).collect();
        let multiexps = self
            .multiexps
            .iter()
            .map(|multiexp| multiexp.source())
            .collect();
        vec![
            COMMON_SRC.to_string(),
            fields,
            extension_fields,
            ffts,
            multiexps,
        ]
        .join("\n\n")
    }
}

/// Trait to implement limbs of different underlying bit sizes.
pub trait Limb: Sized + Clone + Copy {
    /// The underlying size of the limb, e.g. `u32`
    type LimbType: Clone + std::fmt::Display;
    /// Returns the value representing zero.
    fn zero() -> Self;
    /// Returns a new limb.
    fn new(val: Self::LimbType) -> Self;
    /// Returns the raw value of the limb.
    fn value(&self) -> Self::LimbType;
    /// Returns the bit size of the limb.
    fn bits() -> usize {
        mem::size_of::<Self::LimbType>() * 8
    }
    /// Returns a tuple with the strings that PTX is using to describe the type and the register.
    fn ptx_info() -> (&'static str, &'static str);
    /// Returns the type that OpenCL is using to represent the limb.
    fn opencl_type() -> &'static str;
    /// Returns the limbs that represent the multiplicative identity of the given field.
    fn one_limbs<F: GpuField>() -> Vec<Self>;
    /// Returns the field modulus in non-Montgomery form as a vector of `Self::LimbType` (least
    /// significant limb first).
    fn modulus_limbs<F: GpuField>() -> Vec<Self>;
    /// Calculate the `INV` parameter of Montgomery reduction algorithm for 32/64bit limbs
    /// * `a` - Is the first limb of modulus.
    fn calc_inv(a: Self) -> Self;
    /// Returns the limbs that represent `R ^ 2 mod P`.
    fn calculate_r2<F: GpuField>() -> Vec<Self>;
}

/// A 32-bit limb.
#[derive(Clone, Copy)]
pub struct Limb32(u32);
impl Limb for Limb32 {
    type LimbType = u32;
    fn zero() -> Self {
        Self(0)
    }
    fn new(val: Self::LimbType) -> Self {
        Self(val)
    }
    fn value(&self) -> Self::LimbType {
        self.0
    }
    fn ptx_info() -> (&'static str, &'static str) {
        ("u32", "r")
    }
    fn opencl_type() -> &'static str {
        "uint"
    }
    fn one_limbs<F: GpuField>() -> Vec<Self> {
        F::one().into_iter().map(Self::new).collect()
    }
    fn modulus_limbs<F: GpuField>() -> Vec<Self> {
        F::modulus().into_iter().map(Self::new).collect()
    }
    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u32;
        for _ in 0..31 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }
    fn calculate_r2<F: GpuField>() -> Vec<Self> {
        F::r2().into_iter().map(Self::new).collect()
    }
}

/// A 64-bit limb.
#[derive(Clone, Copy)]
pub struct Limb64(u64);
impl Limb for Limb64 {
    type LimbType = u64;
    fn zero() -> Self {
        Self(0)
    }
    fn new(val: Self::LimbType) -> Self {
        Self(val)
    }
    fn value(&self) -> Self::LimbType {
        self.0
    }
    fn ptx_info() -> (&'static str, &'static str) {
        ("u64", "l")
    }
    fn opencl_type() -> &'static str {
        "ulong"
    }
    fn one_limbs<F: GpuField>() -> Vec<Self> {
        F::one()
            .chunks(2)
            .map(|chunk| Self::new(((chunk[1] as u64) << 32) + (chunk[0] as u64)))
            .collect()
    }

    fn modulus_limbs<F: GpuField>() -> Vec<Self> {
        F::modulus()
            .chunks(2)
            .map(|chunk| Self::new(((chunk[1] as u64) << 32) + (chunk[0] as u64)))
            .collect()
    }

    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u64;
        for _ in 0..63 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }
    fn calculate_r2<F: GpuField>() -> Vec<Self> {
        F::r2()
            .chunks(2)
            .map(|chunk| Self::new(((chunk[1] as u64) << 32) + (chunk[0] as u64)))
            .collect()
    }
}

fn const_field<L: Limb>(name: &str, limbs: Vec<L>) -> String {
    format!(
        "CONSTANT FIELD {} = {{ {{ {} }} }};",
        name,
        limbs
            .iter()
            .map(|l| l.value().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Generates CUDA/OpenCL constants and type definitions of prime-field `F`
fn params<F, L: Limb>() -> String
where
    F: GpuField,
{
    let one = L::one_limbs::<F>(); // Get Montgomery form of F::one()
    let p = L::modulus_limbs::<F>(); // Get field modulus in non-Montgomery form
    let r2 = L::calculate_r2::<F>();
    let limbs = one.len(); // Number of limbs
    let inv = L::calc_inv(p[0]);
    let limb_def = format!("#define FIELD_limb {}", L::opencl_type());
    let limbs_def = format!("#define FIELD_LIMBS {}", limbs);
    let limb_bits_def = format!("#define FIELD_LIMB_BITS {}", L::bits());
    let p_def = const_field("FIELD_P", p);
    let r2_def = const_field("FIELD_R2", r2);
    let one_def = const_field("FIELD_ONE", one);
    let zero_def = const_field("FIELD_ZERO", vec![L::zero(); limbs]);
    let inv_def = format!("#define FIELD_INV {}", inv.value());
    let typedef = "typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;".to_string();
    [
        limb_def,
        limbs_def,
        limb_bits_def,
        inv_def,
        typedef,
        one_def,
        p_def,
        r2_def,
        zero_def,
    ]
    .join("\n")
}

/// Generates PTX-Assembly implementation of FIELD_add_/FIELD_sub_
fn field_add_sub_nvidia<F, L: Limb>() -> Result<String, std::fmt::Error>
where
    F: GpuField,
{
    let mut result = String::new();
    let (ptx_type, ptx_reg) = L::ptx_info();

    writeln!(result, "#if defined(OPENCL_NVIDIA) || defined(CUDA)\n")?;
    for op in &["sub", "add"] {
        let len = L::one_limbs::<F>().len();

        writeln!(
            result,
            "DEVICE FIELD FIELD_{}_nvidia(FIELD a, FIELD b) {{",
            op
        )?;
        if len > 1 {
            write!(result, "asm(")?;
            writeln!(result, "\"{}.cc.{} %0, %0, %{};\\r\\n\"", op, ptx_type, len)?;

            for i in 1..len - 1 {
                writeln!(
                    result,
                    "\"{}c.cc.{} %{}, %{}, %{};\\r\\n\"",
                    op,
                    ptx_type,
                    i,
                    i,
                    len + i
                )?;
            }
            writeln!(
                result,
                "\"{}c.{} %{}, %{}, %{};\\r\\n\"",
                op,
                ptx_type,
                len - 1,
                len - 1,
                2 * len - 1
            )?;

            write!(result, ":")?;
            for n in 0..len {
                write!(result, "\"+{}\"(a.val[{}])", ptx_reg, n)?;
                if n != len - 1 {
                    write!(result, ", ")?;
                }
            }

            write!(result, "\n:")?;
            for n in 0..len {
                write!(result, "\"{}\"(b.val[{}])", ptx_reg, n)?;
                if n != len - 1 {
                    write!(result, ", ")?;
                }
            }
            writeln!(result, ");")?;
        }
        writeln!(result, "return a;\n}}")?;
    }
    writeln!(result, "#endif")?;

    Ok(result)
}

#[cfg(all(test, any(feature = "opencl", feature = "cuda")))]
mod tests {
    use super::*;

    use std::sync::Mutex;

    #[cfg(feature = "cuda")]
    use rust_gpu_tools::cuda;
    #[cfg(feature = "opencl")]
    use rust_gpu_tools::opencl;
    use rust_gpu_tools::{program_closures, Device, GPUError, Program};

    use blstrs::Scalar;
    use ff::{Field as _, PrimeField};
    use lazy_static::lazy_static;
    use rand::{thread_rng, Rng};

    static TEST_SRC: &str = include_str!("./cl/test.cl");

    #[derive(PartialEq, Debug, Clone, Copy)]
    #[repr(transparent)]
    pub struct GpuScalar(pub Scalar);
    impl Default for GpuScalar {
        fn default() -> Self {
            Self(Scalar::zero())
        }
    }

    #[cfg(feature = "cuda")]
    impl cuda::KernelArgument for GpuScalar {
        fn as_c_void(&self) -> *mut std::ffi::c_void {
            &self.0 as *const _ as _
        }
    }

    #[cfg(feature = "opencl")]
    impl opencl::KernelArgument for GpuScalar {
        fn push(&self, kernel: &mut opencl::Kernel) {
            kernel.builder.set_arg(&self.0);
        }
    }

    /// The `run` call needs to return a result, use this struct as placeholder.
    #[derive(Debug)]
    struct NoError;
    impl From<GPUError> for NoError {
        fn from(_error: GPUError) -> Self {
            Self
        }
    }

    fn test_source<L: Limb>() -> String {
        let field = Field::<Scalar>::new();
        let field_source = Config::<L>::new()
            .add_field::<Scalar>(field.clone())
            .gen_source();
        let test_source = String::from(TEST_SRC).replace("FIELD", &field.name());
        [field_source, test_source].concat()
    }

    #[cfg(feature = "cuda")]
    lazy_static! {
        static ref CUDA_PROGRAM: Mutex<Program> = {
            use std::ffi::CString;
            use std::fs;
            use std::process::Command;

            let tmpdir = tempfile::tempdir().expect("Cannot create temporary directory.");
            let source_path = tmpdir.path().join("kernel.cu");
            fs::write(&source_path, test_source::<Limb32>().as_bytes())
                .expect("Cannot write kernel source file.");
            let fatbin_path = tmpdir.path().join("kernel.fatbin");

            let nvcc = Command::new("nvcc")
                .arg("--fatbin")
                .arg("--threads=0")
                .arg("--gpu-architecture=sm_86")
                .arg("--generate-code=arch=compute_86,code=sm_86")
                .arg("--generate-code=arch=compute_80,code=sm_80")
                .arg("--generate-code=arch=compute_75,code=sm_75")
                .arg("--output-file")
                .arg(&fatbin_path)
                .arg(&source_path)
                .status()
                .expect("Cannot run nvcc.");

            if !nvcc.success() {
                panic!(
                    "nvcc failed. See the kernel source at {}.",
                    source_path.display()
                );
            }

            let device = *Device::all().first().expect("Cannot get a default device.");
            let cuda_device = device.cuda_device().unwrap();
            let fatbin_path_cstring =
                CString::new(fatbin_path.to_str().expect("path is not valid UTF-8."))
                    .expect("path contains NULL byte.");
            let program =
                cuda::Program::from_binary(cuda_device, fatbin_path_cstring.as_c_str()).unwrap();
            Mutex::new(Program::Cuda(program))
        };
    }

    #[cfg(feature = "opencl")]
    lazy_static! {
        static ref OPENCL_PROGRAM: Mutex<(Program, Program)> = {
            let device = *Device::all().first().expect("Cannot get a default device");
            let opencl_device = device.opencl_device().unwrap();
            let program_32 =
                opencl::Program::from_opencl(opencl_device, &test_source::<Limb32>()).unwrap();
            let program_64 =
                opencl::Program::from_opencl(opencl_device, &test_source::<Limb64>()).unwrap();
            Mutex::new((Program::Opencl(program_32), Program::Opencl(program_64)))
        };
    }

    fn call_kernel(name: &str, scalars: &[GpuScalar], uints: &[u32]) -> Scalar {
        let closures = program_closures!(|program, _args| -> Result<Scalar, NoError> {
            let mut cpu_buffer = vec![GpuScalar::default()];
            let buffer = program.create_buffer_from_slice(&cpu_buffer).unwrap();

            let mut kernel = program.create_kernel(name, 1, 64).unwrap();
            for scalar in scalars {
                kernel = kernel.arg(scalar);
            }
            for uint in uints {
                kernel = kernel.arg(uint);
            }
            kernel.arg(&buffer).run().unwrap();

            program.read_into_buffer(&buffer, &mut cpu_buffer).unwrap();
            Ok(cpu_buffer[0].0)
        });

        // For CUDA we only test 32-bit limbs.
        #[cfg(all(feature = "cuda", not(feature = "opencl")))]
        return CUDA_PROGRAM.lock().unwrap().run(closures, ()).unwrap();

        // For OpenCL we test for 32 and 64-bi limbs.
        #[cfg(all(feature = "opencl", not(feature = "cuda")))]
        {
            let result_32 = OPENCL_PROGRAM.lock().unwrap().0.run(closures, ()).unwrap();
            let result_64 = OPENCL_PROGRAM.lock().unwrap().1.run(closures, ()).unwrap();
            assert_eq!(
                result_32, result_64,
                "Results for 32-bit and 64-bit limbs must be the same."
            );
            result_32
        }

        // When both features are enabled, check if the results are the same
        #[cfg(all(feature = "cuda", feature = "opencl"))]
        {
            let cuda_result = CUDA_PROGRAM.lock().unwrap().run(closures, ()).unwrap();
            let opencl_32_result = OPENCL_PROGRAM.lock().unwrap().0.run(closures, ()).unwrap();
            let opencl_64_result = OPENCL_PROGRAM.lock().unwrap().1.run(closures, ()).unwrap();
            assert_eq!(
                opencl_32_result, opencl_64_result,
                "Results for 32-bit and 64-bit limbs on OpenCL must be the same."
            );
            assert_eq!(
                cuda_result, opencl_32_result,
                "Results for CUDA and OpenCL must be the same."
            );
            cuda_result
        }
    }

    #[test]
    fn test_add() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = Scalar::random(&mut rng);
            let c = a + b;

            assert_eq!(
                call_kernel("test_add", &[GpuScalar(a), GpuScalar(b)], &[]),
                c
            );
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = Scalar::random(&mut rng);
            let c = a - b;
            assert_eq!(
                call_kernel("test_sub", &[GpuScalar(a), GpuScalar(b)], &[]),
                c
            );
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = Scalar::random(&mut rng);
            let c = a * b;

            assert_eq!(
                call_kernel("test_mul", &[GpuScalar(a), GpuScalar(b)], &[]),
                c
            );
        }
    }

    #[test]
    fn test_pow() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = rng.gen::<u32>();
            let c = a.pow_vartime([b as u64]);
            assert_eq!(call_kernel("test_pow", &[GpuScalar(a)], &[b]), c);
        }
    }

    #[test]
    fn test_sqr() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = a.square();

            assert_eq!(call_kernel("test_sqr", &[GpuScalar(a)], &[]), b);
        }
    }

    #[test]
    fn test_double() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = a.double();

            assert_eq!(call_kernel("test_double", &[GpuScalar(a)], &[]), b);
        }
    }

    #[test]
    fn test_unmont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b: Scalar = unsafe { std::mem::transmute(a.to_repr()) };
            assert_eq!(call_kernel("test_unmont", &[GpuScalar(a)], &[]), b);
        }
    }

    #[test]
    fn test_mont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a_repr = Scalar::random(&mut rng).to_repr();
            let a: Scalar = unsafe { std::mem::transmute(a_repr) };
            let b = Scalar::from_repr(a_repr).unwrap();
            assert_eq!(call_kernel("test_mont", &[GpuScalar(a)], &[]), b);
        }
    }
}
