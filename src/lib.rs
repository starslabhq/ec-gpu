mod nvidia;
mod utils;

use ff::PrimeField;
use itertools::join;
use num_bigint::BigUint;

static COMMON_SRC: &str = include_str!("cl/common.cl");
static FIELD_SRC: &str = include_str!("cl/field.cl");

#[derive(Clone, Copy)]
pub struct Limb(u32);
impl Limb {
    fn zero() -> Self {
        Self(0)
    }

    fn new(val: u32) -> Self {
        Self(val)
    }

    fn value(&self) -> u32 {
        self.0
    }

    fn limbs_of<T>(value: T) -> Vec<Self> {
        utils::limbs_of::<T, u32>(value)
            .into_iter()
            .map(Self::new)
            .collect()
    }

    /// Calculate the `INV` parameter of Montgomery reduction algorithm for 32/64bit limbs
    /// * `a` - Is the first limb of modulus
    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u32;
        for _ in 0..31 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }

    fn calculate_r2<F: PrimeField>() -> Vec<Self> {
        calculate_r2::<F>().into_iter().map(Self::new).collect()
    }
}

fn define_field(name: &str, limbs: Vec<Limb>) -> String {
    format!(
        "#define {} ((FIELD){{ {{ {} }} }})",
        name,
        join(limbs.iter().map(|l| l.value()), ", ")
    )
}

/// Calculates `R ^ 2 mod P` and returns the result as a vector of 32bit limbs
fn calculate_r2<F: PrimeField>() -> Vec<u32> {
    // R ^ 2 mod P
    BigUint::new(utils::limbs_of::<_, u32>(F::one()))
        .modpow(
            &BigUint::from_slice(&[2]),                          // ^ 2
            &BigUint::new(utils::limbs_of::<_, u32>(F::char())), // mod P
        )
        .to_u32_digits()
}

/// Generates OpenCL constants and type definitions of prime-field `F`
fn params<F>() -> String
where
    F: PrimeField,
{
    let one = Limb::limbs_of(F::one()); // Get Montgomery form of F::one()
    let p = Limb::limbs_of(F::char()); // Get regular form of field modulus
    let r2 = Limb::calculate_r2::<F>();
    let limbs = one.len(); // Number of limbs
    let inv = Limb::calc_inv(p[0]);
    let limbs_def = format!("#define FIELD_LIMBS {}", limbs);
    let p_def = define_field("FIELD_P", p);
    let r2_def = define_field("FIELD_R2", r2);
    let one_def = define_field("FIELD_ONE", one);
    let zero_def = define_field("FIELD_ZERO", vec![Limb::zero(); limbs]);
    let inv_def = format!("#define FIELD_INV {}", inv.value());
    let typedef = "typedef struct { limb val[FIELD_LIMBS]; } FIELD;".to_string();
    join(
        &[
            limbs_def, one_def, p_def, r2_def, zero_def, inv_def, typedef,
        ],
        "\n",
    )
}

/// Returns OpenCL source-code of a ff::PrimeField with name `name`
/// Find details in README.md
pub fn field<F>(name: &str) -> String
where
    F: PrimeField,
{
    join(
        &[
            COMMON_SRC.to_string(),
            params::<F>(),
            nvidia::field_add_sub_nvidia::<F>(),
            String::from(FIELD_SRC),
        ],
        "\n",
    )
    .replace("FIELD", name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use lazy_static::lazy_static;
    use ocl::{OclPrm, ProQue};
    use paired::bls12_381::{Fr, FrRepr};
    use rand::{thread_rng, Rng};

    #[derive(PartialEq, Debug, Clone, Copy)]
    #[repr(transparent)]
    pub struct GpuFr(pub Fr);
    impl Default for GpuFr {
        fn default() -> Self {
            Self(Fr::zero())
        }
    }
    unsafe impl OclPrm for GpuFr {}

    lazy_static! {
        static ref PROQUE: ProQue = {
            static TEST_SRC: &str = include_str!("cl/test.cl");
            let src = format!("{}\n{}", field::<Fr>("Fr"), TEST_SRC);
            ProQue::builder().src(src).dims(1).build().unwrap()
        };
    }

    macro_rules! call_kernel {
        ($name:expr, $($arg:expr),*) => {{
            let mut cpu_buffer = vec![GpuFr::default()];
            let buffer = PROQUE.create_buffer::<GpuFr>().unwrap();
            buffer.write(&cpu_buffer).enq().unwrap();
            let kernel =
                PROQUE
                .kernel_builder($name)
                $(.arg($arg))*
                .arg(&buffer)
                .build().unwrap();
            unsafe {
                kernel.enq().unwrap();
            }
            // Make sure the queue is fully processed
            PROQUE.finish().unwrap();
            buffer.read(&mut cpu_buffer).enq().unwrap();

            cpu_buffer[0].0
        }};
    }

    #[test]
    fn test_add() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = Fr::random(&mut rng);
            let mut c = a.clone();
            c.add_assign(&b);
            assert_eq!(call_kernel!("test_add", GpuFr(a), GpuFr(b)), c);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = Fr::random(&mut rng);
            let mut c = a.clone();
            c.sub_assign(&b);
            assert_eq!(call_kernel!("test_sub", GpuFr(a), GpuFr(b)), c);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = Fr::random(&mut rng);
            let mut c = a.clone();
            c.mul_assign(&b);
            assert_eq!(call_kernel!("test_mul", GpuFr(a), GpuFr(b)), c);
        }
    }

    #[test]
    fn test_pow() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = rng.gen::<u32>();
            let c = a.pow([b as u64]);
            assert_eq!(call_kernel!("test_pow", GpuFr(a), b), c);
        }
    }

    #[test]
    fn test_sqr() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let mut b = a.clone();
            b.square();
            assert_eq!(call_kernel!("test_sqr", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_double() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let mut b = a.clone();
            b.double();
            assert_eq!(call_kernel!("test_double", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_unmont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = unsafe { std::mem::transmute::<FrRepr, Fr>(a.into_repr()) };
            assert_eq!(call_kernel!("test_unmont", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_mont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a_repr = Fr::random(&mut rng).into_repr();
            let a = unsafe { std::mem::transmute::<FrRepr, Fr>(a_repr) };
            let b = Fr::from_repr(a_repr).unwrap();
            assert_eq!(call_kernel!("test_mont", GpuFr(a)), b);
        }
    }
}
