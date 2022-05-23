/// Describes how to generate the elliptic curve operations for
/// - `Scalar`
/// - `Fp`
/// - `Fp2`
/// - `G1`
/// - `G2`
//pub trait GpuEngine {
//    type Scalar: GpuField;
//    type Fp: GpuField;
//}

///// Describes how to generate elliptic curve operations for a group.
//pub trait GpuGroup {
//    /// The base field.
//    type Fp: GpuField;
//    /// The scalar field.
//    type Scalar: GpuField;
//}

/// Describes how to generate the gpu sources for a Field.
pub trait GpuField {
    /// The name of the field.
    ///
    /// This name is used in the source code that is generated for the GPU. It should be globally
    /// unique so that you don't collide with other libraries. It must *not* contain spaces. Best
    /// is to use a combination of the curve you are using and the field name. For example:
    /// `bls12_381_fp`, `bls12_381_scalar`, or `pallas_fp`.
    fn name() -> String;

    /// Returns `1` as a vector of 32bit limbs.
    fn one() -> Vec<u32>;

    /// Returns `R ^ 2 mod P` as a vector of 32bit limbs.
    fn r2() -> Vec<u32>;

    /// Returns the field modulus in non-Montgomery form (least significant limb first).
    fn modulus() -> Vec<u32>;
}

/// This is just a hack, nothing to see here.
///
/// This is used so that the second generic parameter of a field which is not an extension field
/// can be ommitted.
impl GpuField for () {
   fn name() -> String {
       "DO NOT USE THIS, IT IS JUST THERE TO MAKE THE GPUFIELD USAGE NICER.".to_string()
   }

   fn one() -> Vec<u32> {
       Vec::new()
   }

   fn r2() -> Vec<u32> {
       Vec::new()
   }

   fn modulus() -> Vec<u32> {
       Vec::new()
   }
}
