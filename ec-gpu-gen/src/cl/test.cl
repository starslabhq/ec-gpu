//#define Scalar32_LIMBS 8;
//#define Scalar32_LIMB_BITS 32;

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of Scalar32_add(a, a)
DEVICE Scalar32 vmx_double(Scalar32 a) {
  for(uchar i = Scalar32_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (Scalar32_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(Scalar32_gte(a, Scalar32_P)) a = Scalar32_sub_(a, Scalar32_P);
  return a;
}

KERNEL void test_double_32(Scalar32 a, GLOBAL Scalar32 *result) {
  *result = vmx_double(a);
}
