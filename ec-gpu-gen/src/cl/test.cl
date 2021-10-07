#define FIELD_LIMBS 8;
#define FIELD_LIMB_BITS 32;

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of FIELD_add(a, a)
DEVICE Scalar32 vmx_double(Scalar32 a) {

  for(uchar i = 7; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> 31);
  a.val[0] <<= 1;
  //if(FIELD_gte(a, FIELD_P)) a = FIELD_sub_(a, FIELD_P);
  /*
  if(FIELD_gte(a, FIELD_P)) {
    a.val[0] = 123;
    a.val[1] = 0;
    a.val[2] = 0;
    a.val[3] = 0;
    a.val[4] = 0;
    a.val[5] = 0;
    a.val[6] = 0;
    a.val[7] = 0;
  }
  else {
    a.val[0] = 555;
    a.val[1] = 0;
    a.val[2] = 0;
    a.val[3] = 0;
    a.val[4] = 0;
    a.val[5] = 0;
    a.val[6] = 0;
    a.val[7] = 0;
  };
  */
  return a;
}

KERNEL void test_double_32(Scalar32 a, GLOBAL Scalar32 *result) {
  *result = vmx_double(a);
}

