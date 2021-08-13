use ff::PrimeField;
use itertools::join;

use crate::Limb;

/// Generates PTX-Assembly implementation of FIELD_add_/FIELD_sub_
pub fn field_add_sub_nvidia<F>() -> String
where
    F: PrimeField,
{
    let mut result = String::new();

    result.push_str("#ifdef NVIDIA\n");
    for op in &["sub", "add"] {
        let len = Limb::limbs_of(F::one()).len();

        let mut src = format!("FIELD FIELD_{}_nvidia(FIELD a, FIELD b) {{\n", op);
        if len > 1 {
            src.push_str("asm(");
            src.push_str(format!("\"{}.cc.u32 %0, %0, %{};\\r\\n\"\n", op, len).as_str());
            for i in 1..len - 1 {
                src.push_str(
                    format!("\"{}c.cc.u32 %{}, %{}, %{};\\r\\n\"\n", op, i, i, len + i).as_str(),
                );
            }
            src.push_str(
                format!(
                    "\"{}c.u32 %{}, %{}, %{};\\r\\n\"\n",
                    op,
                    len - 1,
                    len - 1,
                    2 * len - 1
                )
                .as_str(),
            );
            src.push(':');
            let inps = join((0..len).map(|n| format!("\"+r\"(a.val[{}])", n)), ", ");
            src.push_str(inps.as_str());

            src.push_str("\n:");
            let outs = join((0..len).map(|n| format!("\"r\"(b.val[{}])", n)), ", ");
            src.push_str(outs.as_str());
            src.push_str(");\n");
        }
        src.push_str("return a;\n}\n");

        result.push_str(&src);
    }
    result.push_str("#endif\n");

    result
}
