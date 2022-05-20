use std::path::PathBuf;
use std::{env, fs};

#[path = "src/source.rs"]
mod source;

#[cfg(feature = "bls12")]
use source::{Config, Field, Limb};
use source::{Limb32, Limb64};

#[cfg(feature = "bls12")]
fn bls12_config<L: Limb>() -> Config<L> {
    use blstrs::{Fp, Scalar};

    Config::new()
        .add_fft(Field::<Scalar>::new("Fr"))
        // G1
        .add_multiexp(Field::<Fp>::new("Fq"), Field::<Scalar>::new("Fr"))
        // G2
        .add_multiexp(
            Field::<Fp>::quadratic_extension("Fq", "Fq2"),
            Field::<Scalar>::new("Fr"),
        )
}

/// The build script is use to generate the CUDA kernel and OpenCL source at compile-time, if the
/// `bls12` feature is enabled.
#[cfg(all(feature = "bls12", not(feature = "cargo-clippy")))]
fn main() {
    #[cfg(feature = "cuda")]
    generate_cuda();
    #[cfg(feature = "opencl")]
    generate_opencl();
}

#[cfg(feature = "cuda")]
fn generate_cuda() {
    use sha2::{Digest, Sha256};

    // This is a hack for the case when the documentation is built on docs.rs. For the
    // documentation  we don't need a properly compiled kernel, but just some arbitrary bytes.
    if env::var("DOCS_RS").is_ok() {
        println!("cargo:rustc-env=CUDA_KERNEL_FATBIN=../build.rs");
        return;
    }

    let kernel_source = bls12_config::<Limb32>().gen_source();
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR was not set.");

    // Make it possible to override the default options. Though the source and output file is
    // always set automatically.
    let mut nvcc = match env::var("EC_GPU_CUDA_NVCC_ARGS") {
        Ok(args) => execute::command(format!("nvcc {}", args)),
        Err(_) => {
            let mut command = std::process::Command::new("nvcc");
            command
                .arg("--optimize=6")
                // Compile with as many threads as CPUs are available.
                .arg("--threads=0")
                .arg("--fatbin")
                .arg("--gpu-architecture=sm_86")
                .arg("--generate-code=arch=compute_86,code=sm_86");
            //.arg("--generate-code=arch=compute_80,code=sm_80")
            //.arg("--generate-code=arch=compute_75,code=sm_75");
            command
        }
    };

    // Hash the source and the compile flags. Use that as the filename, so that the kernel is only
    // rebuilt if any of them change.
    let mut hasher = Sha256::new();
    hasher.update(kernel_source.as_bytes());
    hasher.update(&format!("{:?}", &nvcc));
    let kernel_digest = hex::encode(hasher.finalize());

    let source_path: PathBuf = [&out_dir, &format!("{}.cu", &kernel_digest)]
        .iter()
        .collect();
    let fatbin_path: PathBuf = [&out_dir, &format!("{}.fatbin", &kernel_digest)]
        .iter()
        .collect();

    fs::write(&source_path, &kernel_source).unwrap_or_else(|_| {
        panic!(
            "Cannot write kernel source at {}.",
            source_path.to_str().unwrap()
        )
    });

    // Only compile if the output doesn't exist yet.
    if !fatbin_path.as_path().exists() {
        let status = nvcc
            .arg("--output-file")
            .arg(&fatbin_path)
            .arg(&source_path)
            .status()
            .expect("Cannot run nvcc. Install the NVIDIA toolkit or disable the `cuda` feature.");

        if !status.success() {
            panic!(
                "nvcc failed. See the kernel source at {}",
                source_path.to_str().unwrap()
            );
        }
    }

    // The idea to put the path to the farbin into a compile-time env variable is from
    // https://github.com/LutzCle/fast-interconnects-demo/blob/b80ea8e04825167f486ab8ac1b5d67cf7dd51d2c/rust-demo/build.rs
    println!(
        "cargo:rustc-env=CUDA_KERNEL_FATBIN={}",
        fatbin_path.to_str().unwrap()
    );
}

#[cfg(feature = "opencl")]
fn generate_opencl() {
    let kernel_source = bls12_config::<Limb64>().gen_source();
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR was not set.");

    // Generating the kernel source is cheap, hence use a fixed name and override it on every
    // build.
    let source_path: PathBuf = [&out_dir, "kernel.cl"].iter().collect();

    fs::write(&source_path, &kernel_source).unwrap_or_else(|_| {
        panic!(
            "Cannot write kernel source at {}.",
            source_path.to_str().unwrap()
        )
    });

    // For OpenCL we only need the kernel source, it is compiled at runtime.
    #[cfg(feature = "opencl")]
    println!(
        "cargo:rustc-env=OPENCL_KERNEL_SOURCE={}",
        source_path.to_str().unwrap()
    );
}

#[cfg(not(all(feature = "bls12", not(feature = "cargo-clippy"))))]
fn main() {
    // This is a hack for the case when we run Clippy while we don't generate any GPU kernel.
    // Clippy we don't need a proper source or properly compiled kernel, but just some arbitrary
    // bytes.
    println!("cargo:rustc-env=CUDA_KERNEL_FATBIN=../build.rs");
    println!("cargo:rustc-env=OPENCL_KERNEL_SOURCE=../build.rs");
}
