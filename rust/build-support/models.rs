//! Imports optional ONNX models from a `build.rs`. Include it with
//! `#[path = "../../build-support/models.rs"] mod models;`.

use std::path::{Path, PathBuf};

use burn_onnx::{LoadStrategy, ModelGen};

pub struct OnnxModel {
    pub stem: &'static str,
    pub cfg: &'static str,
    pub weights_env: &'static str,
}

/// Generates `$OUT_DIR/model/<stem>.rs` and `.bpk` from `<models>/<stem>.onnx`
/// if it exists, then sets the model's cfg and points its env var at the
/// `.bpk`. Always exports `NAHPU_MODELS_DIR`, where the sidecar files are.
pub fn import_if_present(model: &OnnxModel) -> bool {
    println!("cargo::rustc-check-cfg=cfg({})", model.cfg);

    let models = models_dir();
    println!("cargo::rustc-env=NAHPU_MODELS_DIR={}", models.display());
    rebuild_when_models_change(&models);

    let onnx = models.join(format!("{}.onnx", model.stem));
    if !onnx.is_file() {
        warn_model_missing(&onnx);
        return false;
    }

    // Importing takes minutes, so it is skipped while the stamp written by the
    // last import still describes this `.onnx` and this build script.
    let weights = generated_weights(model.stem);
    let stamp_file = weights.with_extension("stamp");
    let stamp = import_stamp(&onnx);
    let imported = weights.is_file()
        && weights.with_extension("rs").is_file()
        && std::fs::read_to_string(&stamp_file).is_ok_and(|last| last == stamp);
    if !imported {
        generate_without_baked_paths(&onnx);
        std::fs::write(&stamp_file, stamp).expect("could not write the import stamp");
    }

    println!(
        "cargo::rustc-env={}={}",
        model.weights_env,
        weights.display()
    );
    println!("cargo::rustc-cfg={}", model.cfg);
    true
}

fn generate_without_baked_paths(onnx: &Path) {
    ModelGen::new()
        .input(onnx.to_str().expect("model path is not valid UTF-8"))
        .out_dir("model/")
        .load_strategy(LoadStrategy::Bytes)
        .run_from_script();
}

fn generated_weights(stem: &str) -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"))
        .join("model")
        .join(format!("{stem}.bpk"))
}

fn models_dir() -> PathBuf {
    if let Ok(overridden) = std::env::var("NAHPU_MODEL_DIR") {
        return PathBuf::from(overridden);
    }
    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models");
    fallback.canonicalize().unwrap_or(fallback)
}

/// Path, size and modification time of the `.onnx` and of this build script.
/// Any change to either, including to an older file or another models
/// directory, changes the stamp.
fn import_stamp(onnx: &Path) -> String {
    let describe = |path: &Path| {
        let file = path.metadata().expect("could not stat an import input");
        format!("{} {} {:?}\n", path.display(), file.len(), file.modified().ok())
    };
    let script = std::env::current_exe().expect("build script has a path");
    describe(onnx) + &describe(&script)
}

/// Never watches a missing `.onnx`: cargo reruns a build script on every build
/// while a watched path is missing. Nor the directory's parent: by default
/// that is the repo root, which holds `rust/target`. Watching the directory
/// notices a model arriving, changing or leaving.
fn rebuild_when_models_change(models: &Path) {
    println!("cargo::rerun-if-env-changed=NAHPU_MODEL_DIR");
    println!("cargo::rerun-if-changed={}", models.display());
}

fn warn_model_missing(onnx: &Path) {
    println!(
        "cargo::warning=no ONNX at {} — building without it. \
         Run scripts/fetch-model.sh to enable it.",
        onnx.display()
    );
}
