#[path = "../../build-support/models.rs"]
mod models;

use models::OnnxModel;

const DET: OnnxModel = OnnxModel {
    stem: "PP-OCRv6_small_det",
    cfg: "has_pp_ocrv6_small_det",
    weights_env: "PP_OCRV6_SMALL_DET_WEIGHTS",
};
const REC: OnnxModel = OnnxModel {
    stem: "PP-OCRv6_small_rec",
    cfg: "has_pp_ocrv6_small_rec",
    weights_env: "PP_OCRV6_SMALL_REC_WEIGHTS",
};

fn main() {
    let det = models::import_if_present(&DET);
    let rec = models::import_if_present(&REC);

    // The engine needs both models.
    println!("cargo::rustc-check-cfg=cfg(has_models)");
    if det && rec {
        println!("cargo::rustc-cfg=has_models");
    }
}
