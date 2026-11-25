#[path = "../../build-support/models.rs"]
mod models;

use models::OnnxModel;

const MODELS: [OnnxModel; 7] = [
    OnnxModel {
        stem: "all-MiniLM-L6-v2",
        cfg: "has_minilm",
        weights_env: "MINILM_WEIGHTS",
    },
    OnnxModel {
        stem: "jina-clip-v1-text",
        cfg: "has_jina_text",
        weights_env: "JINA_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "jina-clip-v1-vision",
        cfg: "has_jina_vision",
        weights_env: "JINA_VISION_WEIGHTS",
    },
    OnnxModel {
        stem: "bioclip-text",
        cfg: "has_bioclip_text",
        weights_env: "BIOCLIP_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "bioclip-vision",
        cfg: "has_bioclip_vision",
        weights_env: "BIOCLIP_VISION_WEIGHTS",
    },
    OnnxModel {
        stem: "nomic-embed-text-v1_5",
        cfg: "has_nomic_text",
        weights_env: "NOMIC_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "nomic-embed-vision-v1_5",
        cfg: "has_nomic_vision",
        weights_env: "NOMIC_VISION_WEIGHTS",
    },
];

fn main() {
    for model in &MODELS {
        models::import_if_present(model);
    }
}
