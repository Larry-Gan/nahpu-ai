//! nomic-embed-text v1.5 and nomic-embed-vision v1.5: text and images in one
//! 768-dimension space, so a text query finds an image.

use anyhow::Result;
use nahpu_embed_core::{ImageEmbedder, TextEmbedder};

/// Nomic publishes no logit scale, so this is the upper limit CLIP trains
/// with. Ranking does not use it. The probabilities it gives are not calibrated.
pub const NOMIC_LOGIT_SCALE: f32 = 100.0;

#[allow(dead_code)]
const DIMENSIONS: usize = 768;
/// The published recipe's limit; the tokenizer file sets none.
#[allow(dead_code)]
const MAX_TOKENS: usize = 512;

pub fn load_nomic_text() -> Result<Box<dyn TextEmbedder>> {
    #[cfg(has_nomic_text)]
    return Ok(Box::new(text::NomicText::load()?));
    #[cfg(not(has_nomic_text))]
    return Err(crate::missing_model("nomic-embed-text-v1_5"));
}

pub fn load_nomic_vision() -> Result<Box<dyn ImageEmbedder>> {
    #[cfg(has_nomic_vision)]
    return Ok(Box::new(vision::NomicVision::load()?));
    #[cfg(not(has_nomic_vision))]
    return Err(crate::missing_model("nomic-embed-vision-v1_5"));
}

#[cfg(has_nomic_text)]
mod text {
    use super::*;

    use burn::prelude::*;
    use nahpu_embed_core::{normalise_to_unit_length, Embedding, TextRole};
    use tokenizers::{Tokenizer, TruncationParams};

    use crate::BACKEND_NAME;

    #[allow(clippy::all, clippy::pedantic, dead_code)]
    mod model {
        include!(concat!(env!("OUT_DIR"), "/model/nomic-embed-text-v1_5.rs"));
    }

    pub struct NomicText {
        model: model::Model,
        tokenizer: Tokenizer,
        device: Device,
        name: String,
    }

    impl NomicText {
        pub fn load() -> Result<Self> {
            let device = crate::selected_device();
            let weights = crate::read_weights(env!("NOMIC_TEXT_WEIGHTS"))?;

            let mut tokenizer = crate::read_tokenizer("nomic-embed-text-v1_5")?;
            // Cuts the text, not the special tokens, so [SEP] stays last.
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: MAX_TOKENS,
                    ..Default::default()
                }))
                .map_err(|e| anyhow::anyhow!("could not set truncation: {e}"))?;

            Ok(Self {
                model: model::Model::from_bytes(weights, &device),
                tokenizer,
                device,
                name: format!("{BACKEND_NAME}/nomic-embed-text-v1.5"),
            })
        }

        fn int_tensor(&self, values: Vec<i64>) -> Tensor<2, Int> {
            let shape = [1, values.len()];
            Tensor::from_data(TensorData::new(values, shape), &self.device)
        }
    }

    impl TextEmbedder for NomicText {
        fn name(&self) -> &str {
            &self.name
        }

        fn dimensions(&self) -> usize {
            DIMENSIONS
        }

        fn embed(&self, text: &str, role: TextRole) -> Result<Embedding> {
            let prefix = match role {
                TextRole::Query => "search_query: ",
                TextRole::Document => "search_document: ",
            };
            let encoded = self
                .tokenizer
                .encode(format!("{prefix}{text}"), true)
                .map_err(|e| anyhow::anyhow!("could not tokenize: {e}"))?;
            let ids: Vec<i64> = encoded.get_ids().iter().map(|&id| i64::from(id)).collect();
            let tokens = ids.len();

            // Unlike MiniLM's, this graph takes token_type_ids second. One
            // text at a time has no padding, so the attention mask is all ones.
            let hidden_states = self.model.forward(
                self.int_tensor(ids),
                self.int_tensor(vec![0; tokens]),
                self.int_tensor(vec![1; tokens]),
            );

            // With no padding, mean pooling over the mask is the plain mean.
            let pooled = hidden_states.mean_dim(1).squeeze_dim::<2>(1);
            let pooled = crate::rows_of(pooled).remove(0);
            Ok(normalise_to_unit_length(layer_norm_without_weights(pooled)))
        }
    }

    /// Without it text and image vectors do not line up, though the cosine
    /// against the text reference barely moves.
    fn layer_norm_without_weights(vector: Vec<f32>) -> Vec<f32> {
        let count = vector.len() as f32;
        let mean = vector.iter().sum::<f32>() / count;
        let variance = vector.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / count;
        let scale = (variance + 1e-5).sqrt();
        vector.into_iter().map(|x| (x - mean) / scale).collect()
    }
}

#[cfg(has_nomic_vision)]
mod vision {
    use super::*;

    use burn::prelude::*;
    use image::imageops::{self, FilterType};
    use nahpu_embed_core::preprocess::{decode_upright, normalise_to_chw};
    use nahpu_embed_core::{normalise_to_unit_length, Embedding, PreprocessConfig};

    use crate::BACKEND_NAME;

    #[allow(clippy::all, clippy::pedantic, dead_code)]
    mod model {
        include!(concat!(
            env!("OUT_DIR"),
            "/model/nomic-embed-vision-v1_5.rs"
        ));
    }

    pub struct NomicVision {
        model: model::Model,
        device: Device,
        name: String,
    }

    impl NomicVision {
        pub fn load() -> Result<Self> {
            let device = crate::selected_device();
            let weights = crate::read_weights(env!("NOMIC_VISION_WEIGHTS"))?;

            Ok(Self {
                model: model::Model::from_bytes(weights, &device),
                device,
                name: format!("{BACKEND_NAME}/nomic-embed-vision-v1.5"),
            })
        }
    }

    impl ImageEmbedder for NomicVision {
        fn name(&self) -> &str {
            &self.name
        }

        fn dimensions(&self) -> usize {
            DIMENSIONS
        }

        fn preprocess_config(&self) -> PreprocessConfig {
            PreprocessConfig::OPENAI_CLIP_224
        }

        fn embed_pixels(&self, pixels: &[f32]) -> Result<Embedding> {
            let size = self.preprocess_config().size as usize;
            anyhow::ensure!(
                pixels.len() == 3 * size * size,
                "expected a 3x{size}x{size} tensor, got {} values",
                pixels.len()
            );
            let input = Tensor::<4>::from_data(
                TensorData::new(pixels.to_vec(), [1, 3, size, size]),
                &self.device,
            );

            // The embedding is token 0 of the first output, last_hidden_state
            // [1, 197, 768]; the second output is not the embedding.
            let (last_hidden_state, _) = self.model.forward(input);
            let first_token = last_hidden_state.slice([0..1, 0..1]).squeeze_dim::<2>(1);
            Ok(normalise_to_unit_length(
                crate::rows_of(first_token).remove(0),
            ))
        }

        /// The published preprocessor squashes the photo to a square, ignoring
        /// its aspect ratio, where the default crops the centre.
        fn embed_image(&self, encoded_image: &[u8]) -> Result<Embedding> {
            let config = self.preprocess_config();
            let upright = decode_upright(encoded_image)?;
            let squashed =
                imageops::resize(&upright, config.size, config.size, FilterType::CatmullRom);
            self.embed_pixels(&normalise_to_chw(&squashed, &config))
        }
    }
}
