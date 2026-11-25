//! What the Burn engine crates share: the device for the backend picked by
//! cargo feature, and reading a model's weights.

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::Bytes;

pub const BACKEND_NAME: &str = if cfg!(feature = "metal") {
    "burn-metal"
} else if cfg!(feature = "vulkan") {
    "burn-vulkan"
} else if cfg!(feature = "wgpu") {
    "burn-wgpu"
} else {
    "burn-flex"
};

/// Never `Device::default()`: it picks the highest-priority backend compiled
/// into Burn, and `BURN_DEVICE` overrides it.
pub fn selected_device() -> Device {
    #[cfg(feature = "metal")]
    return Device::metal(DeviceKind::DefaultDevice);
    #[cfg(all(feature = "vulkan", not(feature = "metal")))]
    return Device::vulkan(DeviceKind::DefaultDevice);
    #[cfg(all(feature = "wgpu", not(any(feature = "metal", feature = "vulkan"))))]
    return Device::wgpu(DeviceKind::DefaultDevice);
    #[cfg(not(any(feature = "metal", feature = "vulkan", feature = "wgpu")))]
    return Device::flex();
}

/// Shared, because Burn's reader clones the buffer and cloning an unshared one
/// copies the weights (jina-clip-v1 peaked at 1,193 MB instead of 917 MB).
pub fn read_weights(path: &str) -> Result<Bytes> {
    let weights = std::fs::read(path).with_context(|| format!("could not read weights {path}"))?;
    Ok(Bytes::from_elems(weights).shared())
}
