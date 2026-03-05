//! Common download utilities.

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::path::Path;

/// Verify SHA-256 checksum of a file.
pub fn verify_checksum(path: &Path, expected_hex: &str) -> Result<bool> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let result = format!("{:x}", hasher.finalize());
    Ok(result == expected_hex)
}

/// Compute SHA-256 checksum of a file.
pub fn compute_checksum(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}
