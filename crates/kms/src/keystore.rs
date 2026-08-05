//! Keystore encryption format and ethers.js keystore migration.
//!
//! This module provides:
//! - A krusty-kms native keystore format (version 1, XChaCha20-Poly1305 + scrypt)
//! - Decryption of ethers.js / Web3 Secret Storage keystores (version 3, AES-128-CTR + scrypt)

use aes::Aes128;
use ctr::cipher::{KeyIvInit, StreamCipher};
use krusty_kms_common::{KmsError, Result};
use scrypt::{scrypt, Params as ScryptParams};
use sha3::{Digest, Keccak256};
use zeroize::Zeroize;

use crate::encryption::{decrypt_with_key, encrypt_with_key, scrypt_log_n};

type Aes128Ctr = ctr::Ctr128BE<Aes128>;

/// Derived-key length required by the Web3 Secret Storage v3 format.
const V3_DKLEN: usize = 32;

/// AES-128-CTR IV length required by the Web3 Secret Storage v3 format.
const V3_IV_LEN: usize = 16;

// ---------------------------------------------------------------------------
// Native keystore (version 1)
// ---------------------------------------------------------------------------

/// Encrypt a mnemonic into a JSON keystore string.
///
/// The resulting JSON has the form:
/// ```json
/// {
///   "version": 1,
///   "crypto": {
///     "cipher": "xchacha20-poly1305",
///     "kdf": "scrypt",
///     "kdfparams": { "n": 32768, "r": 8, "p": 1, "dklen": 32, "salt": "hex..." },
///     "nonce": "hex...",
///     "ciphertext": "hex..."
///   }
/// }
/// ```
///
/// # Arguments
/// * `mnemonic` - The mnemonic phrase to encrypt
/// * `password` - User-supplied password
/// * `scrypt_n` - Scrypt cost parameter N (must be a power of 2)
pub fn encrypt_keystore(mnemonic: &str, password: &str, scrypt_n: u32) -> Result<String> {
    // Generate 16-byte salt
    let salt = krusty_kms_crypto::random_bytes::<16>();

    // Derive encryption key
    let mut key = derive_scrypt_key(password.as_bytes(), &salt, scrypt_n)?;

    // Encrypt mnemonic bytes
    let payload = encrypt_with_key(mnemonic.as_bytes(), &key)?;

    // Zeroize derived key
    key.zeroize();

    // Build JSON
    let keystore = serde_json::json!({
        "version": 1,
        "crypto": {
            "cipher": "xchacha20-poly1305",
            "kdf": "scrypt",
            "kdfparams": {
                "n": scrypt_n,
                "r": 8,
                "p": 1,
                "dklen": 32,
                "salt": hex::encode(salt),
            },
            "nonce": hex::encode(&payload.nonce),
            "ciphertext": hex::encode(&payload.ciphertext),
        }
    });

    serde_json::to_string(&keystore)
        .map_err(|e| KmsError::SerializationError(format!("Failed to serialize keystore: {e}")))
}

/// Decrypt a native krusty-kms keystore (version 1) to recover the mnemonic.
///
/// # Arguments
/// * `keystore_json` - JSON keystore string produced by [`encrypt_keystore`]
/// * `password` - The password used during encryption
pub fn decrypt_keystore(keystore_json: &str, password: &str) -> Result<String> {
    let v: serde_json::Value = serde_json::from_str(keystore_json)
        .map_err(|e| KmsError::DeserializationError(format!("Invalid keystore JSON: {e}")))?;

    let version = v["version"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing version field".to_string()))?;
    if version != 1 {
        return Err(KmsError::DeserializationError(format!(
            "Unsupported keystore version: {version}"
        )));
    }

    let crypto = &v["crypto"];

    let salt = hex::decode(
        crypto["kdfparams"]["salt"]
            .as_str()
            .ok_or_else(|| KmsError::DeserializationError("Missing salt".to_string()))?,
    )
    .map_err(|e| KmsError::DeserializationError(format!("Invalid salt hex: {e}")))?;

    let n = crypto["kdfparams"]["n"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdfparams.n".to_string()))?;

    // Before the cast, which would truncate 2^32 + 1024 to 1024.
    scrypt_log_n(n)?;
    let n = n as u32;

    let nonce = hex::decode(
        crypto["nonce"]
            .as_str()
            .ok_or_else(|| KmsError::DeserializationError("Missing nonce".to_string()))?,
    )
    .map_err(|e| KmsError::DeserializationError(format!("Invalid nonce hex: {e}")))?;

    let ciphertext = hex::decode(
        crypto["ciphertext"]
            .as_str()
            .ok_or_else(|| KmsError::DeserializationError("Missing ciphertext".to_string()))?,
    )
    .map_err(|e| KmsError::DeserializationError(format!("Invalid ciphertext hex: {e}")))?;

    // Validate before deriving
    crate::encryption::xnonce(&nonce)?;

    // Derive key
    let mut key = derive_scrypt_key(password.as_bytes(), &salt, n)?;

    let payload = crate::encryption::EncryptedPayload { nonce, ciphertext };
    let plaintext = decrypt_with_key(&payload, &key)?;

    // Zeroize derived key
    key.zeroize();

    String::from_utf8(plaintext)
        .map_err(|e| KmsError::CryptoError(format!("Decrypted keystore is not valid UTF-8: {e}")))
}

// ---------------------------------------------------------------------------
// ethers.js / Web3 Secret Storage (version 3) migration
// ---------------------------------------------------------------------------

/// Decrypt an ethers.js / Web3 Secret Storage keystore (version 3, scrypt KDF).
///
/// Supports the standard format:
/// ```json
/// {
///   "version": 3,
///   "crypto": {
///     "cipher": "aes-128-ctr",
///     "kdf": "scrypt",
///     "kdfparams": { "n": N, "r": 8, "p": 1, "dklen": 32, "salt": "hex" },
///     "cipherparams": { "iv": "hex" },
///     "ciphertext": "hex",
///     "mac": "hex"
///   }
/// }
/// ```
///
/// # Arguments
/// * `keystore_json` - JSON keystore string in ethers.js format
/// * `password` - The password used during encryption
///
/// # Returns
/// The decrypted content as a hex-encoded string (typically a private key).
pub fn decrypt_ethers_keystore(keystore_json: &str, password: &str) -> Result<String> {
    let v: serde_json::Value = serde_json::from_str(keystore_json)
        .map_err(|e| KmsError::DeserializationError(format!("Invalid keystore JSON: {e}")))?;

    let version = v["version"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing version field".to_string()))?;
    if version != 3 {
        return Err(KmsError::DeserializationError(format!(
            "Expected ethers keystore version 3, got {version}"
        )));
    }

    let crypto = &v["crypto"];

    let kdf = crypto["kdf"]
        .as_str()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdf field".to_string()))?;
    if kdf != "scrypt" {
        return Err(KmsError::DeserializationError(format!(
            "Unsupported KDF: {kdf} (only scrypt is supported)"
        )));
    }

    // Parse kdfparams
    let salt = hex::decode(
        crypto["kdfparams"]["salt"]
            .as_str()
            .ok_or_else(|| KmsError::DeserializationError("Missing salt".to_string()))?,
    )
    .map_err(|e| KmsError::DeserializationError(format!("Invalid salt hex: {e}")))?;

    let n = crypto["kdfparams"]["n"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdfparams.n".to_string()))?;

    let r = crypto["kdfparams"]["r"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdfparams.r".to_string()))?
        as u32;

    let p = crypto["kdfparams"]["p"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdfparams.p".to_string()))?
        as u32;

    let dklen = crypto["kdfparams"]["dklen"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdfparams.dklen".to_string()))?
        as usize;

    // Only `dklen < 32` is unsafe: it puts the MAC key slice out of bounds.
    if dklen < V3_DKLEN {
        return Err(KmsError::DeserializationError(format!(
            "Unsupported kdfparams.dklen: {dklen} (must be at least {V3_DKLEN})"
        )));
    }

    // Parse cipher params
    let iv =
        hex::decode(crypto["cipherparams"]["iv"].as_str().ok_or_else(|| {
            KmsError::DeserializationError("Missing cipherparams.iv".to_string())
        })?)
        .map_err(|e| KmsError::DeserializationError(format!("Invalid IV hex: {e}")))?;

    let mut ciphertext = hex::decode(
        crypto["ciphertext"]
            .as_str()
            .ok_or_else(|| KmsError::DeserializationError("Missing ciphertext".to_string()))?,
    )
    .map_err(|e| KmsError::DeserializationError(format!("Invalid ciphertext hex: {e}")))?;

    let expected_mac = hex::decode(
        crypto["mac"]
            .as_str()
            .ok_or_else(|| KmsError::DeserializationError("Missing mac".to_string()))?,
    )
    .map_err(|e| KmsError::DeserializationError(format!("Invalid mac hex: {e}")))?;

    if iv.len() != V3_IV_LEN {
        return Err(KmsError::DeserializationError(format!(
            "Invalid cipherparams.iv length: expected {V3_IV_LEN} bytes, got {}",
            iv.len()
        )));
    }

    // Derive key via scrypt
    let log_n = scrypt_log_n(n)?;
    let params = ScryptParams::new(log_n, r, p, dklen)
        .map_err(|e| KmsError::CryptoError(format!("Invalid scrypt params: {e}")))?;
    let mut derived_key = vec![0u8; dklen];
    scrypt(password.as_bytes(), &salt, &params, &mut derived_key)
        .map_err(|e| KmsError::CryptoError(format!("Scrypt KDF failed: {e}")))?;

    let aes_key = &derived_key[..V3_DKLEN / 2];
    let mac_key = &derived_key[V3_DKLEN / 2..V3_DKLEN];

    // Verify MAC: Keccak256(mac_key || ciphertext)
    let mut mac_input = Vec::with_capacity(mac_key.len() + ciphertext.len());
    mac_input.extend_from_slice(mac_key);
    mac_input.extend_from_slice(&ciphertext);
    let computed_mac = Keccak256::digest(&mac_input);

    if computed_mac.as_slice() != expected_mac.as_slice() {
        derived_key.zeroize();
        return Err(KmsError::CryptoError(
            "MAC verification failed: wrong password or corrupted keystore".to_string(),
        ));
    }

    // Decrypt with AES-128-CTR using the first 16 bytes of the derived key
    let mut cipher = Aes128Ctr::new(aes_key.into(), iv.as_slice().into());
    cipher.apply_keystream(&mut ciphertext);
    // ciphertext is now plaintext

    // Zeroize derived key
    derived_key.zeroize();

    Ok(hex::encode(ciphertext))
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/// Derive a 32-byte key from a password and salt using scrypt (r=8, p=1).
fn derive_scrypt_key(password: &[u8], kdf_salt: &[u8], n: u32) -> Result<[u8; 32]> {
    let log_n = scrypt_log_n(n as u64)?;
    let params = ScryptParams::new(log_n, 8, 1, 32)
        .map_err(|e| KmsError::CryptoError(format!("Invalid scrypt params: {e}")))?;
    let mut key = [u8::default(); 32];
    scrypt(password, kdf_salt, &params, &mut key)
        .map_err(|e| KmsError::CryptoError(format!("Scrypt KDF failed: {e}")))?;
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SCRYPT_N: u32 = 1024;

    fn test_password(offset: u8) -> String {
        (0..12)
            .map(|index| char::from(b'a' + index + offset))
            .collect()
    }

    #[test]
    fn encrypt_decrypt_keystore_roundtrip() {
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let password = test_password(0);

        let keystore_json = encrypt_keystore(mnemonic, &password, TEST_SCRYPT_N).unwrap();

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&keystore_json).unwrap();
        assert_eq!(parsed["version"], 1);
        assert_eq!(parsed["crypto"]["cipher"], "xchacha20-poly1305");
        assert_eq!(parsed["crypto"]["kdf"], "scrypt");
        assert_eq!(parsed["crypto"]["kdfparams"]["n"], TEST_SCRYPT_N);

        let decrypted = decrypt_keystore(&keystore_json, &password).unwrap();
        assert_eq!(decrypted, mnemonic);
    }

    #[test]
    fn decrypt_keystore_wrong_password_fails() {
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let password = test_password(0);

        let keystore_json = encrypt_keystore(mnemonic, &password, TEST_SCRYPT_N).unwrap();

        let wrong_password = test_password(1);
        let result = decrypt_keystore(&keystore_json, &wrong_password);
        assert!(result.is_err());
    }

    // malformed keystore metadata must not panic

    /// Build a v1 keystore with an arbitrary `nonce`, bypassing `encrypt_keystore`.
    fn keystore_v1_with_nonce_hex(nonce_hex: &str) -> String {
        serde_json::json!({
            "version": 1,
            "crypto": {
                "cipher": "xchacha20-poly1305",
                "kdf": "scrypt",
                "kdfparams": {
                    "n": TEST_SCRYPT_N, "r": 8, "p": 1, "dklen": 32,
                    "salt": hex::encode([0xabu8; 16]),
                },
                "nonce": nonce_hex,
                "ciphertext": hex::encode([0x11u8; 48]),
            }
        })
        .to_string()
    }

    /// Build a v3 keystore with an arbitrary `dklen` and IV length.
    fn ethers_keystore_with(dklen: u64, iv_len: usize) -> String {
        serde_json::json!({
            "version": 3,
            "crypto": {
                "cipher": "aes-128-ctr",
                "kdf": "scrypt",
                "kdfparams": {
                    "n": TEST_SCRYPT_N, "r": 8, "p": 1, "dklen": dklen,
                    "salt": hex::encode([0xabu8; 32]),
                },
                "cipherparams": { "iv": hex::encode(vec![0xcdu8; iv_len]) },
                "ciphertext": hex::encode([0x11u8; 32]),
                "mac": "00",
            }
        })
        .to_string()
    }

    #[test]
    fn decrypt_keystore_rejects_bad_nonce_length_without_panicking() {
        // 4 bytes instead of 24 used to hit an assert inside `generic-array`.
        for nonce_hex in ["", "deadbeef", &hex::encode([0u8; 25])] {
            assert!(
                decrypt_keystore(&keystore_v1_with_nonce_hex(nonce_hex), "password1234").is_err(),
                "nonce {nonce_hex:?} must be rejected"
            );
        }
    }

    #[test]
    fn decrypt_ethers_keystore_rejects_short_dklen_without_panicking() {
        // 10..=31 is the exact panic window: scrypt allows 10..=64, but anything
        // under 32 makes `derived_key[16..32]` go out of bounds.
        for dklen in [10u64, 15, 16, 17, 31] {
            let err = decrypt_ethers_keystore(&ethers_keystore_with(dklen, 16), "password1234")
                .expect_err("must be rejected");
            assert!(
                matches!(err, KmsError::DeserializationError(_)),
                "dklen={dklen}, got {err:?}"
            );
        }
    }

    #[test]
    fn decrypt_ethers_keystore_accepts_oversized_dklen() {
        // 33..=64 never panicked - the slices stay in bounds and the extra
        // bytes are ignored, which is what geth and ethers do.
        for dklen in [33u64, 48, 64] {
            let err = decrypt_ethers_keystore(&ethers_keystore_with(dklen, 16), "password1234")
                .expect_err("MAC must fail");
            assert!(
                format!("{err}").contains("MAC verification failed"),
                "dklen={dklen}, got {err}"
            );
        }
    }

    #[test]
    fn decrypt_ethers_keystore_rejects_bad_iv_length_without_panicking() {
        // `Aes128Ctr::new` converts the IV with `GenericArray::from_slice`,
        // which asserts on a length mismatch.
        for iv_len in [0usize, 8, 15, 17, 32] {
            assert!(
                decrypt_ethers_keystore(&ethers_keystore_with(32, iv_len), "password1234").is_err(),
                "iv_len={iv_len} must be rejected"
            );
        }
    }

    #[test]
    fn well_formed_dklen_and_iv_reach_mac_verification() {
        // Guards against over-tightening: dklen=32 with a 16-byte IV must get
        // past the new length checks and fail on the MAC instead.
        let err = decrypt_ethers_keystore(&ethers_keystore_with(32, 16), "password1234")
            .expect_err("MAC must fail");
        assert!(
            format!("{err}").contains("MAC verification failed"),
            "got {err}"
        );
    }

    #[test]
    fn decrypt_ethers_keystore_known_vector() {
        let decrypted =
            decrypt_ethers_keystore(&valid_ethers_keystore(32), &test_password(0)).unwrap();
        assert_eq!(decrypted, KNOWN_PRIVATE_KEY);
    }

    const KNOWN_PRIVATE_KEY: &str =
        "4c0883a69102937d6231471b5dbb6204fe512961708279f696ae35e0c2a1b5ce";

    /// Build a valid v3 keystore holding [`KNOWN_PRIVATE_KEY`], for a given `dklen`.
    fn valid_ethers_keystore(dklen: usize) -> String {
        let password = test_password(0);
        // Deterministic salt and IV for the test vector
        let salt = vec![0xab; 32];
        let iv = vec![0xcd; 16];

        // Derive key
        let log_n = (TEST_SCRYPT_N as f64).log2() as u8;
        let params = ScryptParams::new(log_n, 8, 1, dklen).unwrap();
        let mut derived_key = vec![0u8; dklen];
        scrypt(password.as_bytes(), &salt, &params, &mut derived_key).unwrap();

        // Encrypt with AES-128-CTR
        let aes_key = &derived_key[..16];
        let mut ciphertext = hex::decode(KNOWN_PRIVATE_KEY).unwrap();
        let mut cipher = Aes128Ctr::new(aes_key.into(), iv.as_slice().into());
        cipher.apply_keystream(&mut ciphertext);

        // Compute MAC
        let mut mac_input = Vec::new();
        mac_input.extend_from_slice(&derived_key[16..32]);
        mac_input.extend_from_slice(&ciphertext);
        let mac = Keccak256::digest(&mac_input);

        serde_json::json!({
            "version": 3,
            "crypto": {
                "cipher": "aes-128-ctr",
                "kdf": "scrypt",
                "kdfparams": {
                    "n": TEST_SCRYPT_N,
                    "r": 8,
                    "p": 1,
                    "dklen": dklen,
                    "salt": hex::encode(&salt),
                },
                "cipherparams": {
                    "iv": hex::encode(&iv),
                },
                "ciphertext": hex::encode(&ciphertext),
                "mac": hex::encode(mac.as_slice()),
            }
        })
        .to_string()
    }

    /// `n` values that must be refused rather than silently floored or truncated.
    ///
    /// All kept small: an `n` big enough to exhaust memory aborts the test
    /// process instead of failing a case, and that guard is not in place.
    const BAD_SCRYPT_N: [u64; 4] = [
        1000,             // floored to 512
        100_000,          // floored to 65536, a weaker KDF
        0,                // log2(0) is -inf
        (1 << 32) + 1024, // truncated to 1024 by `as u32`
    ];

    #[test]
    fn decrypt_ethers_keystore_rejects_non_power_of_two_n() {
        for n in BAD_SCRYPT_N {
            let mut ks: serde_json::Value =
                serde_json::from_str(&valid_ethers_keystore(32)).unwrap();
            ks["crypto"]["kdfparams"]["n"] = serde_json::json!(n);
            let err = decrypt_ethers_keystore(&ks.to_string(), &test_password(0))
                .expect_err("must be rejected");
            assert!(
                matches!(err, KmsError::DeserializationError(_)),
                "n={n}, got {err:?}"
            );
        }
    }

    #[test]
    fn decrypt_keystore_rejects_non_power_of_two_n() {
        // Same guard on the v1 path, which also takes `n` from untrusted JSON.
        for n in BAD_SCRYPT_N {
            let mut ks: serde_json::Value =
                serde_json::from_str(&keystore_v1_with_nonce_hex(&hex::encode([0u8; 24]))).unwrap();
            ks["crypto"]["kdfparams"]["n"] = serde_json::json!(n);
            assert!(
                decrypt_keystore(&ks.to_string(), "password1234").is_err(),
                "n={n} must be rejected"
            );
        }
    }

    #[test]
    fn encrypt_keystore_rejects_non_power_of_two_n() {
        // Matters most here: flooring weakens the KDF the caller asked for.
        for n in BAD_SCRYPT_N {
            let Ok(n) = u32::try_from(n) else { continue };
            assert!(
                encrypt_keystore("test mnemonic", "password1234", n).is_err(),
                "n={n} must be rejected"
            );
        }
    }

    #[test]
    fn decrypt_ethers_keystore_oversized_dklen_yields_the_same_key() {
        // `dklen > 32` is accepted for geth/ethers compatibility, so it must
        // decrypt *correctly*, not merely get past the length check: the tail
        // beyond 32 bytes is ignored and the recovered key is unchanged.
        for dklen in [33usize, 48, 64] {
            let got = decrypt_ethers_keystore(&valid_ethers_keystore(dklen), &test_password(0))
                .unwrap_or_else(|e| panic!("dklen={dklen} must decrypt, got {e}"));
            assert_eq!(got, KNOWN_PRIVATE_KEY, "dklen={dklen} changed the key");
        }
    }

    #[test]
    fn decrypt_ethers_keystore_wrong_password_fails() {
        // Minimal valid keystore with wrong password
        let salt = vec![0xab; 32];
        let iv = vec![0xcd; 16];
        let password = test_password(0);

        let log_n = (TEST_SCRYPT_N as f64).log2() as u8;
        let params = ScryptParams::new(log_n, 8, 1, 32).unwrap();
        let mut derived_key = vec![0u8; 32];
        scrypt(password.as_bytes(), &salt, &params, &mut derived_key).unwrap();

        let ciphertext = vec![0u8; 32];
        let mut mac_input = Vec::new();
        mac_input.extend_from_slice(&derived_key[16..32]);
        mac_input.extend_from_slice(&ciphertext);
        let mac = Keccak256::digest(&mac_input);

        let keystore = serde_json::json!({
            "version": 3,
            "crypto": {
                "cipher": "aes-128-ctr",
                "kdf": "scrypt",
                "kdfparams": {
                    "n": TEST_SCRYPT_N,
                    "r": 8,
                    "p": 1,
                    "dklen": 32,
                    "salt": hex::encode(&salt),
                },
                "cipherparams": { "iv": hex::encode(&iv) },
                "ciphertext": hex::encode(&ciphertext),
                "mac": hex::encode(mac.as_slice()),
            }
        });

        let keystore_json = serde_json::to_string(&keystore).unwrap();

        let wrong_password = test_password(1);
        let result = decrypt_ethers_keystore(&keystore_json, &wrong_password);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("MAC verification failed"));
    }
}
