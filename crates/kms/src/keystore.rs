//! Keystore encryption format and ethers.js keystore migration.
//!
//! This module provides:
//! - A krusty-kms native keystore format (version 1, XChaCha20-Poly1305 + scrypt)
//! - Decryption of ethers.js / Web3 Secret Storage keystores (version 3, AES-128-CTR + scrypt)

use aes::Aes128;
use ctr::cipher::{KeyIvInit, StreamCipher};
use krusty_kms_common::{KmsError, Result};
use sha3::{Digest, Keccak256};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use crate::encryption::{
    caller_param, decrypt_with_key, derive_scrypt_key, encrypt_with_key, scrypt_derive, xnonce,
};

type Aes128Ctr = ctr::Ctr128BE<Aes128>;

/// Derived-key length required by the Web3 Secret Storage v3 format.
const V3_DKLEN: usize = 32;

/// AES-128-CTR IV length required by the Web3 Secret Storage v3 format.
const V3_IV_LEN: usize = 16;

/// Largest `dklen` accepted. Matches scrypt's own limit, but keeps
/// `vec![0u8; dklen]` off a file-chosen size if that ever changes.
const V3_DKLEN_MAX: u64 = 64;

/// Largest keystore accepted, in bytes.
///
/// Checked before `serde_json::from_str`, which is the only place it can be: parsing
/// allocates the whole `Value` tree first, so a per-field bound on `ciphertext` would
/// fire long after the cost was paid. One guard at the entrance covers every field.
///
/// Real files are ~400 bytes (v1) and ~500 (v3, with geth's `address` and `id`), so
/// 64 KiB is two orders of magnitude of headroom for pretty-printing and extra
/// metadata while keeping the parse cost negligible. Without it the `# Cost` bounds
/// below hold only for the small files they name: a 200 MB `ciphertext` allocates the
/// input, the `Value`, and the decoded bytes, none of which the scrypt ceilings touch.
const MAX_KEYSTORE_BYTES: usize = 64 << 10;

/// Reject an oversized keystore before parsing it.
fn check_size(keystore_json: &str) -> Result<()> {
    if keystore_json.len() > MAX_KEYSTORE_BYTES {
        return Err(KmsError::DeserializationError(format!(
            "Keystore too large: {} bytes, limit is {MAX_KEYSTORE_BYTES}",
            keystore_json.len()
        )));
    }
    Ok(())
}

/// Render an untrusted JSON value for an error message, bounded.
///
/// Bounds the *work*, not just the result. Discriminating on the kind is the point:
/// `Display` on a `Value` serializes the whole subtree and on a string copies all of
/// it, so `"cipher": {"a": <10 MB>}` would allocate 10 MB to build a 40-char message
/// -- and these get logged. Numbers, booleans and null are bounded by their syntax.
fn brief(value: &serde_json::Value) -> String {
    const MAX: usize = 40;
    let text = match value {
        serde_json::Value::Object(_) => return "an object".to_string(),
        serde_json::Value::Array(_) => return "an array".to_string(),
        serde_json::Value::String(text) => text.as_str(),
        scalar => return scalar.to_string(),
    };
    // Slice before allocating, so a 10 MB string is never copied.
    match text.char_indices().nth(MAX) {
        Some((cut, _)) => format!("\"{}\"...", &text[..cut]),
        None => format!("\"{text}\""),
    }
}

/// Require a field to hold an exact string, with a bounded error message.
///
/// Both formats do this check and writing it per-site is what left the v3 `kdf` arm
/// interpolating an untrusted string straight into the message while its `cipher`
/// neighbour went through [`brief`]. Absence is a mismatch, not a default: the value
/// is what the decrypt path assumes unconditionally.
fn require_str(
    parent: &serde_json::Value,
    field: &str,
    expected: &str,
    format: &str,
) -> Result<()> {
    let got = &parent[field];
    if got.as_str() != Some(expected) {
        return Err(KmsError::DeserializationError(format!(
            "Unsupported {field}: {} ({format} requires {expected})",
            brief(got)
        )));
    }
    Ok(())
}

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
/// * `scrypt_n` - Scrypt cost parameter N: a power of two from 2 to 262144 (2^18,
///   geth's standard strength). Larger is refused, not written.
pub fn encrypt_keystore(mnemonic: &str, password: &str, scrypt_n: u32) -> Result<String> {
    // Generate 16-byte salt
    let salt = krusty_kms_crypto::random_bytes::<16>();

    // Derive encryption key
    let key =
        derive_scrypt_key(password.as_bytes(), &salt, scrypt_n.into()).map_err(caller_param)?;

    // Encrypt mnemonic bytes
    let payload = encrypt_with_key(mnemonic.as_bytes(), &key)?;

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
            "nonce": hex::encode(payload.nonce),
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
///
/// # Cost
/// The KDF parameters come from the file, so a caller that accepts keystores from
/// untrusted sources must rate-limit this. Rejection is cheap, but everything the
/// ceilings admit is not: worst case is ~2 s of CPU and ~264 MiB of allocation. The
/// input itself is capped at `MAX_KEYSTORE_BYTES`, so that is the whole cost -- the
/// ceilings live in `encryption::scrypt_params`.
pub fn decrypt_keystore(keystore_json: &str, password: &str) -> Result<String> {
    check_size(keystore_json)?;

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

    // Same reasoning as `kdfparams` below, and as the v3 path: this derives with
    // scrypt and decrypts with XChaCha20-Poly1305 unconditionally, so a file naming
    // anything else produces a wrong key reported as a wrong password. Required, not
    // absent-tolerant: `encrypt_keystore` has always written both, so a file missing
    // one is not a v1 keystore and guessing what it is has no upside.
    for (field, expected) in [("kdf", "scrypt"), ("cipher", "xchacha20-poly1305")] {
        require_str(crypto, field, expected, "version 1")?;
    }

    let kdfparams = &crypto["kdfparams"];

    let salt = hex::decode(
        kdfparams["salt"]
            .as_str()
            .ok_or_else(|| KmsError::DeserializationError("Missing salt".to_string()))?,
    )
    .map_err(|e| KmsError::DeserializationError(format!("Invalid salt hex: {e}")))?;

    let n = kdfparams["n"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdfparams.n".to_string()))?;

    // v1 only ever wrote these and always derives with them, so a file claiming
    // otherwise would decrypt to the wrong key and blame the password. Required rather
    // than defaulted: `as_u64().unwrap_or(expected)` would wave through `"8"`, `8.0`
    // and `-1` alongside genuine absence, which is the mismatch this is here to catch.
    for (field, expected) in [("r", 8u64), ("p", 1), ("dklen", 32)] {
        let got = &kdfparams[field];
        if got.as_u64() != Some(expected) {
            return Err(KmsError::DeserializationError(format!(
                "Unsupported kdfparams.{field}: {} (version 1 requires {expected})",
                brief(got)
            )));
        }
    }

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

    // Converted before deriving, so a malformed nonce costs nothing rather than ~2 s
    // of scrypt first. Past this point the length is carried by the type.
    let nonce = xnonce(&nonce)?;

    let key = derive_scrypt_key(password.as_bytes(), &salt, n)?;

    let payload = crate::encryption::EncryptedPayload { nonce, ciphertext };
    let plaintext = decrypt_with_key(&payload, &key)?;

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
///
/// # Cost
/// The KDF parameters come from the file, so a caller that accepts keystores from
/// untrusted sources must rate-limit this. Rejection is cheap, but everything the
/// ceilings admit is not: worst case is ~2 s of CPU and ~264 MiB of allocation. The
/// input itself is capped at `MAX_KEYSTORE_BYTES`, so that is the whole cost -- the
/// ceilings live in `encryption::scrypt_params`.
pub fn decrypt_ethers_keystore(keystore_json: &str, password: &str) -> Result<String> {
    check_size(keystore_json)?;

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

    require_str(crypto, "kdf", "scrypt", "version 3")?;

    // The MAC covers `mac_key || ciphertext`, not the cipher id, so an unchecked
    // `cipher` verifies and is then decrypted as AES-128-CTR regardless, returning
    // garbage as an imported private key.
    //
    // Required, not absent-tolerant like the v1 header: v1 decrypts with an AEAD, so
    // the wrong cipher fails authentication loudly, while CTR has nothing but this
    // MAC and would hand back a silently wrong key. Stripping the field from a real
    // aes-256-ctr keystore is enough to trigger that. The format mandates `cipher`
    // and geth and ethers both emit it, so nothing legitimate is stranded.
    require_str(crypto, "cipher", "aes-128-ctr", "version 3")?;

    // Parse kdfparams
    let kdfparams = &crypto["kdfparams"];

    let salt = hex::decode(
        kdfparams["salt"]
            .as_str()
            .ok_or_else(|| KmsError::DeserializationError("Missing salt".to_string()))?,
    )
    .map_err(|e| KmsError::DeserializationError(format!("Invalid salt hex: {e}")))?;

    let n = kdfparams["n"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdfparams.n".to_string()))?;

    let r = kdfparams["r"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdfparams.r".to_string()))?;

    let p = kdfparams["p"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdfparams.p".to_string()))?;

    let dklen = kdfparams["dklen"]
        .as_u64()
        .ok_or_else(|| KmsError::DeserializationError("Missing kdfparams.dklen".to_string()))?;

    // Below 32 puts the MAC key slice out of bounds; above 64 sizes an allocation
    // from the file. Checked before narrowing, which turns 2^32 + 32 into 32.
    if !(V3_DKLEN as u64..=V3_DKLEN_MAX).contains(&dklen) {
        return Err(KmsError::DeserializationError(format!(
            "Unsupported kdfparams.dklen: {dklen} (must be {V3_DKLEN}..={V3_DKLEN_MAX})"
        )));
    }
    let dklen = dklen as usize;

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

    // Derive key via scrypt.
    let mut derived_key = Zeroizing::new(vec![0u8; dklen]);
    scrypt_derive(password.as_bytes(), &salt, n, r, p, &mut derived_key)?;

    let aes_key = &derived_key[..V3_DKLEN / 2];
    let mac_key = &derived_key[V3_DKLEN / 2..V3_DKLEN];

    // Verify MAC: Keccak256(mac_key || ciphertext). Streamed rather than concatenated
    // to avoid a second copy of the MAC key in a plain `Vec` sized by the file. Not a
    // claim that no copy survives: Keccak256 absorbs `mac_key` into sponge state and
    // implements neither `Zeroize` nor `ZeroizeOnDrop`, so those 16 bytes still reach
    // freed memory. One un-scrubbed copy instead of two, and no file-sized allocation.
    let mut mac = Keccak256::new();
    mac.update(mac_key);
    mac.update(&ciphertext);
    let computed_mac = mac.finalize();

    // Constant-time: a `!=` on these leaks the expected MAC byte by byte to anyone
    // who can submit keystores and time the call. `ct_eq` is 0 on a length mismatch.
    if computed_mac.as_slice().ct_eq(&expected_mac).unwrap_u8() != 1 {
        return Err(KmsError::CryptoError(
            "MAC verification failed: wrong password or corrupted keystore".to_string(),
        ));
    }

    // Decrypt with AES-128-CTR using the first 16 bytes of the derived key.
    // `Zeroizing` because this buffer becomes the private key in place, and it is the
    // higher-value secret of the two -- scrubbing the KDF output while leaving the key
    // it protects in freed memory would be the wrong half. The returned `String` holds
    // the same bytes and cannot be scrubbed without changing the signature.
    let mut plaintext = Zeroizing::new(std::mem::take(&mut ciphertext));
    let mut stream = Aes128Ctr::new(aes_key.into(), iv.as_slice().into());
    stream.apply_keystream(&mut plaintext);

    Ok(hex::encode(&*plaintext))
}

#[cfg(test)]
mod tests {
    use super::*;
    // Fixtures build keystores by hand, bypassing the validated constructor.
    use crate::encryption::scrypt_params;
    use scrypt::{scrypt, Params as ScryptParams};

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
        // Asserting on the message, not just `is_err`: the fixture's ciphertext fails
        // the AEAD anyway, so `is_err` alone would still pass with `xnonce` deleted.
        for nonce_hex in ["", "deadbeef", &hex::encode([0u8; 25])] {
            let err = decrypt_keystore(&keystore_v1_with_nonce_hex(nonce_hex), &test_password(0))
                .expect_err("must be rejected");
            assert!(
                format!("{err}").contains("Invalid nonce length"),
                "nonce {nonce_hex:?}, got {err}"
            );
        }
    }

    #[test]
    fn oversized_keystore_is_rejected_unparsed() {
        // Both formats. The padding sits in an unused field, so everything else about
        // the file is valid and only the size can be what rejects it.
        for (json, decrypt) in [
            (
                encrypt_keystore("m", &test_password(0), TEST_SCRYPT_N).unwrap(),
                decrypt_keystore as fn(&str, &str) -> Result<String>,
            ),
            (valid_ethers_keystore(32), decrypt_ethers_keystore),
        ] {
            let mut ks: serde_json::Value = serde_json::from_str(&json).unwrap();
            ks["padding"] = serde_json::json!("x".repeat(MAX_KEYSTORE_BYTES));
            let err = decrypt(&ks.to_string(), &test_password(0)).expect_err("must be rejected");
            assert!(format!("{err}").contains("Keystore too large"), "got {err}");
        }
    }

    #[test]
    fn a_real_keystore_is_far_below_the_size_limit() {
        // Guards against over-tightening: if a real file ever approaches the cap, this
        // fails before the cap starts rejecting keystores in the field.
        for json in [
            encrypt_keystore("m", &test_password(0), TEST_SCRYPT_N).unwrap(),
            valid_ethers_keystore(64),
        ] {
            assert!(
                json.len() * 8 < MAX_KEYSTORE_BYTES,
                "{} bytes leaves under 8x headroom",
                json.len()
            );
        }
    }

    #[test]
    fn caller_supplied_scrypt_n_is_an_invalid_parameter_not_a_bad_file() {
        // The write path has no untrusted input to blame, so it must not report the
        // caller's own bad argument as a deserialization failure.
        let err = encrypt_keystore("m", &test_password(0), 1000).expect_err("must be rejected");
        assert!(matches!(err, KmsError::InvalidParameter(_)), "got {err:?}");

        // Read stays a deserialization error: there, `n` really did come from a file.
        let keystore = encrypt_keystore("m", &test_password(0), TEST_SCRYPT_N).unwrap();
        let mut ks: serde_json::Value = serde_json::from_str(&keystore).unwrap();
        ks["crypto"]["kdfparams"]["n"] = serde_json::json!(1000);
        let err =
            decrypt_keystore(&ks.to_string(), &test_password(0)).expect_err("must be rejected");
        assert!(
            matches!(err, KmsError::DeserializationError(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn brief_bounds_the_error_message_for_any_value_kind() {
        // A huge value must not become a huge error string. Every string-compared
        // field, not just `cipher`: `kdf` used to interpolate the raw value and was
        // the one arm this test did not reach.
        let huge = "a".repeat(MAX_KEYSTORE_BYTES / 2);
        for field in ["cipher", "kdf"] {
            for value in [
                serde_json::json!({ "nested": huge }),
                serde_json::json!([huge]),
                serde_json::json!(huge),
            ] {
                let mut ks: serde_json::Value =
                    serde_json::from_str(&valid_ethers_keystore(32)).unwrap();
                ks["crypto"][field] = value;
                let json = ks.to_string();
                assert!(json.len() <= MAX_KEYSTORE_BYTES, "fixture hit the size cap");

                let err = decrypt_ethers_keystore(&json, &test_password(0))
                    .expect_err("must be rejected");
                let message = format!("{err}");
                assert!(
                    message.len() < 200,
                    "unbounded {field} message: {} bytes",
                    message.len()
                );
            }
        }
    }

    #[test]
    fn v3_reports_a_present_but_non_string_kdf_as_unsupported_not_missing() {
        // `.as_str().ok_or("Missing kdf field")` told the user a field they can plainly
        // see is absent. Absence and wrong-type are both "unsupported" now.
        for value in [
            serde_json::json!(5),
            serde_json::json!({ "name": "scrypt" }),
        ] {
            let mut ks: serde_json::Value =
                serde_json::from_str(&valid_ethers_keystore(32)).unwrap();
            ks["crypto"]["kdf"] = value.clone();
            let err = decrypt_ethers_keystore(&ks.to_string(), &test_password(0))
                .expect_err("must be rejected");
            assert!(
                format!("{err}").contains("Unsupported kdf"),
                "kdf={value}, got {err}"
            );
        }
    }

    #[test]
    fn a_file_sourced_nonce_length_is_a_malformed_keystore_not_a_crypto_error() {
        // `CryptoError` is what a wrong password returns, so reporting a bad nonce that
        // way sends the caller round a password retry loop for a file that can never
        // decrypt. Everything else read out of the file is a DeserializationError.
        let err = decrypt_keystore(&keystore_v1_with_nonce_hex("deadbeef"), &test_password(0))
            .expect_err("must be rejected");
        assert!(
            matches!(err, KmsError::DeserializationError(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn decrypt_ethers_keystore_rejects_short_dklen_without_panicking() {
        // 10..=31 is the exact panic window: scrypt allows 10..=64, but anything
        // under 32 makes `derived_key[16..32]` go out of bounds.
        for dklen in [10u64, 15, 16, 17, 31] {
            let err = decrypt_ethers_keystore(&ethers_keystore_with(dklen, 16), &test_password(0))
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
            let err = decrypt_ethers_keystore(&ethers_keystore_with(dklen, 16), &test_password(0))
                .expect_err("MAC must fail");
            assert!(
                format!("{err}").contains("MAC verification failed"),
                "dklen={dklen}, got {err}"
            );
        }
    }

    #[test]
    fn decrypt_ethers_keystore_rejects_bad_iv_length_without_panicking() {
        // `Aes128Ctr::new` converts the IV with `GenericArray::from_slice`, which
        // asserts on a length mismatch. Asserting on the message, not just `is_err`,
        // so the case cannot pass for an unrelated reason.
        for iv_len in [0usize, 8, 15, 17, 32] {
            let mut ks: serde_json::Value =
                serde_json::from_str(&valid_ethers_keystore(32)).unwrap();
            ks["crypto"]["cipherparams"]["iv"] =
                serde_json::json!(hex::encode(vec![0xcdu8; iv_len]));
            let err = decrypt_ethers_keystore(&ks.to_string(), &test_password(0))
                .expect_err("must be rejected");
            assert!(
                format!("{err}").contains("Invalid cipherparams.iv length"),
                "iv_len={iv_len}, got {err}"
            );
        }
    }

    #[test]
    fn well_formed_dklen_and_iv_reach_mac_verification() {
        // Guards against over-tightening: dklen=32 with a 16-byte IV must get
        // past the new length checks and fail on the MAC instead.
        let err = decrypt_ethers_keystore(&ethers_keystore_with(32, 16), &test_password(0))
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

    /// `n` values a newly written keystore must refuse.
    const BAD_SCRYPT_N_ENCRYPT: [u32; 4] = [
        1000,    // floored to 512
        100_000, // floored to 65536, a weaker KDF
        0,       // log2(0) is -inf
        1,       // N=1 is a no-op KDF
    ];

    /// `n` values no keystore can legitimately hold, so reading must refuse them.
    /// Both formats refuse the same set.
    ///
    /// `1 << 31` is the memory case: `ScryptParams::new(31, 8, 1, 32)` returns Ok
    /// and scrypt then asks for 2 TiB, which aborts rather than erroring.
    const BAD_SCRYPT_N_DECRYPT: [u64; 5] = [
        0,                // log2(0) is undefined
        1,                // N=1 is a no-op KDF
        1 << 24,          // 16 GiB at r=8
        1 << 31,          // 2 TiB at r=8
        (1 << 32) + 1024, // truncated to 1024 by `as u32`
    ];

    #[test]
    fn encrypt_keystore_rejects_non_power_of_two_n() {
        // Strict here: flooring weakens the KDF the caller asked for.
        for n in BAD_SCRYPT_N_ENCRYPT {
            assert!(
                encrypt_keystore("test mnemonic", &test_password(0), n).is_err(),
                "n={n} must be rejected"
            );
        }
    }

    #[test]
    fn decrypt_keystore_rejects_non_power_of_two_n() {
        // Reading is as strict as writing: flooring 1500 to 1024 would derive a key
        // the file was never encrypted with and report it as a wrong password.
        let keystore = encrypt_keystore("m", &test_password(0), TEST_SCRYPT_N).unwrap();
        let mut ks: serde_json::Value = serde_json::from_str(&keystore).unwrap();
        ks["crypto"]["kdfparams"]["n"] = serde_json::json!(1500);

        let err =
            decrypt_keystore(&ks.to_string(), &test_password(0)).expect_err("must be rejected");
        assert!(
            format!("{err}").contains("power of two"),
            "should blame the params, not the password: {err}"
        );
    }

    #[test]
    fn decrypt_keystore_rejects_mismatched_kdfparams() {
        // v1 always derives with r=8, p=1, dklen=32. A file claiming otherwise would
        // decrypt to the wrong key and surface as a password failure. A wrong-typed
        // value counts, not just a wrong number -- `"8"`, `8.0` and `-1` are the cases
        // an `unwrap_or(8)` would have waved through.
        let cases = [
            ("r", serde_json::json!(16u64)),
            ("p", serde_json::json!(2)),
            ("dklen", serde_json::json!(64)),
            ("r", serde_json::json!("16")),
            ("r", serde_json::json!(16.0)),
            ("r", serde_json::json!(-1)),
            ("p", serde_json::json!("1")),
            ("dklen", serde_json::json!(true)),
        ];
        for (field, value) in cases {
            let keystore = encrypt_keystore("m", &test_password(0), TEST_SCRYPT_N).unwrap();
            let mut ks: serde_json::Value = serde_json::from_str(&keystore).unwrap();
            ks["crypto"]["kdfparams"][field] = value.clone();
            let err =
                decrypt_keystore(&ks.to_string(), &test_password(0)).expect_err("must be rejected");
            assert!(
                format!("{err}").contains(&format!("kdfparams.{field}")),
                "{field}={value}, got {err}"
            );
        }
    }

    #[test]
    fn decrypt_keystore_rejects_mismatched_kdf_and_cipher() {
        // v1 derives with scrypt and decrypts with XChaCha20-Poly1305 unconditionally,
        // so a file naming anything else fails in the AEAD and reads as a wrong
        // password. Same failure the kdfparams check above exists to prevent.
        for (field, value) in [("kdf", "pbkdf2"), ("cipher", "aes-128-ctr")] {
            let keystore = encrypt_keystore("m", &test_password(0), TEST_SCRYPT_N).unwrap();
            let mut ks: serde_json::Value = serde_json::from_str(&keystore).unwrap();
            ks["crypto"][field] = serde_json::json!(value);
            let err =
                decrypt_keystore(&ks.to_string(), &test_password(0)).expect_err("must be rejected");
            assert!(
                format!("{err}").contains(&format!("Unsupported {field}")),
                "{field}={value}, got {err}"
            );
        }
    }

    #[test]
    fn decrypt_keystore_requires_kdf_and_cipher() {
        // Absence is a rejection, not a default: `encrypt_keystore` has always written
        // both, so a file missing one is not a v1 keystore.
        for field in ["kdf", "cipher"] {
            let keystore = encrypt_keystore("m", &test_password(0), TEST_SCRYPT_N).unwrap();
            let mut ks: serde_json::Value = serde_json::from_str(&keystore).unwrap();
            ks["crypto"].as_object_mut().unwrap().remove(field);
            let err =
                decrypt_keystore(&ks.to_string(), &test_password(0)).expect_err("must be rejected");
            assert!(
                format!("{err}").contains(&format!("Unsupported {field}")),
                "absent {field}, got {err}"
            );
        }
    }

    #[test]
    fn decrypt_keystore_requires_kdfparams() {
        for field in ["r", "p", "dklen"] {
            let keystore = encrypt_keystore("m", &test_password(0), TEST_SCRYPT_N).unwrap();
            let mut ks: serde_json::Value = serde_json::from_str(&keystore).unwrap();
            ks["crypto"]["kdfparams"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            let err =
                decrypt_keystore(&ks.to_string(), &test_password(0)).expect_err("must be rejected");
            assert!(
                format!("{err}").contains(&format!("kdfparams.{field}")),
                "absent {field}, got {err}"
            );
        }
    }

    #[test]
    fn decrypt_keystore_rejects_unusable_n() {
        for n in BAD_SCRYPT_N_DECRYPT {
            let mut ks: serde_json::Value =
                serde_json::from_str(&keystore_v1_with_nonce_hex(&hex::encode([0u8; 24]))).unwrap();
            ks["crypto"]["kdfparams"]["n"] = serde_json::json!(n);
            let err =
                decrypt_keystore(&ks.to_string(), &test_password(0)).expect_err("must be rejected");
            assert!(
                matches!(err, KmsError::DeserializationError(_)),
                "n={n}, got {err:?}"
            );
        }
    }

    #[test]
    fn decrypt_ethers_keystore_rejects_unusable_n() {
        for n in BAD_SCRYPT_N_DECRYPT {
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
    fn decrypt_ethers_keystore_rejects_wrong_cipher() {
        // A non-string is a mismatch too, not an absent field.
        for cipher in [
            serde_json::json!("aes-256-ctr"),
            serde_json::json!("aes-128-cbc"),
            serde_json::json!(""),
            serde_json::json!("AES-128-CTR"),
            serde_json::json!(128),
        ] {
            let mut ks: serde_json::Value =
                serde_json::from_str(&valid_ethers_keystore(32)).unwrap();
            ks["crypto"]["cipher"] = cipher.clone();
            let err = decrypt_ethers_keystore(&ks.to_string(), &test_password(0))
                .expect_err("must be rejected");
            assert!(
                matches!(err, KmsError::DeserializationError(_)),
                "cipher={cipher}, got {err:?}"
            );
        }
    }

    #[test]
    fn decrypt_ethers_keystore_requires_cipher() {
        // Absence is not tolerated here, unlike the v1 header. A real aes-256-ctr
        // keystore with the field stripped still MAC-verifies -- the MAC does not
        // cover the cipher id -- and would decrypt as AES-128-CTR into a silently
        // wrong private key. v1 cannot fail this way: its AEAD authenticates.
        let mut ks: serde_json::Value = serde_json::from_str(&valid_ethers_keystore(32)).unwrap();
        ks["crypto"].as_object_mut().unwrap().remove("cipher");
        let err = decrypt_ethers_keystore(&ks.to_string(), &test_password(0))
            .expect_err("absent cipher must be rejected");
        assert!(format!("{err}").contains("Unsupported cipher"), "got {err}");
    }

    #[test]
    fn decrypt_ethers_keystore_rejects_out_of_range_dklen() {
        // 2^32 + 32 narrows to an acceptable 32 on wasm32.
        for dklen in [65u64, 128, 1 << 20, (1 << 32) + 32] {
            let mut ks: serde_json::Value =
                serde_json::from_str(&valid_ethers_keystore(32)).unwrap();
            ks["crypto"]["kdfparams"]["dklen"] = serde_json::json!(dklen);
            let err = decrypt_ethers_keystore(&ks.to_string(), &test_password(0))
                .expect_err("must be rejected");
            assert!(
                matches!(err, KmsError::DeserializationError(_)),
                "dklen={dklen}, got {err:?}"
            );
        }
    }

    #[test]
    fn decrypt_ethers_keystore_rejects_memory_bomb() {
        // ~10 GiB, rejected before allocating. Weaken the guard and this test stops
        // failing and starts hanging the machine.
        let mut ks: serde_json::Value = serde_json::from_str(&valid_ethers_keystore(32)).unwrap();
        ks["crypto"]["kdfparams"]["n"] = serde_json::json!(2);
        ks["crypto"]["kdfparams"]["r"] = serde_json::json!(4_194_304);
        ks["crypto"]["kdfparams"]["p"] = serde_json::json!(16);
        let err = decrypt_ethers_keystore(&ks.to_string(), &test_password(0))
            .expect_err("must be rejected");
        assert!(matches!(err, KmsError::DeserializationError(_)), "{err:?}");
    }

    #[test]
    fn decrypt_ethers_keystore_reports_odd_n_as_a_params_error() {
        // Flooring a corrupt `n` would derive a bogus key and blame the password.
        let mut ks: serde_json::Value = serde_json::from_str(&valid_ethers_keystore(32)).unwrap();
        ks["crypto"]["kdfparams"]["n"] = serde_json::json!(1500);
        let err = decrypt_ethers_keystore(&ks.to_string(), &test_password(0))
            .expect_err("must be rejected");
        assert!(
            format!("{err}").contains("power of two"),
            "should blame the params, not the password: {err}"
        );
    }

    #[test]
    fn decrypt_ethers_keystore_rejects_unusable_r_and_p() {
        // `r = 1000000` with n=2^20 passes `ScryptParams::new` and requests 125 TiB.
        // `as u32` also truncated 2^32 + 8 to a valid-looking 8.
        for (r, p) in [
            (0u64, 1u64),
            (1_000_000, 1),
            ((1 << 32) + 8, 1),
            (8, 0),
            (8, 100_000),
            (8, (1 << 32) + 1),
        ] {
            let mut ks: serde_json::Value =
                serde_json::from_str(&valid_ethers_keystore(32)).unwrap();
            ks["crypto"]["kdfparams"]["r"] = serde_json::json!(r);
            ks["crypto"]["kdfparams"]["p"] = serde_json::json!(p);
            let err = decrypt_ethers_keystore(&ks.to_string(), &test_password(0))
                .expect_err("must be rejected");
            assert!(
                matches!(err, KmsError::DeserializationError(_)),
                "r={r}, p={p}, got {err:?}"
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

    #[test]
    fn decrypt_ethers_keystore_rejects_resource_exhausting_params() {
        // Carried over from `validate_scrypt_resource_params`, retargeted at the
        // ceilings that replaced it. The dklen cases moved to the two tests below,
        // which go through `decrypt_ethers_keystore` where that check now lives.
        let n = u64::from(TEST_SCRYPT_N);
        assert!(scrypt_params(n, 8, 1, 32).is_ok());

        // The memory ceiling still refuses the case the flat caps were aimed at.
        assert!(scrypt_params(1 << 20, 32, 1, 32).is_err());

        // DELIBERATE RELAXATION. The flat `r <= 32` and `p <= 16` caps are gone,
        // replaced by ceilings on what `r` and `p` actually cost. Web3 Secret Storage
        // puts no bound on either, and at a small `N` a large one is cheap -- `r = 33`
        // here is 4 MiB, which the flat cap refused for its shape rather than its cost.
        assert!(scrypt_params(n, 33, 1, 32).is_ok());
        assert!(scrypt_params(n, 8, 17, 32).is_ok());

        // The same parameters priced out of range are still refused, which is what the
        // flat caps were standing in for: `r` over the memory ceiling, `p` over work.
        assert!(scrypt_params(n, 4096, 1, 32).is_err());
        assert!(scrypt_params(1 << 20, 2, 16, 32).is_err());
    }

    fn ethers_keystore_with_dklen(dklen: u64) -> String {
        let keystore = serde_json::json!({
            "version": 3,
            "crypto": {
                "cipher": "aes-128-ctr",
                "kdf": "scrypt",
                "kdfparams": {
                    "n": TEST_SCRYPT_N,
                    "r": 8,
                    "p": 1,
                    "dklen": dklen,
                    "salt": hex::encode([0xab; 32]),
                },
                "cipherparams": { "iv": hex::encode([0xcd; 16]) },
                "ciphertext": hex::encode([0u8; 32]),
                "mac": hex::encode([0u8; 32]),
            }
        });
        serde_json::to_string(&keystore).unwrap()
    }

    #[test]
    fn decrypt_ethers_keystore_rejects_dklen_below_32() {
        // A crafted keystore with dklen < 32 must return an error instead of
        // panicking on derived_key[16..32].
        let keystore_json = ethers_keystore_with_dklen(16);
        let result = decrypt_ethers_keystore(&keystore_json, &test_password(0));
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("dklen"), "unexpected error: {err_msg}");
    }

    #[test]
    fn decrypt_ethers_keystore_rejects_truncating_dklen() {
        // 2^32 + 32 truncates to 32 under `as usize` on wasm32; the checked
        // conversion must reject it before the range validation runs.
        let keystore_json = ethers_keystore_with_dklen((1u64 << 32) + 32);
        let result = decrypt_ethers_keystore(&keystore_json, &test_password(0));
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("dklen"), "unexpected error: {err_msg}");
    }
}
