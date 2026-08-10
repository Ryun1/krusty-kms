//! Private key and payload encryption using XChaCha20-Poly1305 with scrypt KDF.
//!
//! This module provides:
//! - Password-based encryption/decryption of private keys (scrypt + XChaCha20-Poly1305)
//! - Direct key-based encryption/decryption for arbitrary payloads

use chacha20poly1305::{
    aead::{Aead, KeyInit},
    XChaCha20Poly1305, XNonce,
};
use krusty_kms_common::{KmsError, Result};
use scrypt::{scrypt, Params as ScryptParams};
use zeroize::Zeroizing;

/// XChaCha20-Poly1305 extended nonce length, in bytes.
pub const XNONCE_LEN: usize = 24;

/// Convert untrusted bytes into a nonce, or reject them.
///
/// The way to build the `nonce` field on [`EncryptedKey`] and [`EncryptedPayload`]
/// from a decoded hex string. Length is checked here, at the parse boundary, so it
/// cannot reach [`XNonce::from_slice`] -- which asserts rather than erroring, and was
/// the original panic. Everything downstream holds a fixed-size array and needs no
/// guard of its own.
pub fn xnonce(bytes: &[u8]) -> Result<[u8; XNONCE_LEN]> {
    bytes.try_into().map_err(|_| {
        KmsError::DeserializationError(format!(
            "Invalid nonce length: expected {XNONCE_LEN} bytes, got {}",
            bytes.len()
        ))
    })
}

/// Encrypted private key with KDF salt.
///
/// The nonce is a fixed-size array rather than a `Vec`, so a wrong length is
/// unrepresentable instead of being caught by a guard at each point of use.
#[derive(Debug, Clone)]
pub struct EncryptedKey {
    /// 24-byte XChaCha20 nonce. Build from untrusted bytes with [`xnonce`].
    pub nonce: [u8; XNONCE_LEN],
    /// 16-byte scrypt salt.
    pub salt: Vec<u8>,
    /// Ciphertext with 16-byte Poly1305 authentication tag appended.
    pub encrypted_key: Vec<u8>,
}

/// Encrypted payload (no KDF metadata -- caller provides key directly).
#[derive(Debug, Clone)]
pub struct EncryptedPayload {
    /// 24-byte XChaCha20 nonce. Build from untrusted bytes with [`xnonce`].
    pub nonce: [u8; XNONCE_LEN],
    /// Ciphertext with 16-byte Poly1305 authentication tag appended.
    pub ciphertext: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Smallest scrypt `N` accepted, carried over from the audit hardening.
///
/// A floor on *strength*, where the ceilings below are a ceiling on *cost*; the two
/// are independent and both are needed. `scrypt_log_n` folded into [`scrypt_params`]
/// so there is one door rather than two, and this came with it: without the floor a
/// caller could write a keystore at `N = 2`, which this crate already refused.
pub(crate) const SCRYPT_N_MIN: u64 = 1 << 10;

/// Largest scrypt `N` accepted, likewise carried over.
///
/// The memory ceiling below would admit more at a small `r`. This is the tighter of
/// the two and is what a bare `N` is held to.
pub(crate) const SCRYPT_N_MAX: u64 = 1 << 20;

/// Memory ceiling for a scrypt derivation: `N` up to 2^18 at `r = 8`, geth's standard
/// strength.
///
/// That case needs 268,437,504 bytes -- 2048 more than a round 256 MiB, since the
/// total counts `p` and the temp buffer alongside `N` -- so the ceiling is the next
/// round 8 MiB above it. The slack is not load-bearing; the next `N` up is twice this,
/// not marginally over.
///
/// One ceiling for reading and writing, so "anything this build writes it can read
/// again" holds by construction rather than by comparing two numbers. Not
/// target-dependent either: a file one build accepts must not be a trap in another.
const MAX_SCRYPT_MEM: u64 = 264 << 20;

/// Cap on scrypt's work factor, `N * r * p`. The memory ceiling does not bound this:
/// `{n: 2^22, r: 1, p: 2^22}` fits in 1 GiB and asks for ~10^13 block operations.
///
/// Eight times geth's heaviest standard setting (`N = 2^18` at `r = 8`). What that
/// buys in `p` depends on `N`: 8 at geth's heaviest, 512 at `N = 2^12`, i.e. room
/// for a large `p` exactly where a large `p` is cheap. Anything beyond hashes for
/// hours, which is a denial of service whatever the memory use.
const MAX_SCRYPT_WORK: u64 = 1 << 24;

// Representability only -- whether the target can actually grow to the ceiling is
// checked at runtime below, since that is not a compile-time property.
const _: () = assert!(MAX_SCRYPT_MEM <= 1 << (usize::BITS - 1));

/// Build [`ScryptParams`] from untrusted values - the only way to get them here.
///
/// Returns the projected allocation alongside them, for [`scrypt_derive`] to probe.
/// Validation itself allocates nothing: the sizes under test here are exactly the
/// ones that must never be allocated to find out they are too large.
pub(crate) fn scrypt_params(n: u64, r: u64, p: u64, dklen: usize) -> Result<(ScryptParams, u64)> {
    let bad = |why| {
        KmsError::DeserializationError(format!(
            "Unsupported scrypt params (n={n}, r={r}, p={p}): {why}"
        ))
    };

    // Strength floor and ceiling, independent of the cost ceilings below: a weak `N`
    // is cheap, so nothing further down would ever catch it.
    if !(SCRYPT_N_MIN..=SCRYPT_N_MAX).contains(&n) {
        return Err(bad("n must be in [2^10, 2^20]"));
    }
    // Exact, never floored: deriving with the nearest power of two below a corrupt `n`
    // produces a bogus key and reports it as a wrong password.
    if !n.is_power_of_two() {
        return Err(bad("n must be a power of two"));
    }
    if r == 0 || p == 0 {
        return Err(bad("r and p must be non-zero"));
    }

    // `n` is an exact power of two by here, so this round-trips and the arithmetic
    // below can use `n` itself.
    let log_n = n.ilog2() as u8;
    let work = n.checked_mul(r).and_then(|w| w.checked_mul(p));
    match work {
        Some(w) if w <= MAX_SCRYPT_WORK => {}
        _ => return Err(bad("exceeds the scrypt work ceiling")),
    }

    // scrypt 0.11 allocates `128*r*N` + `128*r*p` + `128*r`, so all three need
    // bounding together: capping the first alone let `{n: 2, r: 2^22, p: 16}`
    // through at 1 GiB while really asking ~10 GiB.
    let total = 128u64
        .checked_mul(r)
        .and_then(|acc| acc.checked_mul(n + p + 1));
    let bytes = match total {
        Some(bytes) if bytes <= MAX_SCRYPT_MEM => bytes,
        _ => return Err(bad("exceeds the scrypt memory ceiling")),
    };

    // Both casts are lossless: the work ceiling holds `r` and `p` under 2^24 each,
    // since the other two factors are at least 1.
    //
    // `bad` rather than a `CryptoError`: scrypt's own rules -- `log_n < r * 16`, and
    // `r * p < 2^30` -- reject file-supplied params just like the ceilings above, and
    // that has to stay distinguishable from a wrong password. `InvalidParams` carries
    // no detail beyond the values already in the message.
    let params = ScryptParams::new(log_n, r as u32, p as u32, dklen)
        .map_err(|_| bad("rejected by scrypt's own parameter rules"))?;
    Ok((params, bytes))
}

/// Validate untrusted params, then derive into `out`.
///
/// The single door to [`scrypt`]: nothing else in the crate should call it, since
/// the probe below is not optional on wasm32.
pub(crate) fn scrypt_derive(
    password: &[u8],
    kdf_salt: &[u8],
    n: u64,
    r: u64,
    p: u64,
    out: &mut [u8],
) -> Result<()> {
    let (params, bytes) = scrypt_params(n, r, p, out.len())?;

    // Under the ceiling is not the same as allocatable. scrypt allocates internally
    // and an allocation failure aborts rather than unwinding, so on wasm32 -- where
    // the ceiling is a large fraction of what an instance can grow to -- a permitted
    // file would trap the module instead of erroring. Probing fallibly first keeps
    // that a `Result` without making the ceiling target-dependent, which would turn a
    // file one build accepts into a trap in another.
    //
    // Only the wasm32 case is load-bearing. Where the allocator overcommits, this
    // succeeds without committing pages and the ceiling above is what bounds the
    // damage; the probe is not a promise about hosts that overcommit and then OOM.
    // Placed after `ScryptParams::new` so params it rejects -- `{n: 2^16, r: 1}`
    // fails its `log_n < r * 16` rule -- cost no allocation at all.
    //
    // Three reservations held at once, matching scrypt's own `128*r*p`, `128*r*n` and
    // `128*r`, rather than one block of their sum: the sum is not contiguous in scrypt
    // and demanding that it be here would refuse a legitimate keystore on a fragmented
    // heap -- a false rejection is a lost key, the failure this whole path avoids.
    let probes: std::result::Result<Vec<Vec<u8>>, _> = [128 * r * p, 128 * r * n, 128 * r]
        .into_iter()
        .map(|size| {
            let mut probe: Vec<u8> = Vec::new();
            probe.try_reserve_exact(size as usize).map(|()| probe)
        })
        .collect();
    probes.map_err(|_| {
        KmsError::DeserializationError(format!(
            "Unsupported scrypt params (n={n}, r={r}, p={p}): \
             needs {bytes} bytes, more than this process can allocate"
        ))
    })?;

    scrypt(password, kdf_salt, &params, out)
        .map_err(|e| KmsError::CryptoError(format!("Scrypt KDF failed: {e}")))
}

/// Re-label a param rejection whose values came from a function argument.
///
/// [`scrypt_params`] reports every rejection as a [`KmsError::DeserializationError`],
/// which is right for the paths that read `n` out of a keystore. The three that take
/// `scrypt_n` as an argument have no input to blame, so they wrap this around it --
/// cheaper than threading the distinction back through the KDF path, and it keeps
/// `scrypt_params` with one story about its own errors.
pub(crate) fn caller_param(err: KmsError) -> KmsError {
    match err {
        KmsError::DeserializationError(why) => KmsError::InvalidParameter(why),
        other => other,
    }
}

/// Derive a 32-byte key from a password and salt using scrypt (r=8, p=1).
///
/// [`Zeroizing`] rather than a bare array: every caller has a `?` between the
/// derivation and the end of the function, and a wrong password takes that path.
pub(crate) fn derive_scrypt_key(
    password: &[u8],
    kdf_salt: &[u8],
    n: u64,
) -> Result<Zeroizing<[u8; 32]>> {
    let mut key = Zeroizing::new([u8::default(); 32]);
    scrypt_derive(password, kdf_salt, n, 8, 1, &mut *key)?;
    Ok(key)
}

// ---------------------------------------------------------------------------
// Password-based private key encryption
// ---------------------------------------------------------------------------

/// Encrypt a hex-encoded private key with a password using scrypt + XChaCha20-Poly1305.
///
/// # Arguments
/// * `private_key_hex` - Hex-encoded private key (with or without `0x` prefix)
/// * `password` - User-supplied password
/// * `scrypt_n` - Scrypt cost parameter N: a power of two from 2 to 262144 (2^18,
///   geth's standard strength), e.g. 32768. Larger is refused here rather than
///   written, since a file this build cannot read again is a lost key.
///
/// # Returns
/// An [`EncryptedKey`] containing the nonce, salt, and ciphertext.
pub fn encrypt_private_key(
    private_key_hex: &str,
    password: &str,
    scrypt_n: u32,
) -> Result<EncryptedKey> {
    // Decode hex private key. Before the KDF, not after: a malformed argument should
    // not cost a full derivation first -- up to ~2 s and 264 MiB at the ceiling.
    let hex_str = private_key_hex
        .strip_prefix("0x")
        .unwrap_or(private_key_hex);
    let plaintext =
        hex::decode(hex_str).map_err(|e| KmsError::CryptoError(format!("Invalid hex: {e}")))?;

    // Generate 16-byte salt
    let salt = krusty_kms_crypto::random_bytes::<16>();

    // Derive encryption key
    let key =
        derive_scrypt_key(password.as_bytes(), &salt, scrypt_n.into()).map_err(caller_param)?;

    // Generate 24-byte nonce
    let nonce_bytes = krusty_kms_crypto::random_bytes::<XNONCE_LEN>();

    // Encrypt
    let cipher = XChaCha20Poly1305::new_from_slice(key.as_slice())
        .map_err(|e| KmsError::CryptoError(format!("Invalid key: {e}")))?;
    let nonce = XNonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_ref())
        .map_err(|e| KmsError::CryptoError(format!("Encryption failed: {e}")))?;

    Ok(EncryptedKey {
        nonce: nonce_bytes,
        salt: salt.to_vec(),
        encrypted_key: ciphertext,
    })
}

/// Decrypt a private key that was encrypted with [`encrypt_private_key`].
///
/// # Arguments
/// * `encrypted` - The encrypted key bundle
/// * `password` - The password used during encryption
/// * `scrypt_n` - The same scrypt cost parameter used during encryption
///
/// # Returns
/// Hex-encoded private key (no `0x` prefix).
pub fn decrypt_private_key(
    encrypted: &EncryptedKey,
    password: &str,
    scrypt_n: u32,
) -> Result<String> {
    let nonce = XNonce::from_slice(&encrypted.nonce);

    // `scrypt_n` is an argument here too, not read from the payload.
    let key = derive_scrypt_key(password.as_bytes(), &encrypted.salt, scrypt_n.into())
        .map_err(caller_param)?;

    // Decrypt
    let cipher = XChaCha20Poly1305::new_from_slice(key.as_slice())
        .map_err(|e| KmsError::CryptoError(format!("Invalid key: {e}")))?;
    // `Zeroizing` for the same reason as the derived key above: this buffer holds the
    // raw private key, so leaving it in freed memory would scrub the wrong half.
    let plaintext = Zeroizing::new(
        cipher
            .decrypt(nonce, encrypted.encrypted_key.as_ref())
            .map_err(|e| KmsError::CryptoError(format!("Decryption failed: {e}")))?,
    );

    Ok(hex::encode(&*plaintext))
}

// ---------------------------------------------------------------------------
// Direct key-based encryption
// ---------------------------------------------------------------------------

/// Encrypt arbitrary data with a pre-derived 32-byte key.
///
/// # Arguments
/// * `plaintext` - Raw bytes to encrypt
/// * `key` - 32-byte symmetric key
///
/// # Returns
/// An [`EncryptedPayload`] containing the nonce and ciphertext.
pub fn encrypt_with_key(plaintext: &[u8], key: &[u8; 32]) -> Result<EncryptedPayload> {
    let nonce_bytes = krusty_kms_crypto::random_bytes::<XNONCE_LEN>();

    let cipher = XChaCha20Poly1305::new_from_slice(key)
        .map_err(|e| KmsError::CryptoError(format!("Invalid key: {e}")))?;
    let nonce = XNonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| KmsError::CryptoError(format!("Encryption failed: {e}")))?;

    Ok(EncryptedPayload {
        nonce: nonce_bytes,
        ciphertext,
    })
}

/// Decrypt data that was encrypted with [`encrypt_with_key`].
///
/// # Arguments
/// * `payload` - The encrypted payload
/// * `key` - The same 32-byte symmetric key used during encryption
///
/// # Returns
/// The decrypted plaintext bytes.
pub fn decrypt_with_key(payload: &EncryptedPayload, key: &[u8; 32]) -> Result<Vec<u8>> {
    let cipher = XChaCha20Poly1305::new_from_slice(key)
        .map_err(|e| KmsError::CryptoError(format!("Invalid key: {e}")))?;
    let nonce = XNonce::from_slice(&payload.nonce);
    cipher
        .decrypt(nonce, payload.ciphertext.as_ref())
        .map_err(|e| KmsError::CryptoError(format!("Decryption failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use a low scrypt N for fast tests
    const TEST_SCRYPT_N: u32 = 1024;

    // `scrypt_params` only validates, never runs scrypt -- the only safe way to
    // assert on sizes that would abort the process.

    #[test]
    fn scrypt_params_bounds_all_three_allocations() {
        // scrypt allocates `128*r*N` + `128*r*p` + `128*r`, so the ceiling counts all
        // three; capping the first alone used to let the rest through.
        //
        // These cases had to be retargeted when the `N >= 2^10` floor came in. They
        // used to use `N = 2` so that `p` dominated the sum. With the floor, the work
        // ceiling holds `r*p` under 2^14, so the `p` buffer can never exceed ~2 MiB --
        // the pair below straddles the memory ceiling on exactly that margin. Both are
        // far under the work ceiling and differ only in `p`.
        assert!(scrypt_params(1024, 2100, 1, 32).is_ok());
        assert!(scrypt_params(1024, 2100, 7, 32).is_err());

        // `r` alone over the memory ceiling, still under the work ceiling.
        assert!(scrypt_params(1024, 4096, 1, 32).is_err());

        // And an `N` past its own ceiling, whatever `r` and `p` are.
        assert!(scrypt_params(1 << 31, 1, 1, 32).is_err());
    }

    #[test]
    fn scrypt_params_bounds_work_the_memory_ceiling_misses() {
        // Retargeted for the `N <= 2^20` ceiling: the old case used `N = 2^22`, which
        // the ceiling now rejects first, so it would have passed without the work
        // ceiling existing at all. `N` at its maximum with `p = 16` is 2^25 block
        // operations in 256 MiB -- under the memory ceiling, so only the work ceiling
        // catches it. `r = 2` rather than 1 because scrypt's own `log_n < r * 16` rule
        // rejects `log_n = 20` at `r = 1`, which would be a third reason to fail.
        assert!(scrypt_params(1 << 20, 2, 16, 32).is_err());
        // The same shape one step down, to show it is the work ceiling that moved:
        // 2^24 exactly, which is the limit rather than past it.
        assert!(scrypt_params(1 << 20, 2, 8, 32).is_ok());

        // A large `p` is valid under Web3 Secret Storage and cheap when `N` is small,
        // so the work ceiling has to admit it -- the flat `p <= 16` cap did not.
        assert!(scrypt_params(4096, 8, 32, 32).is_ok());
    }

    #[test]
    fn scrypt_params_admits_geth_standard_strength_and_no_more() {
        // One ceiling for reading and writing, so every case here is both.
        assert!(scrypt_params(1 << 17, 8, 1, 32).is_ok());

        // geth's standard strength: the heaviest thing that fits.
        assert!(scrypt_params(1 << 18, 8, 1, 32).is_ok());
        assert!(scrypt_params(1 << 19, 8, 1, 32).is_err());
        assert!(scrypt_params(1 << 20, 8, 1, 32).is_err());

        // The memory ceiling still bites away from geth's `r = 8` shape, and this pins
        // where. The old pair used `N = 2` for this, which the strength floor now
        // rejects before the memory ceiling is consulted at all.
        assert!(scrypt_params(1024, 2107, 1, 32).is_ok());
        assert!(scrypt_params(1024, 2108, 1, 32).is_err());
    }

    #[test]
    fn scrypt_params_rejects_non_power_of_two_n() {
        // Never floored: 100_000 would derive with N=65536 and blame the password.
        assert!(scrypt_params(100_000, 8, 1, 32).is_err());
    }

    #[test]
    fn scrypt_params_rejects_before_any_allocation() {
        // scrypt requires `log_n < r * 16`, so this fails validation despite passing
        // both ceilings -- 8,388,864 bytes and 65,536 work units are well under. It
        // matters where: with the probe inside `scrypt_params` a param error still
        // reserved and freed the whole projection first, which is a free amplifier for
        // anyone who can submit keystores -- and on wasm32 the heap never shrinks
        // again. `scrypt_params` allocates nothing now; the probe is in `scrypt_derive`.
        assert!(scrypt_params(1 << 16, 1, 1, 32).is_err());
    }

    fn test_password(offset: u8) -> String {
        (0..12)
            .map(|index| char::from(b'a' + index + offset))
            .collect()
    }

    fn test_key(offset: u8) -> [u8; 32] {
        std::array::from_fn(|index| index as u8 + offset)
    }

    #[test]
    fn encrypt_decrypt_private_key_roundtrip() {
        let private_key = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        let password = test_password(0);

        let encrypted = encrypt_private_key(private_key, &password, TEST_SCRYPT_N).unwrap();
        assert_eq!(encrypted.nonce.len(), 24);
        assert_eq!(encrypted.salt.len(), 16);

        let decrypted = decrypt_private_key(&encrypted, &password, TEST_SCRYPT_N).unwrap();
        assert_eq!(decrypted, private_key);
    }

    #[test]
    fn encrypt_decrypt_private_key_with_0x_prefix() {
        let private_key = "0xdeadbeef00112233deadbeef00112233deadbeef00112233deadbeef00112233";
        let password = test_password(0);

        let encrypted = encrypt_private_key(private_key, &password, TEST_SCRYPT_N).unwrap();
        let decrypted = decrypt_private_key(&encrypted, &password, TEST_SCRYPT_N).unwrap();
        // Decrypted is returned without 0x prefix
        assert_eq!(
            decrypted,
            "deadbeef00112233deadbeef00112233deadbeef00112233deadbeef00112233"
        );
    }

    #[test]
    fn wrong_password_fails_decryption() {
        let private_key = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        let password = test_password(0);

        let encrypted = encrypt_private_key(private_key, &password, TEST_SCRYPT_N).unwrap();

        let wrong_password = test_password(1);
        let result = decrypt_private_key(&encrypted, &wrong_password, TEST_SCRYPT_N);
        assert!(result.is_err());
    }

    #[test]
    fn reject_invalid_scrypt_n() {
        // Carried over from the audit hardening and retargeted at `scrypt_params`,
        // which `scrypt_log_n` folded into. Same bounds, same cases: the strength
        // floor is not something the cost ceilings would ever have caught.
        assert!(scrypt_params(1023, 8, 1, 32).is_err());
        assert!(scrypt_params(3, 8, 1, 32).is_err());
        assert!(scrypt_params(1 << 21, 8, 1, 32).is_err());
        assert!(scrypt_params(1024, 8, 1, 32).is_ok());
    }

    #[test]
    fn encrypt_decrypt_with_key_roundtrip() {
        let plaintext = b"some secret data that must remain confidential";
        let key = test_key(0);

        let payload = encrypt_with_key(plaintext, &key).unwrap();
        assert_eq!(payload.nonce.len(), 24);

        let decrypted = decrypt_with_key(&payload, &key).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    // A wrong nonce length is now unrepresentable -- `EncryptedPayload { nonce: vec![0;
    // 4], .. }` does not compile -- so these test the one boundary that can still
    // produce one, rather than each point of use.

    #[test]
    fn xnonce_rejects_wrong_lengths_without_panicking() {
        // `XNonce::from_slice` asserts on a length mismatch, so a nonce taken from
        // untrusted JSON used to abort the process instead of erroring. It cannot
        // reach that call any more; this is where it stops.
        //
        // 12 is the ChaCha20-Poly1305 nonce length, i.e. the plausible wrong answer
        // rather than an arbitrary one. Carried over from #45, which added it to the
        // per-call-site tests this replaces.
        for len in [0usize, 4, 12, 23, 25, 64] {
            let err = xnonce(&vec![0u8; len]).expect_err("must be rejected");
            assert!(
                matches!(err, KmsError::DeserializationError(_)),
                "len={len}, got {err:?}"
            );
        }
    }

    #[test]
    fn xnonce_accepts_the_correct_length() {
        // Guards against over-tightening.
        assert_eq!(
            xnonce(&[7u8; XNONCE_LEN]).expect("must be accepted"),
            [7u8; XNONCE_LEN]
        );
    }

    #[test]
    fn a_well_formed_nonce_reaches_the_aead() {
        // The other half of over-tightening: a valid nonce must fail on the
        // authentication tag, not on length.
        let payload = EncryptedPayload {
            nonce: [0u8; XNONCE_LEN],
            ciphertext: vec![0u8; 48],
        };
        let err = decrypt_with_key(&payload, &test_key(0)).expect_err("tag must fail");
        assert!(format!("{err}").contains("Decryption failed"), "got {err}");
    }

    #[test]
    fn wrong_key_fails_decrypt_with_key() {
        let plaintext = b"secret";
        let key = test_key(0);
        let wrong_key = test_key(1);

        let payload = encrypt_with_key(plaintext, &key).unwrap();

        let result = decrypt_with_key(&payload, &wrong_key);
        assert!(result.is_err());
    }
}
