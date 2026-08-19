#!/usr/bin/env bash
# Lightweight secret-handling hygiene checks for production crates.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
failed=0

# 1) SecretFelt must not implement Display (would encourage logging secrets).
if grep -R --include='*.rs' -nE 'impl\s+(core::fmt::|std::fmt::)?Display\s+for\s+SecretFelt' \
  "$root/crates/common" >/dev/null; then
  echo "::error::SecretFelt must not implement Display"
  failed=1
fi

# 2) SecretFelt Debug must stay redacted (smoke check on source).
if ! grep -n 'SecretFelt(\*\*\*)' "$root/crates/common/src/secret_felt.rs" >/dev/null; then
  echo "::error::SecretFelt Debug redaction marker missing"
  failed=1
fi

# 3) Binding keypair stringifiers must redact the private key. toString()/
# description fire implicitly from interpolation, exception messages and crash
# reporters, so a leak here reaches logs with no call site that looks like it
# touches a secret. The C ABI exposes exactly two secret-carrying aggregates
# (KmsTongoKeyPair, KmsNostrKeyPair), so each binding has two to keep redacted.
# Swift's failure mode is the *absence* of a CustomStringConvertible conformance:
# without one, default struct reflection prints all 32 bytes.
#
# Markers are matched as fixed strings, not regexes, so nothing here needs
# per-language escaping.

# marker_present <file> <fixed-string> <message>
marker_present() {
  if ! grep -qF -- "$2" "$root/$1"; then
    echo "::error file=$1::$3"
    failed=1
  fi
}

# The marker is the only place `privateKey` may appear in a stringifier. Strip
# it literally; any surviving mention on that line is a leak whatever syntax it
# uses — `+` concatenation, a `%s` format argument, or `${...}` interpolation.
# marker_is_only_mention <file> <marker-fixed-string> <message>
marker_is_only_mention() {
  local line leaked=0
  while IFS= read -r line; do
    case "${line//"$2"/}" in *privateKey*) leaked=1 ;; esac
  done < <(grep -F -- "$2" "$root/$1" || true)
  if (( leaked == 1 )); then
    echo "::error file=$1::$3"
    failed=1
  fi
}

# assert_redacted <file> <marker> <label>
assert_redacted() {
  marker_present "$1" "$2" "$3 redaction marker missing"
  marker_is_only_mention "$1" "$2" "$3 leaks the private key alongside the marker"
}

dart_types="packages/kms-dart/lib/src/types.dart"
assert_redacted "$dart_types" 'TongoKeyPair(privateKey: ***' "Dart TongoKeyPair.toString()"
assert_redacted "$dart_types" 'NostrKeyPair(privateKey: [${privateKey.length} bytes]' \
  "Dart NostrKeyPair.toString()"

jvm_dir="packages/kms-jvm/src/main/java/io/krustykms"
assert_redacted "$jvm_dir/TongoKeyPair.java" 'TongoKeyPair(privateKey=***' \
  "Java TongoKeyPair.toString()"
assert_redacted "$jvm_dir/NostrKeyPair.java" 'NostrKeyPair(privateKey=[32 bytes]' \
  "Java NostrKeyPair.toString()"

swift_src="packages/kms-swift/Sources/KrustyKms/KrustyKms.swift"
for t in TongoKeyPair NostrKeyPair; do
  # Without the conformance, default struct reflection prints the key.
  marker_present "$swift_src" "extension $t: CustomStringConvertible" \
    "Swift $t leaks its private key via default struct reflection"
  assert_redacted "$swift_src" "$t(privateKey: ***" "Swift $t.description"
done

# Advisory only: discourage println of exposed secrets outside tests.
while IFS= read -r match; do
  echo "::warning::$match"
done < <(grep -R --include='*.rs' -nE 'println!.*expose_secret|eprintln!.*expose_secret' \
  "$root/crates" \
  --exclude-dir experimental \
  --exclude-dir target \
  --exclude-dir tests \
  || true)

if (( failed == 1 )); then
  exit 1
fi
echo "secret hygiene ok"
