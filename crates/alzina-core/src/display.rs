//! Display safety helpers: terminal-control sanitisation and credential redaction.
//!
//! These are applied to low-authority data flowing into terminal renderers
//! (CLI, SSE payloads, audit logs). The shared D3 sanitiser pipeline in
//! `api::chat::agent_completed_payload` chains them together.

/// Sanitize a string for safe inclusion in a terminal-rendered payload.
///
/// Strips:
/// - C0 control codepoints `0x00..=0x1F` except `\n` (0x0A) and `\t` (0x09).
/// - C1 control codepoints `0x80..=0x9F`.
/// - Bidi-override codepoints: `U+202A..=U+202E` (LRE/RLE/PDF/LRO/RLO),
///   `U+2066..=U+2069` (LRI/RLI/FSI/PDI).
/// - DEL (`0x7F`).
///
/// Replaces `\r` (0x0D) with `\n`. Null bytes are stripped.
///
/// Behaviour is preserved for ordinary printable text and common whitespace.
pub fn sanitize_for_terminal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            // Replace CR with LF.
            '\r' => out.push('\n'),
            // Permitted controls.
            '\n' | '\t' => out.push(c),
            // Drop other C0 controls and DEL.
            c if (c as u32) <= 0x1F => continue,
            c if (c as u32) == 0x7F => continue,
            // Drop C1 controls.
            c if matches!(c as u32, 0x80..=0x9F) => continue,
            // Drop bidi overrides.
            '\u{202A}'..='\u{202E}' => continue,
            '\u{2066}'..='\u{2069}' => continue,
            other => out.push(other),
        }
    }
    out
}

/// Redact common credential shapes inside a string, replacing each match
/// with `[REDACTED:<class>]`.
///
/// Classes:
/// - `aws-access-key`: `AKIA[0-9A-Z]{16}`
/// - `github-token`: `gh[pousr]_[A-Za-z0-9_]{20,}`
/// - `bearer-token`: `Bearer <40+ chars from \w/-/.>`
/// - `private-key`: literal `private_key`/`private-key`
/// - `anthropic-key`: `sk-ant-` + 20+ token chars (covers `sk-ant-api03-…`,
///   `sk-ant-admin-…`, and future versioned shapes via the broader prefix)
/// - `openai-key`: `sk-` + 32+ token chars (covers `sk-proj-…`, legacy
///   `sk-…`); `sk-ant-` is matched first so it does not collide
/// - `stripe-key`: `sk_live_` / `rk_live_` / `whsec_` + 24+ token chars
///   (`pk_live_` publishable keys and `sk_test_` test keys are intentionally
///   NOT redacted — they are designed to be public or non-production)
/// - `jwt`: `eyJ` + base64url + `.` + base64url + `.` + base64url
///   (three segments, each 4+ chars; matches standard `{"alg":…}` header
///   prefix)
///
/// A high-entropy fallback used to be in this list but was removed
/// (2026-05-05) — it over-fired on file paths and any other 32+ char
/// alphanumeric run with reasonable character spread (UUIDs, hashes,
/// base64 chunks of normal data), which masked legitimate envelope
/// `ARTIFACTS` lines as `[REDACTED:high-entropy-token]`. Specific-shape
/// rules above have near-zero false-positive rate and cover the realistic
/// risk surface; if stronger protection is needed later it should live at
/// per-field granularity (e.g. redact `context_update`, leave `artifacts`
/// alone), not as a global text scanner.
///
/// Implemented without the `regex` crate — we walk the string ourselves
/// so alzina-core stays minimal-dep. A pure-rust scanner is plenty fast
/// for the small bodies (envelopes, kilobytes at most) that pass through
/// the chat SSE path.
pub fn redact_secrets(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        // AKIA + 16 [0-9A-Z]
        if i + 20 <= n && &bytes[i..i + 4] == b"AKIA" {
            if (i + 4..i + 20).all(|j| bytes[j].is_ascii_uppercase() || bytes[j].is_ascii_digit()) {
                // Boundary check: not part of a longer identifier
                let end = i + 20;
                let next_is_word = end < n && is_word_byte(bytes[end]);
                if !next_is_word {
                    out.push_str("[REDACTED:aws-access-key]");
                    i = end;
                    continue;
                }
            }
        }
        // GitHub tokens: gh[pousr]_ followed by 20+ word chars
        if i + 4 <= n && bytes[i] == b'g' && bytes[i + 1] == b'h' && bytes[i + 3] == b'_' {
            let prefix = bytes[i + 2];
            if matches!(prefix, b'p' | b'o' | b'u' | b's' | b'r') {
                let mut end = i + 4;
                while end < n && is_word_byte(bytes[end]) {
                    end += 1;
                }
                if end - (i + 4) >= 20 {
                    out.push_str("[REDACTED:github-token]");
                    i = end;
                    continue;
                }
            }
        }
        // Bearer <token>
        if i + 7 <= n && (&bytes[i..i + 7]).eq_ignore_ascii_case(b"Bearer ") {
            let token_start = i + 7;
            let mut end = token_start;
            while end < n && is_token_byte(bytes[end]) {
                end += 1;
            }
            if end - token_start >= 40 {
                out.push_str("[REDACTED:bearer-token]");
                i = end;
                continue;
            }
        }
        // private_key / private-key (case-insensitive)
        if i + 11 <= n {
            let chunk = &bytes[i..i + 11];
            if chunk.eq_ignore_ascii_case(b"private_key")
                || chunk.eq_ignore_ascii_case(b"private-key")
            {
                let prev_word = i > 0 && is_word_byte(bytes[i - 1]);
                let next_word = i + 11 < n && is_word_byte(bytes[i + 11]);
                if !prev_word && !next_word {
                    out.push_str("[REDACTED:private-key]");
                    i += 11;
                    continue;
                }
            }
        }
        // Anthropic API key: sk-ant- + 20+ token chars. Must run BEFORE the
        // generic `sk-` OpenAI rule so the prefix is consumed correctly.
        if i + 7 <= n && &bytes[i..i + 7] == b"sk-ant-" {
            let prev_word = i > 0 && is_word_byte(bytes[i - 1]);
            if !prev_word {
                let mut end = i + 7;
                while end < n && is_token_byte(bytes[end]) {
                    end += 1;
                }
                if end - (i + 7) >= 20 {
                    out.push_str("[REDACTED:anthropic-key]");
                    i = end;
                    continue;
                }
            }
        }
        // OpenAI API key: sk- + 32+ token chars. Prefix is short, so anchor
        // on a long body and require the first body byte to be alphanumeric
        // (rejects prose like "the sk- prefix" where space follows).
        if i + 3 <= n && &bytes[i..i + 3] == b"sk-" {
            let prev_word = i > 0 && is_word_byte(bytes[i - 1]);
            let first_ok = i + 3 < n && bytes[i + 3].is_ascii_alphanumeric();
            if !prev_word && first_ok {
                let mut end = i + 3;
                while end < n && is_token_byte(bytes[end]) {
                    end += 1;
                }
                if end - (i + 3) >= 32 {
                    out.push_str("[REDACTED:openai-key]");
                    i = end;
                    continue;
                }
            }
        }
        // Stripe live / restricted / webhook secrets. `pk_live_` and
        // `sk_test_` are deliberately excluded — Stripe documents them as
        // public/non-production respectively.
        for (prefix, plen) in &[
            (&b"sk_live_"[..], 8usize),
            (&b"rk_live_"[..], 8usize),
            (&b"whsec_"[..], 6usize),
        ] {
            if i + *plen <= n && &bytes[i..i + *plen] == *prefix {
                let prev_word = i > 0 && is_word_byte(bytes[i - 1]);
                if !prev_word {
                    let mut end = i + *plen;
                    while end < n && is_token_byte(bytes[end]) {
                        end += 1;
                    }
                    if end - (i + *plen) >= 24 {
                        out.push_str("[REDACTED:stripe-key]");
                        i = end;
                        break;
                    }
                }
            }
        }
        // The Stripe block above may have advanced `i` past the current
        // position; re-check the loop guard before falling through.
        if i >= n {
            break;
        }
        // JWT: eyJ + base64url + '.' + base64url + '.' + base64url.
        // Each segment must be 4+ chars to reduce false positives on bare
        // base64-ish runs that happen to start with `eyJ`.
        if i + 3 <= n && &bytes[i..i + 3] == b"eyJ" {
            let prev_word = i > 0 && is_word_byte(bytes[i - 1]);
            if !prev_word {
                let mut p = i;
                let mut ok = true;
                for seg_idx in 0..3 {
                    let start = p;
                    while p < n && is_base64url_byte(bytes[p]) {
                        p += 1;
                    }
                    if p - start < 4 {
                        ok = false;
                        break;
                    }
                    if seg_idx < 2 {
                        if p < n && bytes[p] == b'.' {
                            p += 1;
                        } else {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    out.push_str("[REDACTED:jwt]");
                    i = p;
                    continue;
                }
            }
        }
        // Default: copy the next char (one UTF-8 codepoint at a time).
        let ch = match s[i..].chars().next() {
            Some(c) => c,
            None => break,
        };
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
        || b == b'_'
        || b == b'-'
        || b == b'.'
        || b == b'/'
        || b == b'+'
        || b == b'='
}

fn is_base64url_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── sanitize_for_terminal ──────────────────────────────────────────

    #[test]
    fn sanitize_strips_c0_controls_keeps_lf_tab() {
        let input = "hello\x00world\x07!\n\tend";
        let out = sanitize_for_terminal(input);
        assert_eq!(out, "helloworld!\n\tend");
    }

    #[test]
    fn sanitize_strips_c1_controls() {
        // U+0085 NEL (next-line) is C1
        let input = "a\u{0085}b\u{0090}c";
        let out = sanitize_for_terminal(input);
        assert_eq!(out, "abc");
    }

    #[test]
    fn sanitize_strips_bidi_overrides() {
        let input = "left\u{202E}rigth\u{202C} normal\u{2066}iso\u{2069}";
        let out = sanitize_for_terminal(input);
        // U+202C PDF is in the strip range; only printable text remains.
        assert_eq!(out, "leftrigth normaliso");
    }

    #[test]
    fn sanitize_replaces_cr_with_lf() {
        let out = sanitize_for_terminal("first\r\nsecond\rthird");
        // \r\n becomes \n\n, lone \r becomes \n.
        assert_eq!(out, "first\n\nsecond\nthird");
    }

    #[test]
    fn sanitize_strips_del() {
        let out = sanitize_for_terminal("foo\x7Fbar");
        assert_eq!(out, "foobar");
    }

    #[test]
    fn sanitize_passes_through_normal_text() {
        let s = "héllo wörld 你好";
        assert_eq!(sanitize_for_terminal(s), s);
    }

    // ── redact_secrets ─────────────────────────────────────────────────

    #[test]
    fn redact_aws_access_key() {
        let s = "key: AKIAIOSFODNN7EXAMPLE end";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:aws-access-key]"), "got: {out}");
        assert!(!out.contains("AKIAIOSFODNN7EXAMPLE"));
    }

    #[test]
    fn redact_github_token() {
        let s = "ghp_aBcD1234567890aBcD1234567890XXXX is a token";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:github-token]"), "got: {out}");
        assert!(!out.contains("ghp_aBcD1234567890"));
    }

    #[test]
    fn redact_bearer_token() {
        let s = "Authorization: Bearer abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:bearer-token]"), "got: {out}");
        assert!(!out.contains("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP"));
    }

    #[test]
    fn redact_private_key_literal() {
        let s = "field private_key here, and a private-key there";
        let out = redact_secrets(s);
        assert_eq!(
            out.matches("[REDACTED:private-key]").count(),
            2,
            "got: {out}"
        );
    }

    #[test]
    fn redact_negative_normal_words() {
        // Ordinary prose must not be redacted.
        let s = "The quick brown fox jumps over the lazy dog with words.";
        let out = redact_secrets(s);
        assert_eq!(out, s);
    }

    #[test]
    fn redact_negative_artifact_paths_unchanged() {
        // Regression: file paths in envelope ARTIFACTS sections must not
        // be flagged. Before 2026-05-05 these tripped the high-entropy
        // heuristic and were rewritten as `[REDACTED:high-entropy-token]`,
        // breaking sub-agent envelope rendering.
        let s = "ARTIFACTS:\n- artifacts/findings/architecture-maturity.md\n\
                 - artifacts/weave-2026-05-05/orlog.md\n\
                 - artifacts/findings/learning-first-orchestration.md";
        let out = redact_secrets(s);
        assert_eq!(out, s);
    }

    #[test]
    fn redact_negative_uuid_unchanged() {
        // Regression: bare UUIDs (e.g. session ids in low-authority headers)
        // must not be flagged.
        let s = "session=473b6c83-cf41-4600-ba6e-4a36595097a5";
        let out = redact_secrets(s);
        assert_eq!(out, s);
    }

    // ── SaaS API key prefixes (added post-P11 triage) ──────────────────

    #[test]
    fn redact_anthropic_key() {
        let s = "ANTHROPIC_API_KEY=sk-ant-api03-AbCdEf012345_GhIjKl-MnOpQr=. end";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:anthropic-key]"), "got: {out}");
        assert!(!out.contains("sk-ant-api03-AbCdEf012345"));
    }

    #[test]
    fn redact_anthropic_key_admin_variant() {
        // Broader prefix anchor must cover future versioned shapes.
        let s = "key=sk-ant-admin-XyZ012345678901234567890";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:anthropic-key]"), "got: {out}");
    }

    #[test]
    fn redact_openai_key_project() {
        let s = "OPENAI_API_KEY=sk-proj-AbCdEf0123456789AbCdEf0123456789Xyz tail";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:openai-key]"), "got: {out}");
        assert!(!out.contains("sk-proj-AbCdEf"));
    }

    #[test]
    fn redact_openai_key_legacy() {
        let s = "key=sk-AbCdEf0123456789AbCdEf0123456789XyZ";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:openai-key]"), "got: {out}");
    }

    #[test]
    fn redact_negative_short_sk_prefix_unchanged() {
        // Short `sk-` runs in prose must NOT be redacted.
        let s = "the sk- prefix and sk-short-thing are not keys";
        let out = redact_secrets(s);
        assert_eq!(out, s);
    }

    #[test]
    fn redact_stripe_live_secret() {
        let s = "STRIPE_KEY=sk_live_AbCdEf0123456789AbCdEf01 end";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:stripe-key]"), "got: {out}");
        assert!(!out.contains("sk_live_AbCdEf"));
    }

    #[test]
    fn redact_stripe_restricted_and_webhook() {
        let s = "rk=rk_live_AbCdEf0123456789AbCdEf01 wh=whsec_AbCdEf0123456789AbCdEf01";
        let out = redact_secrets(s);
        assert_eq!(
            out.matches("[REDACTED:stripe-key]").count(),
            2,
            "got: {out}"
        );
    }

    #[test]
    fn redact_negative_stripe_publishable_and_test_unchanged() {
        // pk_live_ is publishable (public); sk_test_ is non-production. Both
        // must pass through untouched.
        let pk = "pk_live_AbCdEf0123456789AbCdEf01";
        let sk_test = "sk_test_AbCdEf0123456789AbCdEf01";
        let s = format!("pub={pk} test={sk_test}");
        let out = redact_secrets(&s);
        assert_eq!(out, s);
    }

    #[test]
    fn redact_jwt() {
        // Header.payload.signature, each base64url, 4+ chars per segment.
        let s = "auth=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.SflKxwRJSMeKKF2QT4f end";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:jwt]"), "got: {out}");
        assert!(!out.contains("eyJhbGciOiJIUzI1NiJ9"));
    }

    #[test]
    fn redact_negative_long_random_string_unchanged() {
        // Entropy fallback is intentionally OFF — long random runs without a
        // known prefix must pass through.
        let s = "blob=0123456789abcdef0123456789abcdef0123456789abcdef";
        let out = redact_secrets(s);
        assert_eq!(out, s);
    }

    #[test]
    fn redact_existing_patterns_still_match_regression() {
        // Regression net: the new SaaS rules must not break the existing
        // four redaction classes.
        let s = "AKIAIOSFODNN7EXAMPLE ghp_aBcD1234567890aBcD1234567890XXXX \
                 Bearer abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP private_key";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:aws-access-key]"), "got: {out}");
        assert!(out.contains("[REDACTED:github-token]"), "got: {out}");
        assert!(out.contains("[REDACTED:bearer-token]"), "got: {out}");
        assert!(out.contains("[REDACTED:private-key]"), "got: {out}");
    }
}
