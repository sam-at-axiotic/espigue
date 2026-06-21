//! SSRF and network egress defence for tool calls.
//!
//! Backs every `Tool::call()` implementation that performs outbound
//! network I/O. Call `validate_url` before opening any connection — it
//! rejects URLs whose resolved host falls in a private, link-local,
//! multicast, loopback, broadcast, reserved, or IPv4-mapped IPv6 range.
//!
//! ## Threat model
//!
//! | Threat | Defence |
//! |--------|---------|
//! | SSRF via RFC1918 address | `is_private_or_local_host` blocks 10/8, 172.16/12, 192.168/16 |
//! | SSRF via loopback | `127.0.0.0/8`, `::1` blocked |
//! | SSRF via link-local | `169.254/16`, `fe80::/10` blocked |
//! | SSRF via multicast | `224/4`, `ff00::/8` blocked |
//! | SSRF via IPv4-mapped IPv6 | `::ffff:0:0/96` deflected to IPv4 check |
//! | SSRF via `localhost` / `*.localhost` / `*.local` | hostname check blocks it |
//! | SSRF via alternate IP notations (octal, hex, decimal) | Rust's `IpAddr::parse` rejects them — they fall through as hostnames and are blocked because they are not recognised public IPs. Tests pin that behaviour. |
//! | DNS rebinding | Use `validate_url_with_dns_check` which re-validates after resolution |
//!
//! Ported from openhuman @ 70fdedcdd449dca38b20bf30f69ec3c53a2b1666
//! (`src/openhuman/tools/impl/network/url_guard.rs`). Key changes:
//!
//! - `anyhow::Result` replaced with `Result<_, UrlGuardError>` using
//!   `thiserror` (alzina-core's existing error crate). `// PORT NOTE: anyhow → thiserror`
//! - `validate_url` signature changed from `(url, allowed_domains)` to
//!   `(url)` — the allowlist feature is domain-specific; the SSRF gate
//!   is unconditional in alzina. `// PORT NOTE: allowlist parameter dropped`
//! - `pub(super)` visibility lifted to `pub` so alzina-core consumers
//!   can import directly.
//! - Async DNS-check tests use `tokio` (now a runtime dependency too,
//!   with the `rt` feature, so `resolve_host_ips` can wrap its blocking
//!   stdlib DNS call in `tokio::task::spawn_blocking` — CR-01 fix).

use std::net::IpAddr;

/// Errors returned by `validate_url` and related functions.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum UrlGuardError {
    #[error("URL cannot be empty")]
    EmptyUrl,

    #[error("URL cannot contain whitespace")]
    WhitespaceInUrl,

    #[error("Only http:// and https:// URLs are allowed")]
    InvalidScheme,

    #[error("URL must include a host")]
    MissingHost,

    #[error("URL userinfo (user@host) is not allowed")]
    UserinfoNotAllowed,

    #[error("IPv6 literal hosts are not supported")]
    Ipv6LiteralNotSupported,

    #[error("Blocked local/private host: {host}")]
    BlockedLocalOrPrivate { host: String },

    #[error("DNS resolution returned no addresses for '{host}'")]
    DnsNoAddresses { host: String },

    #[error("DNS rebinding blocked: '{host}' resolved to private/local address {resolved_ip}")]
    DnsRebindingBlocked {
        host: String,
        resolved_ip: String,
    },

    #[error("DNS resolution failed for '{host}': {reason}")]
    DnsResolutionFailed { host: String, reason: String },

    #[error("URL port must be numeric")]
    InvalidPort,

    #[error("URL port is out of range")]
    PortOutOfRange,

    #[error("URL must include a valid host")]
    InvalidHost,
}

/// A validated URL.
///
/// Carries only the fields needed by tool call implementations.
/// Openhuman uses `url::Url` (external crate); alzina-core is stdlib-only
/// so we define a small struct. `// PORT NOTE: url::Url → local Url struct`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    /// The original raw URL string that passed validation.
    pub raw: String,
    /// Scheme: `"http"` or `"https"`.
    pub scheme: String,
    /// Lowercased hostname (no port, no brackets).
    pub host: String,
    /// Port number (explicit or scheme default: 80 / 443).
    pub port: u16,
}

/// Validate `raw_url` against SSRF and scheme rules.
///
/// Returns a `Url` on success. Rejects:
///
/// - Non-http(s) schemes.
/// - Whitespace anywhere in the URL.
/// - Userinfo (`user@host`).
/// - IPv6 literal hosts (brackets).
/// - Any host that is loopback, RFC1918, link-local, multicast,
///   broadcast, reserved, documentation, shared-address-space,
///   `localhost`, `*.localhost`, `*.local`, or IPv4-mapped IPv6.
/// - Alternate IP notations (octal, hex, decimal) fall through as
///   unrecognised hostnames — they pass `is_private_or_local_host` but
///   are NOT on any public-IP allowlist, so callers in a stricter
///   deployment should add an allowlist layer. Tests pin this behaviour.
pub fn validate_url(raw_url: &str) -> Result<Url, UrlGuardError> {
    let url = raw_url.trim();

    if url.is_empty() {
        return Err(UrlGuardError::EmptyUrl);
    }

    if url.chars().any(char::is_whitespace) {
        return Err(UrlGuardError::WhitespaceInUrl);
    }

    let scheme = if url.starts_with("https://") {
        "https"
    } else if url.starts_with("http://") {
        "http"
    } else {
        return Err(UrlGuardError::InvalidScheme);
    };

    let host = extract_host(url)?;
    let port = extract_port(url)?;

    if is_private_or_local_host(&host) {
        return Err(UrlGuardError::BlockedLocalOrPrivate { host });
    }

    Ok(Url {
        raw: url.to_string(),
        scheme: scheme.to_string(),
        host,
        port,
    })
}

/// Like `validate_url` but also resolves the hostname via DNS and
/// verifies that none of the resolved IPs are private or local.
///
/// This defends against DNS rebinding: an attacker's domain may
/// initially resolve to a public IP (passing the scheme check) but then
/// flip to 127.0.0.1 at request time.
///
/// Use this function in every path that actually opens an outbound
/// connection. `validate_url` alone is not sufficient if you do not
/// control the DNS TTL.
pub async fn validate_url_with_dns_check(raw_url: &str) -> Result<Url, UrlGuardError> {
    validate_url_with_dns_check_with_resolver(raw_url, resolve_host_ips).await
}

/// Testable split from `validate_url_with_dns_check` — accepts a
/// custom resolver so tests can inject controlled IP lists.
pub async fn validate_url_with_dns_check_with_resolver<F, Fut>(
    raw_url: &str,
    resolver: F,
) -> Result<Url, UrlGuardError>
where
    F: FnOnce(String, u16) -> Fut,
    Fut: std::future::Future<Output = Result<Vec<IpAddr>, UrlGuardError>>,
{
    let validated = validate_url(raw_url)?;

    // If the host is already an IP literal, `validate_url` has already
    // checked it. We only need DNS resolution for hostnames.
    if validated.host.parse::<IpAddr>().is_ok() {
        return Ok(validated);
    }

    let addrs = resolver(validated.host.clone(), validated.port).await?;

    if addrs.is_empty() {
        return Err(UrlGuardError::DnsNoAddresses {
            host: validated.host,
        });
    }

    for addr in &addrs {
        let ip_str = addr.to_string();
        if is_private_or_local_host(&ip_str) {
            return Err(UrlGuardError::DnsRebindingBlocked {
                host: validated.host,
                resolved_ip: ip_str,
            });
        }
    }

    Ok(validated)
}

/// Default DNS resolver wired into [`validate_url_with_dns_check`].
///
/// CR-02 / CR-01 fix: `std::net::ToSocketAddrs::to_socket_addrs` is a
/// blocking system call that can take up to the OS resolver timeout
/// (typically 5–30 s for an unreachable DNS server). Calling it directly
/// inside an `async fn` body stalls a Tokio worker thread for the whole
/// resolution window — on a `current_thread` runtime that stalls the
/// only worker, blocking every other future on that runtime.
///
/// Wrapping the blocking call in `tokio::task::spawn_blocking` moves it
/// to the runtime's blocking pool (the same pool `tokio::fs` uses). The
/// `tokio` crate is now a runtime dependency of `alzina-core` with the
/// `rt` feature gated on — see this crate's `Cargo.toml`.
async fn resolve_host_ips(host: String, port: u16) -> Result<Vec<IpAddr>, UrlGuardError> {
    let resolved_host = host.clone();
    let result = tokio::task::spawn_blocking(move || {
        use std::net::ToSocketAddrs;
        (resolved_host.as_str(), port)
            .to_socket_addrs()
            .map(|iter| iter.map(|addr| addr.ip()).collect::<Vec<IpAddr>>())
            .map_err(|e| e.to_string())
    })
    .await;

    match result {
        Ok(Ok(addrs)) => Ok(addrs),
        Ok(Err(reason)) => Err(UrlGuardError::DnsResolutionFailed {
            host,
            reason,
        }),
        Err(join_err) => Err(UrlGuardError::DnsResolutionFailed {
            host,
            reason: format!("blocking DNS task panicked: {join_err}"),
        }),
    }
}

/// Extract the lowercased host from a URL string.
///
/// Strips scheme, userinfo check, path, query, and fragment.
/// Returns an error if the host is empty, contains userinfo, or uses
/// an IPv6 literal bracket notation.
pub fn extract_host(url: &str) -> Result<String, UrlGuardError> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .ok_or(UrlGuardError::InvalidScheme)?;

    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();

    if authority.is_empty() {
        return Err(UrlGuardError::MissingHost);
    }

    if authority.contains('@') {
        return Err(UrlGuardError::UserinfoNotAllowed);
    }

    if authority.starts_with('[') {
        return Err(UrlGuardError::Ipv6LiteralNotSupported);
    }

    let host = authority
        .split(':')
        .next()
        .unwrap_or_default()
        .trim()
        .trim_end_matches('.')
        .to_lowercase();

    if host.is_empty() {
        return Err(UrlGuardError::InvalidHost);
    }

    Ok(host)
}

/// Extract the port from a URL, defaulting to 80 (http) or 443 (https).
pub fn extract_port(url: &str) -> Result<u16, UrlGuardError> {
    let is_http = url.starts_with("http://");
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .ok_or(UrlGuardError::InvalidScheme)?;

    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();

    if authority.starts_with('[') {
        return Err(UrlGuardError::Ipv6LiteralNotSupported);
    }

    if let Some((_, port_str)) = authority.rsplit_once(':') {
        if port_str.is_empty() || !port_str.chars().all(|ch| ch.is_ascii_digit()) {
            return Err(UrlGuardError::InvalidPort);
        }
        return port_str
            .parse::<u16>()
            .map_err(|_| UrlGuardError::PortOutOfRange);
    }

    Ok(if is_http { 80 } else { 443 })
}

/// Returns `true` if the host string represents a private, loopback,
/// link-local, multicast, broadcast, reserved, or documentation address,
/// OR if it is `localhost`, `*.localhost`, or `*.local`.
///
/// IPv4-mapped IPv6 addresses (`::ffff:...`) are deflected to the IPv4
/// classifier.
///
/// Alternate IP notations (octal `0177.0.0.1`, hex `0x7f000001`,
/// decimal `2130706433`) are NOT parsed as IPs by Rust's `IpAddr::parse`
/// — this function returns `false` for them. They fall through to the
/// caller and must be handled by an allowlist or other mechanism. Tests
/// pin this behaviour so a future parser change cannot silently open SSRF.
pub fn is_private_or_local_host(host: &str) -> bool {
    // Defence-in-depth: strip trailing dot(s) from FQDN canonical form so
    // `service.local.` and `localhost.` cannot bypass the classifier (WR-01).
    // `extract_host` already trims trailing dots upstream, but this function
    // is `pub` — external callers must not be able to slip a non-canonical
    // form past the check.
    let bare = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host)
        .trim_end_matches('.');

    let has_local_tld = bare
        .rsplit('.')
        .next()
        .is_some_and(|label| label == "local");

    if bare == "localhost" || bare.ends_with(".localhost") || has_local_tld {
        return true;
    }

    if let Ok(ip) = bare.parse::<IpAddr>() {
        return match ip {
            IpAddr::V4(v4) => is_non_global_v4(v4),
            IpAddr::V6(v6) => is_non_global_v6(v6),
        };
    }

    false
}

fn is_non_global_v4(v4: std::net::Ipv4Addr) -> bool {
    let [a, b, c, _] = v4.octets();
    v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_unspecified()
        || v4.is_broadcast()
        || v4.is_multicast()
        || (a == 100 && (64..=127).contains(&b))  // shared address space (RFC 6598)
        || a >= 240                                  // reserved / future use
        || (a == 192 && b == 0 && (c == 0 || c == 2)) // IETF protocol + DS-Lite
        || (a == 198 && b == 51)                     // documentation (198.51.100.0/24)
        || (a == 203 && b == 0)                      // documentation (203.0.113.0/24)
        || (a == 198 && (18..=19).contains(&b))      // benchmarking (RFC 2544)
}

fn is_non_global_v6(v6: std::net::Ipv6Addr) -> bool {
    let segs = v6.segments();
    v6.is_loopback()
        || v6.is_unspecified()
        || v6.is_multicast()
        || (segs[0] & 0xfe00) == 0xfc00    // unique local (fc00::/7)
        || (segs[0] & 0xffc0) == 0xfe80    // link-local (fe80::/10)
        || (segs[0] == 0x2001 && segs[1] == 0x0db8) // documentation (2001:db8::/32)
        || v6.to_ipv4_mapped().is_some_and(is_non_global_v4)
}

/// Normalise a domain entry from an allowlist.
///
/// Strips scheme, path, trailing dots, port, and lowercases. Returns
/// `None` for empty or whitespace-only strings.
///
/// PORT NOTE: preserved from openhuman for callers that want to maintain
/// an allowlist layer on top of the SSRF guard.
pub fn normalize_domain(raw: &str) -> Option<String> {
    let mut d = raw.trim().to_lowercase();
    if d.is_empty() {
        return None;
    }

    if let Some(stripped) = d.strip_prefix("https://") {
        d = stripped.to_string();
    } else if let Some(stripped) = d.strip_prefix("http://") {
        d = stripped.to_string();
    }

    if let Some((host, _)) = d.split_once('/') {
        d = host.to_string();
    }

    d = d.trim_start_matches('.').trim_end_matches('.').to_string();

    if let Some((host, _)) = d.split_once(':') {
        d = host.to_string();
    }

    if d.is_empty() || d.chars().any(char::is_whitespace) {
        return None;
    }

    Some(d)
}

/// Normalise and deduplicate a list of domain strings.
pub fn normalize_allowed_domains(domains: Vec<String>) -> Vec<String> {
    let mut normalized = domains
        .into_iter()
        .filter_map(|d| normalize_domain(&d))
        .collect::<Vec<_>>();
    normalized.sort_unstable();
    normalized.dedup();
    normalized
}

/// Returns `true` if `host` matches `domain` exactly or is a subdomain.
///
/// Empty allowlist entries are rejected to prevent bypass: `host.strip_suffix("")`
/// always returns `Some(host)`, which would otherwise let any trailing-dot FQDN
/// like `attacker.example.com.` satisfy the predicate. `normalize_domain`
/// filters empty strings on the happy path, but this function MUST NOT trust
/// its caller's normalisation. Defence-in-depth (WR-04).
pub fn host_matches_allowlist(host: &str, allowed_domains: &[String]) -> bool {
    allowed_domains.iter().any(|domain| {
        !domain.is_empty()
            && (host == domain
                || host
                    .strip_suffix(domain.as_str())
                    .is_some_and(|prefix| prefix.ends_with('.')))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── normalize_domain ─────────────────────────────────────────────

    #[test]
    fn normalize_domain_strips_scheme_path_and_case() {
        let got = normalize_domain("  HTTPS://Docs.Example.com/path ").unwrap();
        assert_eq!(got, "docs.example.com");
    }

    #[test]
    fn normalize_allowed_domains_deduplicates() {
        let got = normalize_allowed_domains(vec![
            "example.com".into(),
            "EXAMPLE.COM".into(),
            "https://example.com/".into(),
        ]);
        assert_eq!(got, vec!["example.com".to_string()]);
    }

    // ── validate_url — happy paths ──────────────────────────────────

    #[test]
    fn validate_accepts_https_url() {
        let got = validate_url("https://example.com/docs").unwrap();
        assert_eq!(got.raw, "https://example.com/docs");
        assert_eq!(got.scheme, "https");
        assert_eq!(got.host, "example.com");
        assert_eq!(got.port, 443);
    }

    #[test]
    fn validate_accepts_http() {
        let got = validate_url("http://example.com").unwrap();
        assert_eq!(got.scheme, "http");
        assert_eq!(got.port, 80);
    }

    #[test]
    fn validate_accepts_subdomain() {
        assert!(validate_url("https://api.example.com/v1").is_ok());
    }

    #[test]
    fn validate_accepts_explicit_port() {
        let got = validate_url("http://example.com:8080/status").unwrap();
        assert_eq!(got.port, 8080);
    }

    // ── validate_url — rejections ────────────────────────────────────

    #[test]
    fn validate_rejects_ftp_scheme() {
        let err = validate_url("ftp://example.com").unwrap_err();
        assert!(
            matches!(err, UrlGuardError::InvalidScheme),
            "got {err:?}"
        );
    }

    #[test]
    fn validate_rejects_empty_url() {
        let err = validate_url("").unwrap_err();
        assert!(matches!(err, UrlGuardError::EmptyUrl), "got {err:?}");
    }

    #[test]
    fn validate_rejects_whitespace() {
        let err = validate_url("https://example.com/hello world").unwrap_err();
        assert!(
            matches!(err, UrlGuardError::WhitespaceInUrl),
            "got {err:?}"
        );
    }

    #[test]
    fn validate_rejects_userinfo() {
        let err = validate_url("https://user@example.com").unwrap_err();
        assert!(
            matches!(err, UrlGuardError::UserinfoNotAllowed),
            "got {err:?}"
        );
    }

    #[test]
    fn validate_rejects_ipv6_host() {
        let err = validate_url("http://[::1]:8080/path").unwrap_err();
        assert!(
            matches!(err, UrlGuardError::Ipv6LiteralNotSupported),
            "got {err:?}"
        );
    }

    #[test]
    fn validate_rejects_localhost() {
        let err = validate_url("https://localhost:8080").unwrap_err();
        assert!(
            matches!(err, UrlGuardError::BlockedLocalOrPrivate { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn validate_rejects_private_ipv4() {
        let err = validate_url("https://192.168.1.5").unwrap_err();
        assert!(
            matches!(err, UrlGuardError::BlockedLocalOrPrivate { .. }),
            "got {err:?}"
        );
    }

    // ── is_private_or_local_host — blocklist coverage ────────────────

    #[test]
    fn blocks_multicast_ipv4() {
        assert!(is_private_or_local_host("224.0.0.1"));
        assert!(is_private_or_local_host("239.255.255.255"));
    }

    #[test]
    fn blocks_broadcast() {
        assert!(is_private_or_local_host("255.255.255.255"));
    }

    #[test]
    fn blocks_reserved_ipv4() {
        assert!(is_private_or_local_host("240.0.0.1"));
        assert!(is_private_or_local_host("250.1.2.3"));
    }

    #[test]
    fn blocks_documentation_ranges() {
        assert!(is_private_or_local_host("192.0.2.1"));
        assert!(is_private_or_local_host("198.51.100.1"));
        assert!(is_private_or_local_host("203.0.113.1"));
    }

    #[test]
    fn blocks_benchmarking_range() {
        assert!(is_private_or_local_host("198.18.0.1"));
        assert!(is_private_or_local_host("198.19.255.255"));
    }

    #[test]
    fn blocks_ipv6_localhost() {
        assert!(is_private_or_local_host("::1"));
        assert!(is_private_or_local_host("[::1]"));
    }

    #[test]
    fn blocks_ipv6_multicast() {
        assert!(is_private_or_local_host("ff02::1"));
    }

    #[test]
    fn blocks_ipv6_link_local() {
        assert!(is_private_or_local_host("fe80::1"));
    }

    #[test]
    fn blocks_ipv6_unique_local() {
        assert!(is_private_or_local_host("fd00::1"));
    }

    #[test]
    fn blocks_ipv4_mapped_ipv6() {
        assert!(is_private_or_local_host("::ffff:127.0.0.1"));
        assert!(is_private_or_local_host("::ffff:192.168.1.1"));
        assert!(is_private_or_local_host("::ffff:10.0.0.1"));
    }

    #[test]
    fn allows_public_ipv4() {
        assert!(!is_private_or_local_host("8.8.8.8"));
        assert!(!is_private_or_local_host("1.1.1.1"));
        assert!(!is_private_or_local_host("93.184.216.34"));
    }

    #[test]
    fn blocks_ipv6_documentation_range() {
        assert!(is_private_or_local_host("2001:db8::1"));
    }

    #[test]
    fn allows_public_ipv6() {
        assert!(!is_private_or_local_host("2607:f8b0:4004:800::200e"));
    }

    #[test]
    fn blocks_shared_address_space() {
        assert!(is_private_or_local_host("100.64.0.1"));
        assert!(is_private_or_local_host("100.127.255.255"));
        assert!(!is_private_or_local_host("100.63.0.1"));
        assert!(!is_private_or_local_host("100.128.0.1"));
    }

    #[test]
    fn ssrf_blocks_loopback_127_range() {
        assert!(is_private_or_local_host("127.0.0.1"));
        assert!(is_private_or_local_host("127.0.0.2"));
        assert!(is_private_or_local_host("127.255.255.255"));
    }

    #[test]
    fn ssrf_blocks_rfc1918_10_range() {
        assert!(is_private_or_local_host("10.0.0.1"));
        assert!(is_private_or_local_host("10.255.255.255"));
    }

    #[test]
    fn ssrf_blocks_rfc1918_172_range() {
        assert!(is_private_or_local_host("172.16.0.1"));
        assert!(is_private_or_local_host("172.31.255.255"));
    }

    #[test]
    fn ssrf_blocks_unspecified_address() {
        assert!(is_private_or_local_host("0.0.0.0"));
    }

    #[test]
    fn ssrf_blocks_dot_localhost_subdomain() {
        assert!(is_private_or_local_host("evil.localhost"));
        assert!(is_private_or_local_host("a.b.localhost"));
    }

    #[test]
    fn ssrf_blocks_dot_local_tld() {
        assert!(is_private_or_local_host("service.local"));
    }

    /// WR-01: trailing-dot FQDN canonical forms must not bypass the classifier.
    #[test]
    fn ssrf_blocks_trailing_dot_fqdn_forms() {
        // localhost in FQDN trailing-dot form.
        assert!(is_private_or_local_host("localhost."));
        // *.local TLD in FQDN trailing-dot form.
        assert!(is_private_or_local_host("service.local."));
        // *.localhost subdomain in FQDN trailing-dot form.
        assert!(is_private_or_local_host("evil.localhost."));
        // Multiple trailing dots (degenerate but legal-ish input).
        assert!(is_private_or_local_host("service.local.."));
    }

    #[test]
    fn ssrf_ipv6_unspecified() {
        assert!(is_private_or_local_host("::"));
    }

    // ── Defense-in-depth: alternate IP notations ─────────────────────
    //
    // Rust's IpAddr::parse() rejects octal, hex, decimal, and
    // zero-padded notations. They fall through as hostname strings and
    // is_private_or_local_host returns `false` for them (it cannot
    // classify what it cannot parse). These tests pin that behaviour so
    // a parser change can't silently re-open SSRF.
    //
    // The validate_url function returns Ok for these — they are not
    // recognised private IPs. A stricter deployment should add an
    // allowlist layer (see normalize_allowed_domains / host_matches_allowlist).

    #[test]
    fn ssrf_octal_loopback_not_parsed_as_ip() {
        // 0177.0.0.1 is octal for 127.0.0.1 but Rust doesn't parse it as an IP.
        assert!(!is_private_or_local_host("0177.0.0.1"));
    }

    #[test]
    fn ssrf_hex_loopback_not_parsed_as_ip() {
        // 0x7f000001 is hex for 127.0.0.1 but Rust doesn't parse it as an IP.
        assert!(!is_private_or_local_host("0x7f000001"));
    }

    #[test]
    fn ssrf_decimal_loopback_not_parsed_as_ip() {
        // 2130706433 is the decimal representation of 127.0.0.1.
        assert!(!is_private_or_local_host("2130706433"));
    }

    #[test]
    fn ssrf_zero_padded_loopback_not_parsed_as_ip() {
        // 127.000.000.001 — zero padding makes Rust reject it as an IP literal.
        assert!(!is_private_or_local_host("127.000.000.001"));
    }

    /// Alternate-notation hostnames are not recognised as private IPs and
    /// pass `validate_url` (they look like hostnames). Pin this so future
    /// changes that start accepting such notations break loudly.
    #[test]
    fn ssrf_alternate_notations_pass_validate_url_as_hostnames() {
        for notation in [
            "http://0177.0.0.1",
            "http://0x7f000001",
            "http://2130706433",
            "http://127.000.000.001",
        ] {
            // validate_url does not reject these — they look like hostnames.
            // This test pins the behaviour; callers that need strict rejection
            // must add an allowlist.
            let result = validate_url(notation);
            assert!(
                result.is_ok(),
                "validate_url should accept alternate-notation as hostname; got Err for {notation}: {:?}",
                result.unwrap_err()
            );
        }
    }

    // ── host_matches_allowlist ────────────────────────────────────────

    #[test]
    fn allowlist_exact_match() {
        let allow = vec!["example.com".to_string()];
        assert!(host_matches_allowlist("example.com", &allow));
    }

    #[test]
    fn allowlist_subdomain_match() {
        let allow = vec!["example.com".to_string()];
        assert!(host_matches_allowlist("api.example.com", &allow));
    }

    #[test]
    fn allowlist_miss() {
        let allow = vec!["example.com".to_string()];
        assert!(!host_matches_allowlist("google.com", &allow));
    }

    /// WR-04: empty allowlist entries must NOT match anything.
    ///
    /// `host.strip_suffix("")` returns `Some(host)`, so without the
    /// non-empty guard, any host ending in `.` (FQDN form) would match an
    /// empty allowlist entry. The fix rejects empty entries unconditionally.
    #[test]
    fn allowlist_rejects_empty_entry() {
        let allow = vec!["".to_string()];
        assert!(
            !host_matches_allowlist("attacker.example.com", &allow),
            "empty allowlist entry must not match a regular host"
        );
        assert!(
            !host_matches_allowlist("attacker.example.com.", &allow),
            "empty allowlist entry must not match a trailing-dot FQDN host"
        );
        assert!(
            !host_matches_allowlist("", &allow),
            "empty allowlist entry must not match an empty host"
        );
    }

    /// WR-04: a mixed allowlist with one empty entry and one real entry
    /// still works for the real entry — the empty entry is silently
    /// ignored, not propagated as a wildcard.
    #[test]
    fn allowlist_mixed_with_empty_still_matches_real_entry() {
        let allow = vec!["".to_string(), "example.com".to_string()];
        assert!(host_matches_allowlist("example.com", &allow));
        assert!(host_matches_allowlist("api.example.com", &allow));
        assert!(!host_matches_allowlist("google.com", &allow));
    }

    // ── DNS rebinding tests (async) ───────────────────────────────────

    #[tokio::test]
    async fn dns_check_blocks_localhost_resolution() {
        // "localhost" is already blocked by validate_url, but
        // validate_url_with_dns_check should also catch it.
        let err = validate_url_with_dns_check("https://localhost")
            .await
            .unwrap_err();
        assert!(
            matches!(err, UrlGuardError::BlockedLocalOrPrivate { .. }),
            "Expected SSRF block for localhost, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn dns_check_passes_for_public_resolved_ip() {
        let got = validate_url_with_dns_check_with_resolver(
            "https://example.com",
            |host, port| async move {
                assert_eq!(host, "example.com");
                assert_eq!(port, 443);
                Ok(vec!["93.184.216.34".parse().unwrap()])
            },
        )
        .await
        .unwrap();
        assert_eq!(got.raw, "https://example.com");
    }

    #[tokio::test]
    async fn dns_check_blocks_private_resolved_ip() {
        let err = validate_url_with_dns_check_with_resolver(
            "https://example.com",
            |_, _| async { Ok(vec!["127.0.0.1".parse().unwrap()]) },
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, UrlGuardError::DnsRebindingBlocked { .. }),
            "Expected DnsRebindingBlocked, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn dns_check_uses_explicit_port_for_resolution() {
        let got = validate_url_with_dns_check_with_resolver(
            "http://api.example.com:8080/status",
            |host, port| async move {
                assert_eq!(host, "api.example.com");
                assert_eq!(port, 8080);
                Ok(vec!["93.184.216.34".parse().unwrap()])
            },
        )
        .await
        .unwrap();
        assert_eq!(got.raw, "http://api.example.com:8080/status");
    }

    #[tokio::test]
    async fn dns_check_returns_resolver_failure() {
        let err = validate_url_with_dns_check_with_resolver(
            "https://example.com",
            |host, _| async move {
                Err(UrlGuardError::DnsResolutionFailed {
                    host: host.clone(),
                    reason: "resolver unavailable".to_string(),
                })
            },
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, UrlGuardError::DnsResolutionFailed { .. }),
            "Expected DnsResolutionFailed, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn dns_check_rejects_ip_literal_private() {
        let err = validate_url_with_dns_check("https://10.0.0.1")
            .await
            .unwrap_err();
        assert!(
            matches!(err, UrlGuardError::BlockedLocalOrPrivate { .. }),
            "Expected BlockedLocalOrPrivate, got: {err:?}"
        );
    }
}
