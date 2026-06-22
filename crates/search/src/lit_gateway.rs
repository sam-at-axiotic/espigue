//! Literature gateway — the single chokepoint for external literature traffic.
//!
//! Every arxiv search, ar5iv full-text fetch, and Semantic Scholar call made
//! during a TTD run goes through one `Arc<LitGateway>` shared across lanes,
//! gaps, and (future) parallel trajectories. Upstream parallelism therefore
//! cannot multiply external API pressure.
//!
//! Semantics are a precise port of clawd's lit-review rate limiting
//! (`~/clawd/skills/lit-review/scripts/semantic_scholar.py` +
//! `lib/retrieval.py::S2RateLimiter`), per operator instruction 2026-06-11:
//!
//! - **Token bucket per endpoint**: capacity 1, refill = 1/min_interval,
//!   acquire-before-call. S2 spacing: 1.1s with an API key (clawd uses 1.0s;
//!   we stay just below S2's 1-req/s cumulative threshold per its issuance
//!   email), 0.5s without (`RATE_LIMIT_DELAY`). arxiv API uses 3s (arxiv's
//!   published courtesy rate); ar5iv fetches use 1s.
//! - **Per-run budget cap**: `S2_MAX_CALLS_PER_RUN`-style hard caps. When a
//!   budget is exhausted the gateway returns `BudgetExhausted` — callers
//!   degrade to local-only retrieval LOUDLY, never error.
//! - **Backoff**: `MAX_RETRIES = 3`, `BACKOFF_BASE = 2.0`, honouring
//!   Retry-After when the caller surfaces it (clawd `_request` semantics).
//! - **Single-flight**: concurrent identical requests (same key) coalesce so
//!   two trajectories promoting the same paper trigger one fetch.
//!
//! Probe-10 evidence motivating this module: unpaced per-gap fusion drove
//! arxiv to HTTP 500 and S2 to 429 by denoise step 1, silently collapsing
//! live exploration to internal-only — the opposite of the design intent.
//!
//! clawd's disk response cache (`S2Cache`) is NOT ported here — it lands with
//! the smart_explore port where batch ID lookups make it pay. This module is
//! pacing + budgets + backoff + single-flight only.

use std::collections::HashSet;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio::time::Instant;

// ── clawd-parity constants ────────────────────────────────────────────────────

/// clawd `RATE_LIMIT_DELAY` — S2 spacing without an API key.
pub const S2_DELAY_UNKEYED: Duration = Duration::from_millis(500);
/// Keyed S2 spacing. clawd uses 1.0s (`delay = 1.0 if self.api_key else
/// RATE_LIMIT_DELAY`), but S2's key-issuance email sets the limit at 1 req/s
/// cumulative and asks callers to stay *below* the threshold. 1.0s sits exactly
/// at the line, so clock jitter on S2's side can count two calls in one window
/// → an avoidable 429. 1100ms keeps cumulative throughput provably under 1/s
/// (margin ~100ms/call), costing ~4s on a 40-call run. Deliberate divergence
/// from clawd, per operator instruction 2026-06-12.
pub const S2_DELAY_KEYED: Duration = Duration::from_millis(1100);
/// arxiv API courtesy rate (arxiv ToS: no more than 1 request per 3 seconds).
pub const ARXIV_DELAY: Duration = Duration::from_secs(3);
/// ar5iv full-text fetch spacing.
pub const AR5IV_DELAY: Duration = Duration::from_secs(1);
/// PDF full-text fetch spacing (F10). Targets are mixed publisher hosts — 1s
/// is polite and matches the ar5iv lane sizing rationale.
pub const PDF_DELAY: Duration = Duration::from_secs(1);

/// clawd `MAX_RETRIES`.
pub const MAX_RETRIES: u32 = 3;
/// clawd `BACKOFF_BASE`.
pub const BACKOFF_BASE: f64 = 2.0;

// F4 sizing (probe 11 → probe 12):
//
// Probe-11 exhausted 20/20 arxiv and 20/20 S2 ~2.5 min into the 6.6-min graph stage.
// This is a deliberate divergence from clawd's default of 20 for both endpoints.
//
//   arxiv: 40 slots × 3s spacing = 120s — fits within graph stage; leaves room for gap fills.
//          Budgets are shared across Stage 0 (smart_explore) + initial fusion + gap fills.
//   S2 keyed: 40 slots × 1.0s keyed spacing = 40s — well within graph stage budget.
//          Covers smart_explore depth-2 traversal + per-paper citation/ref calls.
//   ar5iv: 150 (raised from 30, operator sizing 2026-06-12 after probe 17). The 30
//          was sized at probe 11, before stage-0 explore, canonical S2→arxiv keying,
//          and the F10 PDF lane multiplied promotion candidates; probe 17 exhausted
//          30 mid-graph-stage and dozens of gap-fill papers degraded to abstract-only.
//
// Env overrides (ALZINA_ARXIV_MAX_CALLS_PER_RUN, ALZINA_S2_MAX_CALLS_PER_RUN,
// ALZINA_AR5IV_MAX_CALLS_PER_RUN) remain the operator lever for further tuning.
//
// Note: S2_API_KEY must be exported by the launching shell — there is NO dotenv
// loader in any crate under crates/; a key in `.env` does NOT reach the daemon.
// The key flows into S2Client via std::env::var("S2_API_KEY") in s2_enrichment.rs.
// LitGateway receives only the `s2_keyed` bool — the key itself never enters this module
// and must never be logged.
/// clawd `S2_MAX_CALLS_PER_RUN` default — raised to 40 (F4 sizing, probe-11 exhaustion).
const DEFAULT_S2_BUDGET: usize = 40;
/// arxiv default — raised to 40 (F4 sizing, probe-11 exhaustion).
const DEFAULT_ARXIV_BUDGET: usize = 40;
/// ar5iv default — raised to 150 (probe-17 exhausted 30 mid-graph-stage;
/// operator sizing 2026-06-12).
const DEFAULT_AR5IV_BUDGET: usize = 150;
/// PDF fetch default — 30 (F10; fetches fire only on s2:* promotion hits, matching ar5iv sizing).
const DEFAULT_PDF_BUDGET: usize = 30;

// ── Types ─────────────────────────────────────────────────────────────────────

/// External endpoints the gateway paces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endpoint {
    /// arxiv Atom search API.
    ArxivSearch,
    /// ar5iv HTML full-text fetch.
    Ar5ivFetch,
    /// Semantic Scholar graph API.
    S2,
    /// Open-access PDF full-text fetch (F10). Mixed publisher hosts.
    PdfFetch,
}

impl Endpoint {
    fn label(&self) -> &'static str {
        match self {
            Endpoint::ArxivSearch => "arxiv_search",
            Endpoint::Ar5ivFetch => "ar5iv_fetch",
            Endpoint::S2 => "s2",
            Endpoint::PdfFetch => "pdf_fetch",
        }
    }
}

/// Outcome of an acquire attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acquire {
    /// Token granted — proceed with the external call.
    Proceed,
    /// Per-run budget exhausted — degrade to local-only, loudly.
    BudgetExhausted,
}

/// Per-endpoint pacing + budget state (one token-bucket lane).
struct Lane {
    min_interval: Duration,
    max_calls: usize,
    state: Mutex<LaneState>,
}

struct LaneState {
    /// When the next call may fire.
    next_allowed: Instant,
    calls_made: usize,
}

impl Lane {
    fn new(min_interval: Duration, max_calls: usize) -> Self {
        Self {
            min_interval,
            max_calls,
            state: Mutex::new(LaneState {
                next_allowed: Instant::now(),
                calls_made: 0,
            }),
        }
    }

    /// Token-bucket acquire: waits out the spacing interval, enforces budget.
    ///
    /// Mirrors clawd `S2RateLimiter.acquire` (capacity 1, refill 1/interval,
    /// budget checked first — an exhausted budget never sleeps).
    async fn acquire(&self) -> Acquire {
        // Hold the lock across the wait so concurrent acquirers queue FIFO
        // and each reserves its own slot (capacity-1 bucket semantics).
        let mut s = self.state.lock().await;
        if s.calls_made >= self.max_calls {
            return Acquire::BudgetExhausted;
        }
        let now = Instant::now();
        let wait_until = s.next_allowed.max(now);
        s.next_allowed = wait_until + self.min_interval;
        s.calls_made += 1;
        drop(s);

        tokio::time::sleep_until(wait_until).await;
        Acquire::Proceed
    }

    async fn calls_made(&self) -> usize {
        self.state.lock().await.calls_made
    }
}

/// Snapshot of gateway counters for run-end logging.
#[derive(Debug, Clone)]
pub struct GatewaySnapshot {
    pub arxiv_calls: usize,
    pub arxiv_budget: usize,
    pub ar5iv_calls: usize,
    pub ar5iv_budget: usize,
    pub s2_calls: usize,
    pub s2_budget: usize,
    pub pdf_calls: usize,
    pub pdf_budget: usize,
    pub backoffs: usize,
}

// ── LitGateway ────────────────────────────────────────────────────────────────

/// One instance per run, `Arc`-shared by every caller that touches an
/// external literature endpoint.
pub struct LitGateway {
    arxiv: Lane,
    ar5iv: Lane,
    s2: Lane,
    pdf: Lane,
    /// Count of backoff sleeps taken (observability).
    backoffs: StdMutex<usize>,
    /// Single-flight: keys currently being fetched. A second caller asking
    /// for the same key gets `false` from `begin_flight` and should skip.
    in_flight: StdMutex<HashSet<String>>,
}

impl LitGateway {
    /// Build with explicit budgets. `s2_keyed` selects clawd's keyed vs
    /// unkeyed S2 spacing.
    pub fn new(
        arxiv_budget: usize,
        ar5iv_budget: usize,
        s2_budget: usize,
        s2_keyed: bool,
    ) -> Self {
        Self::new_with_pdf(arxiv_budget, ar5iv_budget, s2_budget, DEFAULT_PDF_BUDGET, s2_keyed)
    }

    /// Build with explicit budgets including the PDF lane.
    pub fn new_with_pdf(
        arxiv_budget: usize,
        ar5iv_budget: usize,
        s2_budget: usize,
        pdf_budget: usize,
        s2_keyed: bool,
    ) -> Self {
        let s2_delay = if s2_keyed { S2_DELAY_KEYED } else { S2_DELAY_UNKEYED };
        Self {
            arxiv: Lane::new(ARXIV_DELAY, arxiv_budget),
            ar5iv: Lane::new(AR5IV_DELAY, ar5iv_budget),
            s2: Lane::new(s2_delay, s2_budget),
            pdf: Lane::new(PDF_DELAY, pdf_budget),
            backoffs: StdMutex::new(0),
            in_flight: StdMutex::new(HashSet::new()),
        }
    }

    /// Build from env overrides, falling back to F4-sized defaults.
    ///
    /// - `ALZINA_ARXIV_MAX_CALLS_PER_RUN` (default 40, raised from clawd's 20 — F4 sizing)
    /// - `ALZINA_AR5IV_MAX_CALLS_PER_RUN` (default 150, raised from 30 — probe-17 sizing)
    /// - `ALZINA_S2_MAX_CALLS_PER_RUN` (default 40, raised from clawd's 20 — F4 sizing)
    /// - `ALZINA_PDF_MAX_CALLS_PER_RUN` (default 30, F10)
    ///
    /// **S2_API_KEY requirement:** the launching shell must export `S2_API_KEY`.
    /// There is NO dotenv loader in any crate; a key in `.env` is inert for
    /// the daemon process. LitGateway receives only the `s2_keyed` bool — the
    /// key itself flows through `S2Client::from_env()` in s2_enrichment.rs and
    /// must never be logged or included in fixtures.
    pub fn from_env(s2_keyed: bool) -> Self {
        fn env_usize(key: &str, default: usize) -> usize {
            std::env::var(key)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(default)
        }
        Self::new_with_pdf(
            env_usize("ALZINA_ARXIV_MAX_CALLS_PER_RUN", DEFAULT_ARXIV_BUDGET),
            env_usize("ALZINA_AR5IV_MAX_CALLS_PER_RUN", DEFAULT_AR5IV_BUDGET),
            env_usize("ALZINA_S2_MAX_CALLS_PER_RUN", DEFAULT_S2_BUDGET),
            env_usize("ALZINA_PDF_MAX_CALLS_PER_RUN", DEFAULT_PDF_BUDGET),
            s2_keyed,
        )
    }

    fn lane(&self, endpoint: Endpoint) -> &Lane {
        match endpoint {
            Endpoint::ArxivSearch => &self.arxiv,
            Endpoint::Ar5ivFetch => &self.ar5iv,
            Endpoint::S2 => &self.s2,
            Endpoint::PdfFetch => &self.pdf,
        }
    }

    /// Acquire a slot for one external call: budget check + paced wait.
    ///
    /// `BudgetExhausted` is logged loudly here (one place) so every caller's
    /// degrade path is observable without per-call-site discipline.
    pub async fn acquire(&self, endpoint: Endpoint) -> Acquire {
        let outcome = self.lane(endpoint).acquire().await;
        if outcome == Acquire::BudgetExhausted {
            tracing::warn!(
                endpoint = endpoint.label(),
                budget = self.lane(endpoint).max_calls,
                "lit_gateway: per-run budget exhausted — degrading to local-only"
            );
        }
        outcome
    }

    /// Run `op` with clawd-parity backoff: up to `MAX_RETRIES` attempts;
    /// retryable failures sleep `BACKOFF_BASE^attempt` seconds (or the
    /// server-provided Retry-After if the caller surfaces one).
    ///
    /// `op` returns `Ok(T)` or `Err(RetryAdvice)`. Non-retryable errors
    /// surface immediately.
    pub async fn with_backoff<T, F, Fut>(
        &self,
        endpoint: Endpoint,
        mut op: F,
    ) -> Result<T, String>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, RetryAdvice>>,
    {
        let mut last_err = String::new();
        for attempt in 0..MAX_RETRIES {
            match op().await {
                Ok(v) => return Ok(v),
                Err(RetryAdvice::Fatal(e)) => return Err(e),
                Err(RetryAdvice::Retry { error, retry_after }) => {
                    last_err = error;
                    let wait = retry_after.unwrap_or_else(|| {
                        Duration::from_secs_f64(BACKOFF_BASE.powi(attempt as i32))
                    });
                    {
                        let mut b = self.backoffs.lock().unwrap();
                        *b += 1;
                    }
                    tracing::warn!(
                        endpoint = endpoint.label(),
                        attempt,
                        wait_secs = wait.as_secs_f64(),
                        error = %last_err,
                        "lit_gateway: retryable failure — backing off"
                    );
                    tokio::time::sleep(wait).await;
                }
            }
        }
        Err(format!(
            "{} failed after {MAX_RETRIES} attempts: {last_err}",
            endpoint.label()
        ))
    }

    /// Single-flight begin: returns true if this caller owns the flight for
    /// `key`; false if an identical flight is already in progress (skip it —
    /// the owner's result lands in the shared store).
    pub fn begin_flight(&self, key: &str) -> bool {
        self.in_flight.lock().unwrap().insert(key.to_string())
    }

    /// Single-flight end — call when the owning fetch completes (ok or not).
    pub fn end_flight(&self, key: &str) {
        self.in_flight.lock().unwrap().remove(key);
    }

    /// Counters for run-end observability.
    pub async fn snapshot(&self) -> GatewaySnapshot {
        GatewaySnapshot {
            arxiv_calls: self.arxiv.calls_made().await,
            arxiv_budget: self.arxiv.max_calls,
            ar5iv_calls: self.ar5iv.calls_made().await,
            ar5iv_budget: self.ar5iv.max_calls,
            s2_calls: self.s2.calls_made().await,
            s2_budget: self.s2.max_calls,
            pdf_calls: self.pdf.calls_made().await,
            pdf_budget: self.pdf.max_calls,
            backoffs: *self.backoffs.lock().unwrap(),
        }
    }
}

/// Retry guidance from a gateway-wrapped operation.
pub enum RetryAdvice {
    /// Retryable (429 / 5xx / transient network). Sleep and try again;
    /// `retry_after` carries a server-provided Retry-After when present.
    Retry {
        error: String,
        retry_after: Option<Duration>,
    },
    /// Not retryable — surface immediately.
    Fatal(String),
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Budget cap: calls beyond max return BudgetExhausted, never sleep.
    #[tokio::test]
    async fn budget_exhaustion_is_loud_and_immediate() {
        let gw = LitGateway::new(2, 2, 2, false);
        // Use ar5iv lane (1s spacing) — first call free, no prior interval.
        assert_eq!(gw.acquire(Endpoint::Ar5ivFetch).await, Acquire::Proceed);
        // Second call would wait 1s — acceptable in test.
        assert_eq!(gw.acquire(Endpoint::Ar5ivFetch).await, Acquire::Proceed);
        // Third call: budget gone — immediate, no sleep.
        let start = std::time::Instant::now();
        assert_eq!(
            gw.acquire(Endpoint::Ar5ivFetch).await,
            Acquire::BudgetExhausted
        );
        assert!(
            start.elapsed() < Duration::from_millis(100),
            "budget exhaustion must not sleep"
        );
        let snap = gw.snapshot().await;
        assert_eq!(snap.ar5iv_calls, 2);
    }

    /// PdfFetch budget exhaustion mirrors the ar5iv test — immediate, no sleep.
    #[tokio::test]
    async fn pdf_fetch_budget_exhaustion_is_loud_and_immediate() {
        let gw = LitGateway::new_with_pdf(40, 30, 40, 2, false);
        // Use pdf lane (1s spacing) — first call free.
        assert_eq!(gw.acquire(Endpoint::PdfFetch).await, Acquire::Proceed);
        assert_eq!(gw.acquire(Endpoint::PdfFetch).await, Acquire::Proceed);
        // Third call: budget gone — immediate, no sleep.
        let start = std::time::Instant::now();
        assert_eq!(
            gw.acquire(Endpoint::PdfFetch).await,
            Acquire::BudgetExhausted
        );
        assert!(
            start.elapsed() < Duration::from_millis(100),
            "pdf budget exhaustion must not sleep"
        );
        let snap = gw.snapshot().await;
        assert_eq!(snap.pdf_calls, 2, "pdf_calls must be 2 after budget exhausted");
        assert_eq!(snap.pdf_budget, 2);
    }

    /// Token bucket: two paced acquires are spaced by >= min_interval.
    #[tokio::test(start_paused = true)]
    async fn acquires_are_paced_by_min_interval() {
        let gw = Arc::new(LitGateway::new(10, 10, 10, false));
        let t0 = Instant::now();
        gw.acquire(Endpoint::S2).await; // unkeyed → 500ms spacing
        gw.acquire(Endpoint::S2).await;
        gw.acquire(Endpoint::S2).await;
        // With paused time, sleeps auto-advance the clock deterministically.
        let elapsed = t0.elapsed();
        assert!(
            elapsed >= Duration::from_millis(1000),
            "third S2 call must wait 2 × 500ms spacing; elapsed {elapsed:?}"
        );
    }

    /// Concurrent acquirers share ONE budget and ONE pace lane (the
    /// parallel-trajectories requirement).
    #[tokio::test(start_paused = true)]
    async fn concurrent_acquirers_share_budget_and_pacing() {
        let gw = Arc::new(LitGateway::new(10, 10, 3, false));
        let granted = Arc::new(AtomicUsize::new(0));
        let denied = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];
        for _ in 0..6 {
            let gw = Arc::clone(&gw);
            let granted = Arc::clone(&granted);
            let denied = Arc::clone(&denied);
            handles.push(tokio::spawn(async move {
                match gw.acquire(Endpoint::S2).await {
                    Acquire::Proceed => granted.fetch_add(1, Ordering::SeqCst),
                    Acquire::BudgetExhausted => denied.fetch_add(1, Ordering::SeqCst),
                };
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(granted.load(Ordering::SeqCst), 3, "budget 3 → 3 grants");
        assert_eq!(denied.load(Ordering::SeqCst), 3, "remaining 3 denied");
    }

    /// Backoff: retryable errors retry up to MAX_RETRIES with growing waits;
    /// success on a later attempt returns Ok.
    #[tokio::test(start_paused = true)]
    async fn backoff_retries_then_succeeds() {
        let gw = LitGateway::new(10, 10, 10, false);
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_c = Arc::clone(&attempts);

        let result = gw
            .with_backoff(Endpoint::S2, move || {
                let attempts = Arc::clone(&attempts_c);
                async move {
                    let n = attempts.fetch_add(1, Ordering::SeqCst);
                    if n < 2 {
                        Err(RetryAdvice::Retry {
                            error: "HTTP 429".into(),
                            retry_after: None,
                        })
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result, Ok(42));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        let snap = gw.snapshot().await;
        assert_eq!(snap.backoffs, 2, "two backoff sleeps before success");
    }

    /// Backoff: fatal errors surface immediately without retry.
    #[tokio::test]
    async fn backoff_fatal_is_immediate() {
        let gw = LitGateway::new(10, 10, 10, false);
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_c = Arc::clone(&attempts);

        let result: Result<i32, String> = gw
            .with_backoff(Endpoint::ArxivSearch, move || {
                let attempts = Arc::clone(&attempts_c);
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err(RetryAdvice::Fatal("HTTP 400 bad query".into()))
                }
            })
            .await;

        assert_eq!(result, Err("HTTP 400 bad query".to_string()));
        assert_eq!(attempts.load(Ordering::SeqCst), 1, "no retries on fatal");
    }

    /// Single-flight: second caller for the same key is told to skip;
    /// after end_flight the key is fetchable again.
    #[test]
    fn single_flight_coalesces_identical_keys() {
        let gw = LitGateway::new(10, 10, 10, false);
        assert!(gw.begin_flight("arxiv:2105.14103"));
        assert!(
            !gw.begin_flight("arxiv:2105.14103"),
            "identical concurrent flight must coalesce"
        );
        assert!(gw.begin_flight("arxiv:9999.00001"), "different key proceeds");
        gw.end_flight("arxiv:2105.14103");
        assert!(gw.begin_flight("arxiv:2105.14103"), "released key re-fetchable");
    }

    /// Default budgets reflect F4 sizing — deliberate divergence from clawd's 20.
    ///
    /// Probe-11 exhausted 20/20 arxiv and 20/20 S2 ~2.5 min into the 6.6-min graph
    /// stage. The F4 resize raises both to 40 with documented arithmetic (see constant
    /// block doc comment). ar5iv raised 30 → 150 after probe 17 exhausted 30
    /// mid-graph-stage (operator sizing 2026-06-12). PDF default keeps the original
    /// promotion-hits sizing (F10: fetches fire only on s2:* promotion hits).
    #[test]
    fn env_defaults_are_f4_sized() {
        // No env vars set in test environment for these keys (best-effort).
        let gw = LitGateway::new_with_pdf(
            DEFAULT_ARXIV_BUDGET, DEFAULT_AR5IV_BUDGET,
            DEFAULT_S2_BUDGET, DEFAULT_PDF_BUDGET, false
        );
        // Constructed without panic; budget values exercised in other tests.
        drop(gw);
        assert_eq!(
            DEFAULT_S2_BUDGET, 40,
            "F4 divergence from clawd 20: probe-11 exhausted S2 budget at 2.5 min into 6.6-min stage"
        );
        assert_eq!(
            DEFAULT_ARXIV_BUDGET, 40,
            "F4 divergence from clawd 20: probe-11 exhausted arxiv budget at 2.5 min into 6.6-min stage"
        );
        assert_eq!(
            DEFAULT_AR5IV_BUDGET, 150,
            "probe-17 divergence from 30: stage-0 explore + canonical keying + \
             PDF lane multiplied promotion candidates; 30 exhausted mid-graph-stage"
        );
        assert_eq!(
            DEFAULT_PDF_BUDGET, 30,
            "pdf default matches ar5iv sizing — fetches fire only on promotion hits"
        );
    }
}
