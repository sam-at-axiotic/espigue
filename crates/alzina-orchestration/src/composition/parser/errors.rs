//! Parser error taxonomy.
//!
//! Three categories per `docs/composition-grammar.md` §6.1:
//! - **A**: Unknown tag or attribute (transport-level rejection from §1.5
//!   strict list + unknown operator / attribute names).
//! - **B**: Shape violation (missing required attribute, wrong child count,
//!   plan exceeds 64 KiB cap).
//! - **C**: Semantic violation (cross-Parallel-branch reference, duplicate
//!   node id, unknown id reference, unknown channel name, reserved-channel
//!   used outside Loop/Gate body).
//!
//! JSON wire contract per §6.2: every error serialises as
//! `{category, code, line, column, message, hint}`. `code` is the
//! SCREAMING_SNAKE_CASE form of `ParseErrorCode` — public contract Vefr
//! learns; renaming requires the §8 doc round-trip update.

use serde::{Deserialize, Serialize};

/// Error category per §6.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    A,
    B,
    C,
}

/// Stable wire-format parser error codes.
///
/// Adding new codes is additive. Renaming or removing a code is a
/// breaking change to the §6 contract and requires coordinated update
/// across `docs/composition-grammar.md`, this enum, the
/// `build_dispatch_tools` tool description in `chat.rs`, and
/// `config/agents/vefr/narrative.md` (the §8 doc-round-trip rule).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ParseErrorCode {
    // Category A — unknown tag/attribute, transport-level rejection
    UnknownTag,
    UnknownAttribute,
    DtdRejected,
    ExternalEntityRejected,
    ProcessingInstructionRejected,
    NamespaceRejected,
    MalformedXml,

    // Category B — shape violation
    MissingRequiredAttribute,
    WrongChildCount,
    ChildOrderViolation,
    PlanTooLarge,
    AttributeValueInvalid,

    // Category C — semantic violation
    RefNonAncestor,
    RefUnknownId,
    RefUnknownChannel,
    DuplicateNodeId,
    GateFeedbackOutsideRetry,
    ReservedChannelOutsideLoop,
}

impl ParseErrorCode {
    /// Category derivation. Tied to the §6.1 taxonomy; do not collapse
    /// without updating the doc round-trip.
    pub fn category(&self) -> ErrorCategory {
        use ParseErrorCode::*;
        match self {
            UnknownTag
            | UnknownAttribute
            | DtdRejected
            | ExternalEntityRejected
            | ProcessingInstructionRejected
            | NamespaceRejected
            | MalformedXml => ErrorCategory::A,

            MissingRequiredAttribute
            | WrongChildCount
            | ChildOrderViolation
            | PlanTooLarge
            | AttributeValueInvalid => ErrorCategory::B,

            RefNonAncestor
            | RefUnknownId
            | RefUnknownChannel
            | DuplicateNodeId
            | GateFeedbackOutsideRetry
            | ReservedChannelOutsideLoop => ErrorCategory::C,
        }
    }
}

/// A single parser error.
///
/// Wire shape exactly matches `docs/composition-grammar.md` §6.2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseError {
    pub category: ErrorCategory,
    pub code: ParseErrorCode,
    /// 1-indexed line of the offending token (NOT the token after it —
    /// the tokenizer captures `buffer_position()` BEFORE `read_event()`
    /// per `tokenizer.rs` Pitfall-2 invariant).
    pub line: u32,
    /// 1-indexed column.
    pub column: u32,
    pub message: String,
    pub hint: String,
}

/// Collection of parser errors. Multiple errors MAY be returned per §6.2
/// when the parser can recover and continue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseErrors {
    pub errors: Vec<ParseError>,
}

impl ParseErrors {
    pub fn single(err: ParseError) -> Self {
        Self { errors: vec![err] }
    }

    /// Serialise to the `{ok: false, errors: [...]}` envelope per §6.2.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "ok": false,
            "errors": self.errors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_a_codes_categorize_as_a() {
        for code in [
            ParseErrorCode::UnknownTag,
            ParseErrorCode::UnknownAttribute,
            ParseErrorCode::DtdRejected,
            ParseErrorCode::ExternalEntityRejected,
            ParseErrorCode::ProcessingInstructionRejected,
            ParseErrorCode::NamespaceRejected,
            ParseErrorCode::MalformedXml,
        ] {
            assert_eq!(
                code.category(),
                ErrorCategory::A,
                "code {:?} should be A",
                code
            );
        }
    }

    #[test]
    fn category_b_codes_categorize_as_b() {
        for code in [
            ParseErrorCode::MissingRequiredAttribute,
            ParseErrorCode::WrongChildCount,
            ParseErrorCode::ChildOrderViolation,
            ParseErrorCode::PlanTooLarge,
            ParseErrorCode::AttributeValueInvalid,
        ] {
            assert_eq!(code.category(), ErrorCategory::B);
        }
    }

    #[test]
    fn category_c_codes_categorize_as_c() {
        for code in [
            ParseErrorCode::RefNonAncestor,
            ParseErrorCode::RefUnknownId,
            ParseErrorCode::RefUnknownChannel,
            ParseErrorCode::DuplicateNodeId,
            ParseErrorCode::GateFeedbackOutsideRetry,
            ParseErrorCode::ReservedChannelOutsideLoop,
        ] {
            assert_eq!(code.category(), ErrorCategory::C);
        }
    }

    #[test]
    fn ref_non_ancestor_serialises_screaming_snake_case() {
        let json = serde_json::to_string(&ParseErrorCode::RefNonAncestor).unwrap();
        assert_eq!(json, "\"REF_NON_ANCESTOR\"");
    }

    #[test]
    fn unknown_tag_serialises_screaming_snake_case() {
        let json = serde_json::to_string(&ParseErrorCode::UnknownTag).unwrap();
        assert_eq!(json, "\"UNKNOWN_TAG\"");
    }

    #[test]
    fn parse_error_json_shape_matches_doc_section_6_2() {
        let err = ParseError {
            category: ErrorCategory::C,
            code: ParseErrorCode::RefNonAncestor,
            line: 6,
            column: 25,
            message: "Reference {past:envelope} in <Spawn agent=\"future\"> at line 6 is invalid."
                .into(),
            hint: "\"past\" is a sibling of \"future\" under <Parallel> at line 4; ...".into(),
        };
        let json = serde_json::to_value(&err).unwrap();
        assert!(json.get("category").is_some());
        assert_eq!(json["code"].as_str().unwrap(), "REF_NON_ANCESTOR");
        assert_eq!(json["line"].as_u64().unwrap(), 6);
        assert_eq!(json["column"].as_u64().unwrap(), 25);
        assert!(json.get("message").is_some());
        assert!(json.get("hint").is_some());
    }

    #[test]
    fn parse_errors_envelope_wraps_with_ok_false() {
        let errs = ParseErrors::single(ParseError {
            category: ErrorCategory::A,
            code: ParseErrorCode::UnknownTag,
            line: 1,
            column: 1,
            message: "x".into(),
            hint: "y".into(),
        });
        let env = errs.to_json();
        assert_eq!(env["ok"].as_bool().unwrap(), false);
        assert!(env["errors"].is_array());
        assert_eq!(env["errors"].as_array().unwrap().len(), 1);
    }
}
