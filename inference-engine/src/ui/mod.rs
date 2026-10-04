//! Built-in web UI (`--ui`): model list and inference playground.
//!
//! Plain HTML/CSS/JS embedded into the binary — no Node toolchain, no extra
//! port or container. Served from `/ui`.

use axum::http::{header, HeaderValue};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::Router;
use std::sync::Arc;

use crate::serving::ServeContext;

const INDEX_HTML: &str = include_str!("index.html");
const APP_JS: &str = include_str!("app.js");
const STYLE_CSS: &str = include_str!("style.css");

/// Scripts and styles only from this origin; the UI only talks to its own
/// server, and cannot be framed.
const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; \
                   connect-src 'self'; img-src 'self' data:; base-uri 'none'; \
                   form-action 'none'; frame-ancestors 'none'";

fn asset(content_type: &'static str, body: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, HeaderValue::from_static(content_type)),
            (
                header::CONTENT_SECURITY_POLICY,
                HeaderValue::from_static(CSP),
            ),
            (
                header::X_CONTENT_TYPE_OPTIONS,
                HeaderValue::from_static("nosniff"),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
        ],
        body,
    )
        .into_response()
}

pub fn router() -> Router<Arc<ServeContext>> {
    Router::new()
        .route("/ui", get(|| async { Redirect::permanent("/ui/") }))
        .route(
            "/ui/",
            get(|| async { asset("text/html; charset=utf-8", INDEX_HTML) }),
        )
        .route(
            "/ui/app.js",
            get(|| async { asset("text/javascript; charset=utf-8", APP_JS) }),
        )
        .route(
            "/ui/style.css",
            get(|| async { asset("text/css; charset=utf-8", STYLE_CSS) }),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assets_are_embedded() {
        assert!(INDEX_HTML.contains("<title>Axon"));
        assert!(INDEX_HTML.contains("app.js"));
        assert!(!APP_JS.is_empty());
        assert!(!STYLE_CSS.is_empty());
        // No inline scripts: the CSP forbids them.
        assert!(!INDEX_HTML.contains("<script>"));
    }
}
