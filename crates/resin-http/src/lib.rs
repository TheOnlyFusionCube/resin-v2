mod anthropic;
mod metrics;
mod metrics_encoder;
mod openai;
mod protocol;
mod ui;

use std::{
    future::{Future, IntoFuture},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use tokio::sync::{mpsc, oneshot};

pub use metrics::{
    MetricsEndpoint, MetricsPublishError, decode_metrics, encode_metrics, metrics_text,
};
pub use metrics_encoder::{MetricsEncodeError, MetricsEncoder};
pub use protocol::{
    ConversationRequest, FinishReason, GeneratedTokenLogprob, GenerationEvent, GenerationJob,
    TokenLogprob,
};

pub const EVENT_QUEUE_CAPACITY: usize = 64;

#[derive(Clone, Debug)]
pub struct RouteConfig {
    pub enable_openai: bool,
    pub enable_anthropic: bool,
    pub enable_demo: bool,
    pub routed_dp_size: usize,
    /// When set, `/v1/*` requires `Authorization: Bearer <key>` or `x-api-key: <key>`.
    pub api_key: Option<Arc<str>>,
}

impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            enable_openai: true,
            enable_anthropic: true,
            enable_demo: true,
            routed_dp_size: 1,
            api_key: None,
        }
    }
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<GenerationJob>,
    model: Arc<str>,
    next_id: Arc<AtomicU64>,
    routed_dp_size: usize,
}

impl AppState {
    fn request_id(&self, prefix: &str) -> String {
        let sequence = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}_{sequence:016x}")
    }

    fn enqueue(
        &self,
        request: ConversationRequest,
    ) -> Result<mpsc::Receiver<GenerationEvent>, (StatusCode, &'static str)> {
        let (events, receiver) = mpsc::channel(EVENT_QUEUE_CAPACITY);
        match self.jobs.try_send(GenerationJob {
            request,
            events,
            enqueued_at: Instant::now(),
        }) {
            Ok(()) => Ok(receiver),
            Err(mpsc::error::TrySendError::Full(_)) => Err((
                StatusCode::TOO_MANY_REQUESTS,
                "the inference request queue is full; retry later",
            )),
            Err(mpsc::error::TrySendError::Closed(_)) => Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "the inference engine is not running",
            )),
        }
    }
}

pub fn router(
    model: impl Into<Arc<str>>,
    jobs: mpsc::Sender<GenerationJob>,
    routes: RouteConfig,
    metrics: Option<MetricsEndpoint>,
) -> Router {
    let state = AppState {
        jobs,
        model: model.into(),
        next_id: Arc::new(AtomicU64::new(1)),
        routed_dp_size: routes.routed_dp_size,
    };
    let mut api = Router::new();
    if routes.enable_openai {
        api = api
            .route("/v1/models", get(openai::models))
            .route("/v1/chat/completions", post(openai::chat_completions))
            .route("/v1/responses", post(openai::responses));
    }
    if routes.enable_anthropic {
        api = api.route("/v1/messages", post(anthropic::messages));
    }
    if let Some(key) = routes.api_key {
        api = api.route_layer(middleware::from_fn_with_state(key, require_api_key));
    }
    let mut router = Router::new()
        .route("/health", get(openai::health))
        .merge(api);
    if let Some(metrics) = metrics {
        router = router.route(
            "/metrics",
            get(move |headers: HeaderMap| {
                let metrics = metrics.clone();
                async move { metrics.response(&headers) }
            }),
        );
    }
    if routes.enable_demo {
        router = router.route("/", get(openai::index));
    }
    router
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    model: impl Into<Arc<str>>,
    jobs: mpsc::Sender<GenerationJob>,
    routes: RouteConfig,
    metrics: Option<MetricsEndpoint>,
) -> std::io::Result<()> {
    serve_with_shutdown(listener, model, jobs, routes, metrics, shutdown_signal()).await
}

/// Resolve on Ctrl-C or, on Unix, SIGTERM, the signal container runtimes send
/// before killing a process.
pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {},
                    _ = terminate.recv() => {},
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

async fn require_api_key(State(key): State<Arc<str>>, request: Request, next: Next) -> Response {
    let headers = request.headers();
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
        });
    if presented.is_some_and(|presented| constant_time_eq(presented.as_bytes(), key.as_bytes())) {
        return next.run(request).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        Json(protocol::json_error(
            "invalid or missing API key",
            "authentication_error",
        )),
    )
        .into_response()
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && std::hint::black_box(
            left.iter()
                .zip(right)
                .fold(0u8, |acc, (left, right)| acc | (left ^ right)),
        ) == 0
}

/// Stop accepting connections when requested and bound the time spent draining
/// existing HTTP connections. The hosting runtime may then be dropped even if
/// a client has stopped reading its response.
pub async fn serve_with_shutdown(
    listener: tokio::net::TcpListener,
    model: impl Into<Arc<str>>,
    jobs: mpsc::Sender<GenerationJob>,
    routes: RouteConfig,
    metrics: Option<MetricsEndpoint>,
    shutdown: impl Future<Output = ()>,
) -> std::io::Result<()> {
    let (stop, stopped) = oneshot::channel::<()>();
    let server = axum::serve(listener, router(model, jobs, routes, metrics))
        .with_graceful_shutdown(async {
            let _ = stopped.await;
        })
        .into_future();
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result,
        () = shutdown => {
            let _ = stop.send(());
            tokio::time::timeout(Duration::from_secs(5), &mut server)
                .await
                .unwrap_or(Ok(()))
        }
    }
}
