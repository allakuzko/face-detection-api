use axum::{
    extract::Multipart,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use ort::{inputs, session::{builder::GraphOptimizationLevel, Session}, value::Tensor};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{error, info};

// --- Помилки ---

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        error!("Error: {}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        AppError(e.into())
    }
}

type AppResult<T> = Result<Json<T>, AppError>;

// --- Структури ---

#[derive(Serialize)]
struct Face {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    confidence: f32,
}

#[derive(Serialize)]
struct DetectResponse {
    faces: Vec<Face>,
    total_faces: usize,
    inference_ms: u128,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

// --- Стан ---

struct AppState {
    session: Mutex<Session>,
}

// --- Main ---

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let session = Session::builder()
        .expect("Failed to create session builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("Failed to set optimization level")
        .with_intra_threads(4)
        .expect("Failed to set threads")
        .commit_from_file("model/model.onnx")
        .expect("Failed to load model");

    info!("Model loaded successfully!");

    let state = Arc::new(AppState {
        session: Mutex::new(session),
    });

    let app = Router::new()
        .route("/detect", post(detect))
        .route("/health", get(health))
        .with_state(state);

    let addr = "0.0.0.0:3000";
    info!("Server running on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind");

    axum::serve(listener, app).await.expect("Server error");
}

// --- Handlers ---

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn detect(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    mut multipart: Multipart,
) -> AppResult<DetectResponse> {
    // Читаємо зображення з multipart
    let mut image_data: Option<Vec<u8>> = None;

    while let Some(field) = multipart.next_field().await
        .map_err(|e| anyhow::anyhow!("Multipart error: {}", e))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "image" {
            image_data = Some(
                field.bytes().await
                    .map_err(|e| anyhow::anyhow!("Failed to read image: {}", e))?
                    .to_vec(),
            );
        }
    }

    let image_data = image_data
        .ok_or_else(|| anyhow::anyhow!("No image field found in request"))?;

    // Декодуємо зображення
    let img = image::load_from_memory(&image_data)
        .map_err(|e| anyhow::anyhow!("Failed to decode image: {}", e))?;

    let orig_w = img.width() as f32;
    let orig_h = img.height() as f32;

    // Resize до 640x640
    let resized = img.resize_exact(640, 640, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    // Конвертуємо в тензор [1, 3, 640, 640]
    let mut data: Vec<f32> = vec![0.0; 1 * 3 * 640 * 640];
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let x = x as usize;
        let y = y as usize;
        data[0 * 640 * 640 + y * 640 + x] = pixel[0] as f32 / 255.0;
        data[1 * 640 * 640 + y * 640 + x] = pixel[1] as f32 / 255.0;
        data[2 * 640 * 640 + y * 640 + x] = pixel[2] as f32 / 255.0;
    }

    let tensor = Tensor::<f32>::from_array(([1usize, 3, 640, 640], data))
        .map_err(|e| anyhow::anyhow!("Failed to create tensor: {}", e))?;

    // Запускаємо inference
    let start = Instant::now();
    let mut session = state.session.lock()
        .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;

    let outputs = session
        .run(inputs!["images" => tensor])
        .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;

    let inference_ms = start.elapsed().as_millis();

    // Парсимо результати — output shape: [1, 5, 8400]
    let (_, output_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract output: {}", e))?;

    let confidence_threshold = 0.5;
    let num_detections = 8400;
    let mut faces = Vec::new();

    for i in 0..num_detections {
        let confidence = output_data[4 * num_detections + i];

        if confidence > confidence_threshold {
            let cx = output_data[0 * num_detections + i];
            let cy = output_data[1 * num_detections + i];
            let w  = output_data[2 * num_detections + i];
            let h  = output_data[3 * num_detections + i];

            let scale_x = orig_w / 640.0;
            let scale_y = orig_h / 640.0;

            faces.push(Face {
                x: (cx - w / 2.0) * scale_x,
                y: (cy - h / 2.0) * scale_y,
                width: w * scale_x,
                height: h * scale_y,
                confidence,
            });
        }
    }

    faces.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let total_faces = faces.len();
    info!(total_faces = %total_faces, inference_ms = %inference_ms, "Detection complete");

    Ok(Json(DetectResponse {
        faces,
        total_faces,
        inference_ms,
    }))
}
