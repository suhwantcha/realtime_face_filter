# Real-time Face Privacy Filter

A web-based application that provides real-time privacy for faces in a webcam stream. The system detects all faces, recognizes a pre-registered "protected" face, and applies a blur filter to all other faces, ensuring their privacy.

This project is built with FastAPI, InsightFace, and DeepSORT, demonstrating a highly optimized pipeline for real-time object detection, recognition, and tracking.

## How It Works

The application's core logic is a multi-stage pipeline designed for high performance and accuracy.

### 1. Face Registration
- The user uploads a clear, frontal photo of the person whose face should *not* be blurred (the "protected" face).
- The application uses the **InsightFace** library (specifically, the `buffalo_s` model pack) to generate a 512-dimension facial embedding vector from this image.
- This embedding is saved locally as `registered_face.npy` and serves as the reference for recognizing the protected person.

### 2. Real-time Video Processing
When the webcam stream starts, each frame goes through the following process:

#### a. Face Detection and Embedding Extraction
- **InsightFace** processes the frame to detect all visible faces and extracts an embedding for each one. This step produces bounding boxes, detection scores, and facial embeddings for every face in the frame.

#### b. Object Tracking with DeepSORT
- The detection results (bounding boxes, scores, and embeddings) are fed into the **DeepSORT** tracker.
- DeepSORT is a powerful tracking algorithm that uses a combination of motion prediction (Kalman Filter) and appearance features (the facial embeddings) to assign and maintain a stable **track ID** for each individual.
- By using embeddings, DeepSORT can reliably track a person even if they are temporarily occluded or if detections momentarily fail, which is crucial for preventing flickering and ID swaps.

#### c. Recognition and Privacy Filtering
- For each stable track provided by DeepSORT, the system checks if an identity has already been assigned.
- If a track ID is new (`track_id not in track_identities`), the system performs a one-time recognition check:
    1. It finds the detected face that best matches the track's bounding box (using an IoU > 0.5 threshold for a confident match).
    2. It compares this face's embedding to the `registered_face.npy` embedding using L2 distance.
    3. If the distance is below a certain threshold, the track ID is marked as `'recognized'`; otherwise, it's marked as `'unrecognized'`.
- For every subsequent frame, the system simply looks up the track ID. If the ID is marked as `'unrecognized'`, a Gaussian blur is applied to the face's bounding box. This is highly efficient as the expensive recognition step is only performed once per person.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8+
    *   NVIDIA GPU with CUDA and cuDNN installed (for GPU-accelerated performance). The application can run on a CPU, but performance will be significantly slower.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/suhwantcha/realtime_face_filter.git
    cd realtime_face_filter
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If you have a compatible GPU, ensure you install the GPU version of ONNX Runtime:
    ```bash
    pip uninstall -y onnxruntime
    pip install onnxruntime-gpu
    ```

## How to Run

1.  **Start the FastAPI server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

2.  **Open the web interface:**
    *   Open your browser and navigate to `http://127.0.0.1:8000`.

3.  **Use the Application:**
    *   **Step 1:** Upload a photo to register the "protected" face.
    *   **Step 2:** Click "Start Webcam" to begin the real-time privacy filter.
