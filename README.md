<<<<<<< HEAD
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

## Advanced Concepts

### High-Performance I/O with Multithreading

A common bottleneck in real-time video processing is the mismatch in speed between reading frames from a camera (I/O-bound) and processing each frame with a neural network (CPU/GPU-bound). To solve this, the application implements a `FrameBuffer` class that uses a multithreaded producer-consumer pattern.

-   **Producer Thread**: A dedicated background thread continuously reads frames from the webcam as fast as possible and places them into a fixed-size queue (`queue.Queue`).
-   **Consumer Thread**: The main processing loop requests a frame from the queue.
-   **Optimized Queuing**: If the processing is slow and the queue becomes full, the producer thread does not wait. Instead, it uses a non-blocking `put_nowait()` and discards the frame if the queue is full. This ensures that the main loop always gets the most recent frame available, preventing lag and keeping the video feed truly "real-time".

## Directory Structure

```
.
├── main.py                   # Main FastAPI application, handles web server and video streaming.
├── insightface_utils.py      # Helper functions for InsightFace model loading and inference.
├── recognition_utils.py      # Functions for saving/loading embeddings and comparing faces.
├── face_blur_utils.py        # Contains the function for applying a blur effect to face regions.
├── index.html                # The single-page HTML frontend for the user interface.
├── requirements.txt          # A list of all Python dependencies for the project.
├── registered_face.npy       # Stores the facial embedding of the registered "protected" person.
└── README.md                 # This file.
```

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
=======
# realtime_face_filter
>>>>>>> dcd087827ac8a3e76ae351fcf6a27df1df6a0e6c
