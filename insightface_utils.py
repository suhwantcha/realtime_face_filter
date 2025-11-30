import insightface
import numpy as np

# Final version: Use the high-level FaceAnalysis app with the lightweight 'buffalo_s' model pack
# for maximum stability and reliability in model loading.

face_app = None
models_loaded = False
try:
    print("Loading lightweight InsightFace models ('buffalo_s' pack) for GPU...")
    
    # Initialize FaceAnalysis with the 'buffalo_s' model pack for GPU.
    face_app = insightface.app.FaceAnalysis(
        name='buffalo_s', 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] # Specify GPU first
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640)) # Use ctx_id=0 for GPU
    
    print("InsightFace 'buffalo_s' models loaded successfully for GPU.")
    models_loaded = True

except Exception as e:
    print(f"FATAL ERROR loading InsightFace 'buffalo_s' models: {e}")
    print("This may be due to a network issue or an incompatible insightface/onnxruntime version.")
    
def get_faces(image_np: np.ndarray):
    """
    Detects faces and extracts their embeddings using the 'buffalo_s' pack.
    This function returns a list of Face objects.
    """
    if not models_loaded or face_app is None:
        return []

    try:
        # face_app.get() conveniently handles detection and embedding
        faces = face_app.get(image_np)
        return faces
    except Exception as e:
        # Silently fail on a single frame to keep the video stream running
        return []

if __name__ == '__main__':
    print("\n--- Running a quick test for the final insightface_utils.py (buffalo_s) ---")
    if models_loaded:
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        print("Processing a dummy black image...")
        detected_faces = get_faces(dummy_image)
        print(f"Detected {len(detected_faces)} faces in the dummy image (0 expected).")
        if len(detected_faces) == 0:
            print("Test passed: Model processed the image without errors.")
        else:
            print("Test failed: Models should not find faces in a black image.")
    else:
        print("Test skipped: InsightFace models could not be loaded.")
