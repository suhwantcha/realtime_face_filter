import numpy as np
import os
from typing import Optional

# --- Constants ---
REGISTERED_EMBEDDING_PATH = "registered_face.npy"
# ArcFace 계열 모델의 L2 거리는 0.8~1.2 사이가 일반적입니다.
DISTANCE_THRESHOLD = 0.9 # 0.9로 설정하여 덜 엄격하게 만듭니다. 

def save_embedding(embedding: np.ndarray, path: str = REGISTERED_EMBEDDING_PATH):
    """Saves a face embedding to a file."""
    np.save(path, embedding)
    print(f"Embedding saved to {path}")

def load_registered_embedding(path: str = REGISTERED_EMBEDDING_PATH) -> Optional[np.ndarray]:
    """
    Loads the registered face embedding from a file.

    Returns:
        Optional[np.ndarray]: The loaded embedding as a NumPy array, or None if the file doesn't exist.
    """
    if os.path.exists(path):
        return np.load(path)
    return None

def is_recognized_face(
    new_embedding: np.ndarray, 
    registered_embedding: np.ndarray, 
    threshold: float = DISTANCE_THRESHOLD
) -> bool:
    """
    Compares two embeddings using L2 (Euclidean) distance to determine if they are the same person.

    Args:
        new_embedding (np.ndarray): The embedding of the newly detected face.
        registered_embedding (np.ndarray): The embedding of the registered face.
        threshold (float): The distance threshold to consider a match.

    Returns:
        bool: True if the distance is below the threshold, False otherwise.
    """
    if registered_embedding is None:
        return False
        
    # Calculate L2 distance
    distance = np.linalg.norm(new_embedding - registered_embedding)
    
    return distance < threshold

if __name__ == "__main__":
    print("Testing recognition utilities with L2 Distance...")

    # --- Test Case 1: No registered embedding exists ---
    if os.path.exists(REGISTERED_EMBEDDING_PATH):
        os.remove(REGISTERED_EMBEDDING_PATH)
    
    loaded_emb = load_registered_embedding()
    assert loaded_emb is None, "Test Case 1 Failed: Embedding should be None if file doesn't exist."
    print("Test Case 1 Passed: Correctly handles non-existent embedding file.")

    # --- Test Case 2: Save and load embedding ---
    dummy_registered_emb = np.random.rand(1, 512) # Embeddings are now 512-dim
    save_embedding(dummy_registered_emb)
    loaded_emb = load_registered_embedding()
    assert np.array_equal(dummy_registered_emb, loaded_emb), "Test Case 2 Failed: Loaded embedding does not match saved one."
    print("Test Case 2 Passed: Save and load functions work correctly.")

    # --- Test Case 3: Distance check (match) ---
    # Create a very similar embedding by adding small noise
    noise = (np.random.rand(1, 512) - 0.5) * 0.1
    new_emb_similar = dummy_registered_emb + noise
    is_match = is_recognized_face(new_emb_similar, loaded_emb)
    assert is_match is True, "Test Case 3 Failed: Similar embeddings should be recognized."
    print("Test Case 3 Passed: Correctly recognizes a similar face.")
    
    # --- Test Case 4: Distance check (no match) ---
    # Create a different embedding that is far away
    new_emb_different = dummy_registered_emb + 2.0 
    is_match = is_recognized_face(new_emb_different, loaded_emb)
    assert is_match is False, "Test Case 4 Failed: Different embeddings should not be recognized."
    print("Test Case 4 Passed: Correctly rejects a different face.")

    # --- Clean up ---
    if os.path.exists(REGISTERED_EMBEDDING_PATH):
        os.remove(REGISTERED_EMBEDDING_PATH)
    
    print("\nAll recognition utility tests passed!")
