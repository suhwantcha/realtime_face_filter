import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import StreamingResponse, FileResponse
import io
import threading
import queue
import time

# Import our custom utility functions
from insightface_utils import get_faces
from face_blur_utils import apply_blur_to_boxes
from recognition_utils import save_embedding, load_registered_embedding, is_recognized_face

# Import DeepSort
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Thread-Safe Video Capture Class (Corrected) ---
class FrameBuffer:
    def __init__(self, source, maxsize=64):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int): # Apply properties only for webcams
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.q = queue.Queue(maxsize=maxsize)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def _reader(self):
        """The loop that reads frames from the source and puts them in the queue."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            
            try:
                # 큐가 꽉 찼으면 이전 프레임을 버리고 최신 프레임을 넣습니다. (성능 우선)
                self.q.put_nowait(frame) 
            except queue.Full:
                # 큐가 꽉 찼을 때만 건너뛰고 계속 읽기
                pass
        
        self.cap.release()

    def start(self):
        self.thread.start()
        return self

    def read(self):
        """Reads a frame from the buffer."""
        try:
            # Wait up to 1 second for a new frame to be available.
            return self.q.get(timeout=1)
        except queue.Empty:
            # If the queue is empty for 1 second, it might mean the reader thread has stopped.
            return None

    def stop(self):
        """Signals the reader thread to stop and waits for it to finish."""
        self.stopped = True
        # Wait for the reader thread to finish, with a timeout.
        self.thread.join(timeout=2)

# --- App Initialization ---
app = FastAPI(title="Multithreaded CPU Face Blurring System (vFinal-fixed)")

def iou(boxA, boxB):
    # ... (IoU calculation remains the same)
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

@app.post("/register")
async def register_face(file: UploadFile = File(...)):
    # ... (Registration logic remains the same)
    try:
        contents = await file.read()
        image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return Response(content="Error: Could not read uploaded image.", status_code=400)
    faces = get_faces(image_np)
    if not faces or len(faces) != 1:
        return Response(content=f"Error: Found {len(faces) if faces else 0} faces. Please upload an image with exactly one clear face.", status_code=400)
    embedding = faces[0].embedding
    save_embedding(embedding)
    return {"message": f"Face registered successfully for {file.filename}."}

async def video_stream_generator(source):
    registered_embedding = load_registered_embedding()
    if registered_embedding is not None: print("Registered face loaded for comparison.")
    else: print("WARNING: No registered face found. All detected faces will be blurred.")

    deepsort_tracker = DeepSort(max_age=30, n_init=3)
    ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    
    frame_buffer = FrameBuffer(source).start()
    track_identities = {}

    print(f"Starting multithreaded video stream from source: {source}")
    try:
        while True:
            frame = frame_buffer.read()
            if frame is None:
                print("DEBUG: Frame buffer timed out or stream ended. Exiting loop.")
                break
            
            # --- The rest of the processing loop is the same ---
            boxes_to_blur = []
            faces = get_faces(frame)
            
            detections_for_deepsort = []
            if len(faces) > 0:
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    score = face.det_score if hasattr(face, 'det_score') else 1.0
                    w, h = x2 - x1, y2 - y1
                    
                    # ⚠️ 임베딩을 DeepSORT에 제공하여 추적 안정성을 높입니다.
                    embedding = face.embedding 
                    
                    detections_for_deepsort.append(([x1, y1, w, h], score, 'face', embedding))
            
            tracks = deepsort_tracker.update_tracks(detections_for_deepsort, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                track_id = track.track_id
                track_box_tlbr = track.to_tlbr()
                
                # 🌟 최종 수정 로직: 트랙 ID가 처음 등장했을 때만 인식 수행
                if track_id not in track_identities:
                    # 1. 트랙 박스에 가장 가까운 얼굴을 다시 탐지 결과(faces)에서 찾습니다.
                    best_iou, matched_face_embedding = 0, None
                    for face in faces:
                        current_iou = iou(track_box_tlbr, face.bbox)
                        if current_iou > best_iou:
                            best_iou = current_iou
                            matched_face_embedding = face.embedding
                    
                    # 2. IoU가 충분히 높을 때만 인식하여 ID 부여
                    if matched_face_embedding is not None and registered_embedding is not None and best_iou > 0.5:
                        is_match = is_recognized_face(matched_face_embedding, registered_embedding)
                        track_identities[track_id] = 'recognized' if is_match else 'unrecognized'
                    else:
                        track_identities[track_id] = 'unrecognized' # 기본적으로 블러 처리
                
                identity = track_identities.get(track_id, 'unrecognized')
                if identity == 'unrecognized':
                    boxes_to_blur.append(np.array(track_box_tlbr))

            processed_frame = apply_blur_to_boxes(frame, boxes_to_blur)
            (flag, encoded_image) = cv2.imencode(".jpg", processed_frame, ENCODE_PARAM)
            if not flag: continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
    finally:
        print("Video stream stopping. Cleaning up resources.")
        frame_buffer.stop()

@app.get("/video_feed_webcam")
def video_feed_webcam():
    return StreamingResponse(video_stream_generator(0), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse('index.html')
