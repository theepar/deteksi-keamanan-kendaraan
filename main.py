import sys
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist
import threading
import time
import winsound
import mediapipe as mp
import os

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QScrollArea
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ==============================================================================
# 1. THREAD UNTUK PROSES AI
# ==============================================================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_status_signal = pyqtSignal(list)  # list of dict: [{'user': n, 'kantuk': bool, 'ancaman': bool}]

    def run(self):
        self.running = True
        print("THREAD: Memuat model...")
        def load_model_safe(path):
            try:
                if os.path.exists(path): return YOLO(path)
                else: print(f"Model file not found: {path}"); return None
            except Exception as e: print(f"Model load error for {path}: {e}"); return None

        model_yolov5s = load_model_safe('yolov5su.pt')
        model_common_medium = load_model_safe('yolov8m.pt')
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("THREAD: Model loading completed.")

        CONFIDENCE_THRESHOLD, EAR_THRESHOLD, EAR_CONSEC_FRAMES = 0.5, 0.23, 10  # Lebih cepat deteksi kantuk
        drowsiness_counters = {}  # per user idx

        def eye_aspect_ratio_mp(eye):
            A, B, C = dist.euclidean(eye[1], eye[5]), dist.euclidean(eye[2], eye[4]), dist.euclidean(eye[0], eye[3])
            return (A + B) / (2.0 * C) if C != 0 else 0.0

        def play_sound(sound_type):
            try:
                if sound_type == "kantuk": winsound.Beep(800, 500)
                elif sound_type == "ancaman": winsound.Beep(1500, 700)
            except Exception as e: print(f"Gagal memainkan suara: {e}")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Tidak bisa membuka kamera."); self.update_status_signal.emit([{'user': 0, 'kantuk': False, 'ancaman': False, 'error': True}]); return
            
        alarm_kantuk_on, alarm_ancaman_on = False, False

        def iou(boxA, boxB):
            # box: (xmin, ymin, xmax, ymax)
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
            return iou

        def merge_person_boxes(detections, iou_threshold=0.5):
            # detections: [{'box': (xmin, ymin, xmax, ymax), 'label': label}]
            person_boxes = [d['box'] for d in detections if d['label'] == 'PERSON']
            merged = []
            used = [False] * len(person_boxes)
            for i in range(len(person_boxes)):
                if used[i]: continue
                boxA = person_boxes[i]
                group = [boxA]
                used[i] = True
                for j in range(i+1, len(person_boxes)):
                    if used[j]: continue
                    boxB = person_boxes[j]
                    if iou(boxA, boxB) > iou_threshold:
                        group.append(boxB)
                        used[j] = True
                # Gabungkan group jadi satu box (ambil min/max koordinat)
                xs = [b[0] for b in group] + [b[2] for b in group]
                ys = [b[1] for b in group] + [b[3] for b in group]
                merged.append({'box': (min(xs), min(ys), max(xs), max(ys)), 'label': 'PERSON'})
            # Gabungkan dengan deteksi lain
            others = [d for d in detections if d['label'] != 'PERSON']
            return merged + others

        while self.running:
            success, frame = cap.read()
            if not success: continue

            all_models = []
            if model_common_medium: all_models.append(model_common_medium)
            if model_yolov5s: all_models.append(model_yolov5s)

            all_detections = []
            for model_ref in all_models:
                results = model_ref(frame, verbose=False)[0]
                for r in results.boxes:
                    if r.conf[0] < CONFIDENCE_THRESHOLD: continue
                    xmin, ymin, xmax, ymax = map(int, r.xyxy[0].tolist())
                    label = model_ref.names[int(r.cls[0])].upper()
                    all_detections.append({'box': (xmin, ymin, xmax, ymax), 'label': label})

            # Gabungkan box PERSON yang overlap
            all_detections = merge_person_boxes(all_detections, iou_threshold=0.5)

            # Deteksi ancaman global
            ancaman_boxes = []
            for det in all_detections:
                label = det['label']
                if label in ['GUN', 'PISTOL', 'KNIFE', 'RIFLE', 'WEAPON']:
                    ancaman_boxes.append(det['box'])

            user_statuses = []
            person_idx = 0
            for det in all_detections:
                label = det['label']
                xmin, ymin, xmax, ymax = det['box']
                if label == 'PERSON':
                    person_roi = frame[ymin:ymax, xmin:xmax]
                    if person_roi.size == 0: continue
                    results_mp = face_mesh.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                    kantuk_terdeteksi = False
                    if person_idx not in drowsiness_counters: drowsiness_counters[person_idx] = 0
                    if results_mp.multi_face_landmarks:
                        face_landmarks = results_mp.multi_face_landmarks[0]
                        LEFT_EYE_IDXS, RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380], [33, 160, 158, 133, 153, 144]
                        left_eye = np.array([[face_landmarks.landmark[i].x * person_roi.shape[1], face_landmarks.landmark[i].y * person_roi.shape[0]] for i in LEFT_EYE_IDXS])
                        right_eye = np.array([[face_landmarks.landmark[i].x * person_roi.shape[1], face_landmarks.landmark[i].y * person_roi.shape[0]] for i in RIGHT_EYE_IDXS])
                        ear = (eye_aspect_ratio_mp(left_eye) + eye_aspect_ratio_mp(right_eye)) / 2.0
                        drowsiness_counters[person_idx] = drowsiness_counters[person_idx] + 1 if ear < EAR_THRESHOLD else 0
                        if drowsiness_counters[person_idx] >= EAR_CONSEC_FRAMES: kantuk_terdeteksi = True
                        cv2.putText(frame, f"EAR: {ear:.2f}", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # Jika landmark tidak ditemukan (mungkin nunduk/mata tertutup), langsung anggap kantuk
                        kantuk_terdeteksi = True
                        drowsiness_counters[person_idx] = EAR_CONSEC_FRAMES
                        cv2.putText(frame, "EAR: -", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # Cek ancaman di sekitar user
                    ancaman_terdeteksi = False
                    for abox in ancaman_boxes:
                        axmin, aymin, axmax, aymax = abox
                        # Jika overlap area dengan user
                        if not (xmax < axmin or xmin > axmax or ymax < aymin or ymin > aymax):
                            ancaman_terdeteksi = True
                            break
                    user_statuses.append({'user': person_idx+1, 'kantuk': kantuk_terdeteksi, 'ancaman': ancaman_terdeteksi})
                    color = (0, 255, 255) if kantuk_terdeteksi else (255, 0, 0)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, f"USER {person_idx+1}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    if kantuk_terdeteksi and not alarm_kantuk_on:
                        alarm_kantuk_on = True; threading.Thread(target=play_sound, args=("kantuk",), daemon=True).start()
                    elif not kantuk_terdeteksi: alarm_kantuk_on = False
                    if ancaman_terdeteksi and not alarm_ancaman_on:
                        alarm_ancaman_on = True; threading.Thread(target=play_sound, args=("ancaman",), daemon=True).start()
                    elif not ancaman_terdeteksi: alarm_ancaman_on = False
                    person_idx += 1
                elif label in ['GUN', 'PISTOL', 'KNIFE', 'RIFLE', 'WEAPON']:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                elif label != 'PERSON':
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self.change_pixmap_signal.emit(frame)
            self.update_status_signal.emit(user_statuses)
        
        cap.release()
        print("THREAD: Kamera dilepaskan dan thread berhenti.")

    def stop(self):
        self.running = False
        self.wait()

# ==============================================================================
# 2. KELAS UTAMA UNTUK GUI APLIKASI
# ==============================================================================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guardian AI - Dashboard Keamanan")
        self.setGeometry(100, 100, 960, 720)
        self.setMinimumSize(400, 300)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000000;")
        layout.addWidget(self.image_label, 1)

        font = QFont('Arial', 14)
        font.setBold(True)

        # Scroll area untuk status user
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.status_widget = QWidget()
        self.status_layout = QVBoxLayout(self.status_widget)
        self.scroll_area.setWidget(self.status_widget)
        layout.addWidget(self.scroll_area, 0)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_status_signal.connect(self.update_status)
        self.thread.start()

        self.user_labels = []

    def closeEvent(self, event):
        print("MAIN: Menutup aplikasi, menghentikan thread...")
        self.thread.stop()
        event.accept()

    def update_image(self, cv_img):
        label_size = self.image_label.size()
        target_w = label_size.width()
        target_h = label_size.height()
        if target_w / target_h > 4/3:
            video_h = target_h
            video_w = int(video_h * 4 / 3)
        else:
            video_w = target_w
            video_h = int(video_w * 3 / 4)
        qt_img_original = self.convert_cv_qt(cv_img)
        scaled_pixmap = qt_img_original.scaled(video_w, video_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        final_pixmap = QPixmap(label_size)
        final_pixmap.fill(Qt.black)
        painter = QPainter(final_pixmap)
        x = (label_size.width() - scaled_pixmap.width()) // 2
        y = (label_size.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()
        self.image_label.setPixmap(final_pixmap)

    def update_status(self, user_statuses):
        # Bersihkan label lama
        for lbl in self.user_labels:
            self.status_layout.removeWidget(lbl)
            lbl.deleteLater()
        self.user_labels = []
        if not user_statuses:
            lbl = QLabel("USER: Tidak Terdeteksi")
            lbl.setFont(QFont('Arial', 14))
            lbl.setStyleSheet("color: #bdc3c7; padding: 5px; font-weight: bold;")
            self.status_layout.addWidget(lbl)
            self.user_labels.append(lbl)
            return
        for status in user_statuses:
            user = status.get('user', 0)
            kantuk = status.get('kantuk', False)
            ancaman = status.get('ancaman', False)
            error = status.get('error', False)
            if error:
                text = "ERROR: Kamera tidak ditemukan"
                color = "#e74c3c"
            else:
                text = f"USER {user}: {'MENGANTUK' if kantuk else 'Terjaga'} | {'ANCAMAN' if ancaman else 'Aman'}"
                if ancaman:
                    color = "#e74c3c"
                elif kantuk:
                    color = "#f1c40f"
                else:
                    color = "#2ecc71"
            lbl = QLabel(text)
            lbl.setFont(QFont('Arial', 14))
            lbl.setStyleSheet(f"color: {color}; padding: 5px; font-weight: bold;")
            self.status_layout.addWidget(lbl)
            self.user_labels.append(lbl)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

# ==============================================================================
# 3. JALANKAN APLIKASI
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
