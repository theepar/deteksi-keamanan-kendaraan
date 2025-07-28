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

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter # PERBAIKAN: Tambahkan import QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ==============================================================================
# 1. THREAD UNTUK PROSES AI (BAGIAN INI TIDAK DIUBAH)
# ==============================================================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_status_signal = pyqtSignal(str, str)

    def run(self):
        self.running = True
        print("THREAD: Memuat model...")
        def load_model_safe(path):
            try:
                if os.path.exists(path): return YOLO(path)
                else: print(f"Model file not found: {path}"); return None
            except Exception as e: print(f"Model load error for {path}: {e}"); return None

        model_yolov5s = load_model_safe('yolov5su.pt')
        model_common_medium = load_model_safe('yolov8m-object.pt')
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("THREAD: Model loading completed.")

        CONFIDENCE_THRESHOLD, EAR_THRESHOLD, EAR_CONSEC_FRAMES = 0.5, 0.23, 25
        drowsiness_counter = 0

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
            print("Error: Tidak bisa membuka kamera."); self.update_status_signal.emit("ERROR", "Kamera tidak ditemukan"); return
            
        alarm_kantuk_on, alarm_ancaman_on = False, False

        while self.running:
            success, frame = cap.read()
            if not success: continue

            kantuk_terdeteksi, ancaman_terdeteksi, driver_found = False, False, False
            driver_status_text, threat_status_text = "USER: Tidak Terdeteksi", "STATUS: Aman"

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

            for det in all_detections:
                label = det['label']
                xmin, ymin, xmax, ymax = det['box']
                if label == 'PERSON' and not driver_found:
                    driver_found = True
                    person_roi = frame[ymin:ymax, xmin:xmax]
                    if person_roi.size == 0: continue
                    results_mp = face_mesh.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                    if results_mp.multi_face_landmarks:
                        face_landmarks = results_mp.multi_face_landmarks[0]
                        LEFT_EYE_IDXS, RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380], [33, 160, 158, 133, 153, 144]
                        left_eye = np.array([[face_landmarks.landmark[i].x * person_roi.shape[1], face_landmarks.landmark[i].y * person_roi.shape[0]] for i in LEFT_EYE_IDXS])
                        right_eye = np.array([[face_landmarks.landmark[i].x * person_roi.shape[1], face_landmarks.landmark[i].y * person_roi.shape[0]] for i in RIGHT_EYE_IDXS])
                        ear = (eye_aspect_ratio_mp(left_eye) + eye_aspect_ratio_mp(right_eye)) / 2.0
                        drowsiness_counter = drowsiness_counter + 1 if ear < EAR_THRESHOLD else 0
                        if drowsiness_counter >= EAR_CONSEC_FRAMES: kantuk_terdeteksi = True
                        cv2.putText(frame, f"EAR: {ear:.2f}", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    driver_status_text = "USER: MENGANTUK" if kantuk_terdeteksi else "USER: Terjaga"
                    color = (0, 255, 255) if kantuk_terdeteksi else (255, 0, 0)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, "USER", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                elif label in ['GUN', 'PISTOL', 'KNIFE', 'RIFLE', 'WEAPON']:
                    ancaman_terdeteksi = True
                    threat_status_text = "STATUS: ANCAMAN TERDETEKSI!"
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                elif not ancaman_terdeteksi and label != 'PERSON':
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if kantuk_terdeteksi and not alarm_kantuk_on:
                alarm_kantuk_on = True; threading.Thread(target=play_sound, args=("kantuk",), daemon=True).start()
            elif not kantuk_terdeteksi: alarm_kantuk_on = False
            if ancaman_terdeteksi and not alarm_ancaman_on:
                alarm_ancaman_on = True; threading.Thread(target=play_sound, args=("ancaman",), daemon=True).start()
            elif not ancaman_terdeteksi: alarm_ancaman_on = False

            self.change_pixmap_signal.emit(frame)
            self.update_status_signal.emit(driver_status_text, threat_status_text)
        
        cap.release()
        print("THREAD: Kamera dilepaskan dan thread berhenti.")

    def stop(self):
        self.running = False
        self.wait()

# ==============================================================================
# 2. KELAS UTAMA UNTUK GUI APLIKASI (BAGIAN INI DIUBAH)
# ==============================================================================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guardian AI - Dashboard Keamanan")
        # Default window 4:3 (misal 960x720), tetap bisa resize
        self.setGeometry(100, 100, 960, 720)
        self.setMinimumSize(400, 300)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000000;")
        layout.addWidget(self.image_label, 1)

        status_layout = QHBoxLayout()
        font = QFont('Arial', 14)
        font.setBold(True)

        self.driver_status_label = QLabel("USER: -")
        self.driver_status_label.setFont(font)
        
        self.threat_status_label = QLabel("STATUS: -")
        self.threat_status_label.setFont(font)
        
        status_layout.addWidget(self.driver_status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.threat_status_label)
        
        layout.addLayout(status_layout)
        self.setStyleSheet("background-color: #2c3e50; color: white;")

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_status_signal.connect(self.update_status)
        self.thread.start()

    def closeEvent(self, event):
        print("MAIN: Menutup aplikasi, menghentikan thread...")
        self.thread.stop()
        event.accept()

    def update_image(self, cv_img):
        """
        Video selalu tampil dengan rasio 4:3 di tengah label, apapun ukuran window. Area kosong hitam.
        """
        label_size = self.image_label.size()
        target_w = label_size.width()
        target_h = label_size.height()
        # Hitung area 4:3 terbesar yang muat di label
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


    def update_status(self, driver_status, threat_status):
        self.driver_status_label.setText(driver_status)
        self.threat_status_label.setText(threat_status)
        self.driver_status_label.setStyleSheet(f"color: {'#f1c40f' if 'MENGANTUK' in driver_status else '#2ecc71'}; padding: 5px; font-weight: bold;")
        self.threat_status_label.setStyleSheet(f"color: {'#e74c3c' if 'ANCAMAN' in threat_status else '#2ecc71'}; padding: 5px; font-weight: bold;")
        
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
