# 🛡️ Guardian AI - Dashboard Keamanan Kendaraan

Guardian AI adalah sebuah sistem dashboard keamanan kendaraan berbasis kecerdasan buatan (AI). Aplikasi ini menggunakan kamera untuk mendeteksi berbagai objek seperti orang, senjata tajam, dan senjata api secara *real-time*. Selain itu, sistem ini juga dilengkapi fitur deteksi kantuk pada pengemudi untuk meningkatkan keselamatan berkendara.

---

## ✨ Fitur Utama

-   **Deteksi Objek Real-time**: Mengidentifikasi orang, pisau, pistol, dan senjata api lainnya menggunakan model YOLOv5 dan YOLOv8.
-   **Deteksi Kantuk Pengemudi (Drowsiness Detection)**: Menggunakan MediaPipe untuk memonitor *Eye Aspect Ratio* (EAR) dan membunyikan alarm jika pengemudi terindikasi mengantuk.
-   **Antarmuka Responsif**: Tampilan video dari kamera dapat diubah ukurannya dan menyesuaikan secara proporsional dengan jendela aplikasi.
-   **Tampilan Informasi (Overlay)**: Menampilkan status keamanan, daftar objek yang terdeteksi, dan peringatan ancaman langsung di atas tayangan video.
-   **Sistem Peringatan Audio**: Alarm akan berbunyi secara otomatis saat terdeteksi adanya ancaman atau kondisi kantuk pada pengemudi.
-   **Dukungan Multi-Model**: Menggabungkan beberapa model YOLO untuk meningkatkan akurasi dan cakupan deteksi.

## 🛠️ Teknologi yang Digunakan

-   **Python 3.8+**
-   **OpenCV**: Untuk pemrosesan video dan gambar.
-   **YOLOv5 & YOLOv8**: Sebagai model utama untuk deteksi objek.
-   **MediaPipe**: Untuk deteksi landmark wajah dan analisis kantuk.
-   **Pygame**: Untuk memutar suara alarm.
-   **Tkinter**: Untuk membangun antarmuka grafis (GUI).

## 🚀 Panduan Instalasi

Pastikan semua persyaratan sistem di bawah ini terpenuhi sebelum melanjutkan instalasi.

#### Prasyarat

-   Python 3.8 atau versi lebih baru.
-   Sistem Operasi: Windows 10/11 (direkomendasikan).
-   Webcam yang berfungsi (internal atau eksternal).

#### Langkah-langkah Instalasi

1.  **Clone Repository Ini**
    Buka terminal atau Git Bash, lalu jalankan perintah berikut:
    ```bash
    git clone https://github.com/theepar/deteksi-keamanan-kendaraan.git
    cd deteksi-keamanan-kendaraan
    ```

2.  **Buat dan Aktifkan Virtual Environment (Sangat Direkomendasikan)**
    Ini akan menjaga dependensi proyek agar tidak bercampur dengan instalasi Python global Anda.
    ```powershell
    # Buat environment
    python -m venv venv

    # Aktifkan environment
    .\venv\Scripts\activate
    ```

3.  **Install Semua Dependensi**
    Pastikan virtual environment Anda sudah aktif, lalu jalankan:
    ```bash
    # Upgrade pip terlebih dahulu
    pip install --upgrade pip

    # Install semua library yang dibutuhkan dari requirements.txt
    pip install -r requirements.txt
    ```

4.  **Siapkan File Model**
    File model YOLO (`yolov5su.pt`, `yolov8m.pt`, dll.) tidak termasuk dalam repository karena ukurannya yang besar.
    -   Unduh file model yang diperlukan.
    -   Letakkan semua file model (`.pt`) di dalam folder utama proyek.

## 🏃 Cara Menjalankan Aplikasi

Cara termudah untuk menjalankan aplikasi di Windows adalah dengan menggunakan file batch yang telah disediakan.

```bash
# Cukup klik dua kali file ini di File Explorer
run_app.bat