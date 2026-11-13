# Perbandingan CNN 1D dan GRU untuk Analisis Sentimen Ulasan Film

Proyek ini membandingkan dua arsitektur deep learning, yaitu **Convolutional Neural Network 1D (CNN 1D)** dan **Gated Recurrent Unit (GRU)**, untuk mengklasifikasikan sentimen ulasan film menjadi positif atau negatif.

## 1. Tujuan

1. Mengimplementasikan model CNN 1D untuk klasifikasi sentimen teks.
2. Mengimplementasikan model GRU untuk klasifikasi sentimen teks.
3. Mengevaluasi kedua model menggunakan data pengujian yang sama.
4. Membandingkan Accuracy, Precision, Recall, F1-Score, dan waktu pelatihan.

## 2. Dataset

Dataset yang digunakan adalah **IMDB Movie Reviews** yang tersedia melalui `tf.keras.datasets.imdb`.

Dataset berisi 50.000 ulasan film berlabel positif atau negatif, dengan 25.000 data training dan 25.000 data testing. Ulasan pada dataset bawaan Keras sudah diubah menjadi urutan indeks kata.

Dataset diunduh otomatis saat program pertama kali dijalankan. Tidak perlu mengunduh atau memasukkan file dataset secara manual.

Sumber dokumentasi: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data

## 3. Arsitektur Model

### CNN 1D

- Input: urutan indeks kata dengan panjang 200.
- Embedding: dimensi vektor 128.
- Conv1D: 128 filter dengan ukuran kernel 5 dan aktivasi ReLU.
- GlobalMaxPooling1D.
- Dense: 64 neuron dengan aktivasi ReLU.
- Dropout: 0,5.
- Output: 1 neuron dengan aktivasi sigmoid.

CNN 1D menggunakan filter konvolusi untuk mempelajari pola lokal pada urutan kata, misalnya kombinasi kata yang sering muncul pada ulasan positif atau negatif.

### GRU

- Input: urutan indeks kata dengan panjang 200.
- Embedding: dimensi vektor 128 dan `mask_zero=True`.
- GRU: 64 unit.
- Dropout: 0,5.
- Output: 1 neuron dengan aktivasi sigmoid.

GRU merupakan jenis recurrent neural network yang memproses urutan data dan menggunakan mekanisme gate untuk membantu mempertahankan informasi penting sepanjang urutan.

## 4. Pengaturan Eksperimen

| Parameter | Nilai |
|---|---:|
| Jumlah kata maksimum | 10.000 |
| Panjang sequence | 200 |
| Dimensi embedding | 128 |
| Pembagian data training | 80% dari data training awal |
| Pembagian data validation | 20% dari data training awal |
| Data testing | 25.000 ulasan bawaan dataset |
| Batch size | 128 |
| Epoch maksimum | 5 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Threshold klasifikasi | 0,5 |

Early stopping digunakan dengan memantau validation loss dan `patience=2`.

## 5. Instalasi dan Menjalankan Program

Disarankan menggunakan Python 3.11 atau versi yang kompatibel dengan TensorFlow yang terpasang.

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Apabila PowerShell menolak aktivasi environment, jalankan perintah berikut pada jendela PowerShell tersebut, lalu aktifkan kembali environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Google Colab

Unggah `main.py` dan `requirements.txt` ke Colab, kemudian jalankan instalasi dependency jika diperlukan:

```python
!pip install -r requirements.txt
!python main.py
```

Pelatihan dapat memerlukan waktu cukup lama, terutama jika dijalankan menggunakan CPU. Program akan mengunduh dataset secara otomatis pada proses pertama.

## 6. Struktur Repository

```text
deep-learning-cnn-gru/
├── main.py
├── predict.py
├── requirements.txt
├── README.md
├── report_template.md
├── .gitignore
├── models/
│   ├── cnn_1d.keras       # dibuat setelah training
│   └── gru.keras          # dibuat setelah training
└── results/
    ├── comparison_metrics.csv
    ├── cnn_1d_training.png
    ├── gru_training.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── cnn_1d_classification_report.txt
    └── gru_classification_report.txt
```

File di folder `models/` dan `results/` dibuat otomatis setelah program berhasil dijalankan.

## 7. Evaluasi

Program menghitung metrik berikut pada data testing:

- **Accuracy:** proporsi seluruh prediksi yang benar.
- **Precision:** ketepatan prediksi kelas positif.
- **Recall:** kemampuan model menemukan data positif.
- **F1-Score:** rata-rata harmonis Precision dan Recall.
- **Waktu pelatihan:** durasi proses training dalam detik.

Hasil angka aktual disimpan pada `results/comparison_metrics.csv`. Classification report dan confusion matrix juga disimpan agar hasil kedua model dapat dianalisis.

**Penting:** jangan mengisi angka hasil secara perkiraan. Jalankan program dan gunakan angka yang benar-benar dihasilkan oleh eksperimen.

## 8. Mencoba Prediksi Ulasan Baru

Setelah training selesai, jalankan:

```powershell
python predict.py
```

Pilih `cnn` atau `gru`, lalu masukkan ulasan film dalam bahasa Inggris. Program akan menampilkan prediksi positif/negatif dan skor probabilitas positif.

## 9. Referensi

- TensorFlow. IMDB Dataset API: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data
- TensorFlow. Text classification with an RNN: https://www.tensorflow.org/text/tutorials/text_classification_rnn
- TensorFlow. Basic text classification: https://www.tensorflow.org/tutorials/keras/text_classification

## 10. Catatan Pengembangan

Eksperimen ini menggunakan dataset berbahasa Inggris dan representasi kata yang telah disediakan oleh dataset Keras. Hasil penelitian hanya berlaku pada konfigurasi, dataset, dan pembagian data yang digunakan. Perbandingan performa perlu didasarkan pada hasil aktual, bukan asumsi bahwa salah satu arsitektur pasti lebih baik.
