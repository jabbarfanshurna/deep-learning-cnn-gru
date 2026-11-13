# LAPORAN TUGAS DEEP LEARNING

## Perbandingan CNN 1D dan GRU untuk Analisis Sentimen Ulasan Film

**Nama:** [Isi nama]  
**NIM:** [Isi NIM]  
**Mata Kuliah:** Deep Learning  

## 1. Pendahuluan

Analisis sentimen merupakan proses mengidentifikasi kecenderungan opini atau perasaan dalam suatu teks. Pada proyek ini, analisis sentimen diterapkan pada ulasan film untuk mengelompokkan ulasan ke dalam dua kelas, yaitu positif dan negatif. Penelitian ini mengimplementasikan dan membandingkan dua arsitektur deep learning, yaitu CNN 1D dan GRU.

## 2. Tujuan

1. Mengimplementasikan CNN 1D untuk klasifikasi sentimen ulasan film.
2. Mengimplementasikan GRU untuk klasifikasi sentimen ulasan film.
3. Membandingkan performa kedua model menggunakan metrik evaluasi yang sama.

## 3. Dataset

Dataset yang digunakan adalah IMDB Movie Reviews dari `tf.keras.datasets.imdb`. Dataset memiliki 50.000 ulasan film yang terbagi menjadi 25.000 data training dan 25.000 data testing. Label terdiri atas 0 (negatif) dan 1 (positif). Data yang disediakan Keras telah direpresentasikan dalam bentuk urutan indeks kata.

## 4. Metodologi

### 4.1 Preprocessing

Urutan indeks kata dibatasi pada 10.000 kata yang paling sering digunakan. Seluruh ulasan disamakan menjadi 200 token menggunakan padding dan truncating. Data training awal dibagi menjadi 80% data training dan 20% data validation. Data testing tetap digunakan untuk evaluasi akhir.

### 4.2 Arsitektur CNN 1D

[Tuliskan penjelasan singkat arsitektur dan sertakan ringkasan model dari terminal atau output program.]

### 4.3 Arsitektur GRU

[Tuliskan penjelasan singkat arsitektur dan sertakan ringkasan model dari terminal atau output program.]

### 4.4 Konfigurasi Training

| Parameter | Nilai |
|---|---:|
| Vocabulary maksimum | 10.000 |
| Panjang sequence | 200 |
| Embedding dimension | 128 |
| Batch size | 128 |
| Epoch maksimum | 5 |
| Optimizer | Adam |
| Loss function | Binary Crossentropy |

## 5. Hasil dan Pembahasan

Masukkan hasil aktual dari `results/comparison_metrics.csv`.

| Model | Accuracy | Precision | Recall | F1-Score | Waktu training (detik) |
|---|---:|---:|---:|---:|---:|
| CNN 1D | [isi] | [isi] | [isi] | [isi] | [isi] |
| GRU | [isi] | [isi] | [isi] | [isi] | [isi] |

### 5.1 Grafik Training

[Sisipkan `cnn_1d_training.png` dan `gru_training.png`. Jelaskan perubahan training/validation accuracy dan loss.]

### 5.2 Confusion Matrix

[Sisipkan `confusion_matrices.png`. Jelaskan prediksi benar dan salah pada kelas positif dan negatif.]

### 5.3 Perbandingan Model

[Sisipkan `model_comparison.png`. Bahas model yang memperoleh hasil lebih baik berdasarkan metrik aktual. Jangan menyimpulkan hanya dari accuracy apabila precision, recall, atau F1-Score menunjukkan pola yang berbeda.]

## 6. Kesimpulan

Berdasarkan eksperimen yang dilakukan, model [isi model] memperoleh hasil [isi berdasarkan hasil aktual]. Perbandingan ini menunjukkan bahwa performa arsitektur dapat dipengaruhi oleh karakteristik data, konfigurasi model, serta parameter pelatihan. Hasil ini terbatas pada dataset IMDB dan konfigurasi eksperimen yang digunakan.

## 7. Tautan Repository

[Masukkan URL repository GitHub setelah proyek diunggah.]
