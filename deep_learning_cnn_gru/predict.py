"""Mencoba model CNN atau GRU pada satu ulasan film berbahasa Inggris."""

import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 10_000
MAX_LEN = 200
BASE_DIR = Path(__file__).resolve().parent


def encode_review(review, word_index):
    """Mengubah teks menjadi urutan angka sesuai indeks kata dataset IMDB."""
    tokens = re.findall(r"[a-zA-Z']+", review.lower())
    sequence = [1]  # Penanda awal ulasan pada dataset IMDB.

    for word in tokens:
        word_number = word_index.get(word)
        if word_number is None or word_number + 3 >= MAX_WORDS:
            sequence.append(2)  # Token untuk kata yang tidak dikenal.
        else:
            sequence.append(word_number + 3)

    return pad_sequences(
        [sequence],
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )


def main():
    model_choice = input("Pilih model (cnn/gru): ").strip().lower()
    if model_choice not in {"cnn", "gru"}:
        print("Pilihan tidak valid. Masukkan cnn atau gru.")
        return

    model_file = "cnn_1d.keras" if model_choice == "cnn" else "gru.keras"
    model_path = BASE_DIR / "models" / model_file
    if not model_path.exists():
        print(f"Model belum ditemukan: {model_path}")
        print("Jalankan python main.py terlebih dahulu.")
        return

    review = input("Masukkan ulasan film dalam bahasa Inggris: ").strip()
    if not review:
        print("Ulasan tidak boleh kosong.")
        return

    model = tf.keras.models.load_model(model_path)
    word_index = tf.keras.datasets.imdb.get_word_index()
    sequence = encode_review(review, word_index)
    probability = float(model.predict(sequence, verbose=0)[0][0])

    sentiment = "POSITIF" if probability >= 0.5 else "NEGATIF"
    print(f"\nPrediksi sentimen: {sentiment}")
    print(f"Skor probabilitas positif: {probability:.4f}")
    print("Catatan: model dilatih menggunakan ulasan film berbahasa Inggris.")


if __name__ == "__main__":
    main()
