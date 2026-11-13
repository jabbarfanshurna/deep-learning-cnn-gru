"""Perbandingan CNN 1D dan GRU untuk analisis sentimen IMDB."""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Pengaturan eksperimen
SEED = 42
MAX_WORDS = 10_000
MAX_LEN = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 128
EPOCHS = 5

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def build_cnn_model():
    """Membangun CNN 1D untuk mengenali pola lokal pada urutan kata."""
    inputs = layers.Input(shape=(MAX_LEN,), dtype="int32", name="review_input")
    x = layers.Embedding(MAX_WORDS, EMBEDDING_DIM, name="embedding")(inputs)
    x = layers.Conv1D(128, kernel_size=5, activation="relu", name="conv1d_1")(x)
    x = layers.GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="sentiment_output")(x)

    model = models.Model(inputs, outputs, name="cnn_1d_sentiment")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru_model():
    """Membangun GRU untuk mempelajari urutan kata dalam ulasan."""
    inputs = layers.Input(shape=(MAX_LEN,), dtype="int32", name="review_input")
    x = layers.Embedding(
        MAX_WORDS,
        EMBEDDING_DIM,
        mask_zero=True,
        name="embedding",
    )(inputs)
    x = layers.GRU(64, name="gru")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="sentiment_output")(x)

    model = models.Model(inputs, outputs, name="gru_sentiment")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, model_name):
    """Menyimpan grafik akurasi dan loss selama pelatihan."""
    history_data = history.history
    epochs_range = range(1, len(history_data["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, history_data["accuracy"], label="Training")
    axes[0].plot(epochs_range, history_data["val_accuracy"], label="Validation")
    axes[0].set_title(f"{model_name} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history_data["loss"], label="Training")
    axes[1].plot(epochs_range, history_data["val_loss"], label="Validation")
    axes[1].set_title(f"{model_name} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{model_name.lower()}_training.png", dpi=150)
    plt.close(fig)


def train_and_evaluate(model_name, model, x_train, y_train, x_val, y_val, x_test, y_test):
    """Melatih model, mengukur waktu, dan mengevaluasi data testing."""
    print(f"\n{'=' * 60}\nMelatih model {model_name}\n{'=' * 60}")
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    start_time = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )
    training_seconds = time.perf_counter() - start_time

    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype("int32")

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, zero_division=0),
        "Waktu_Pelatihan_Detik": training_seconds,
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=["Negatif", "Positif"],
        zero_division=0,
    )
    report_path = RESULTS_DIR / f"{model_name.lower()}_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_test, predictions)
    model.save(MODELS_DIR / f"{model_name.lower()}.keras")
    plot_history(history, model_name)

    print(f"\nHasil evaluasi {model_name}:")
    for metric_name, value in metrics.items():
        if metric_name != "Model":
            print(f"{metric_name}: {value:.4f}")
    print("\nClassification Report:")
    print(report)

    return metrics, matrix


def save_comparison_chart(results_df):
    """Membuat grafik perbandingan metrik kedua model."""
    metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score"]
    chart_data = results_df.set_index("Model")[metric_columns]

    ax = chart_data.plot(kind="bar", figsize=(10, 5), ylim=(0, 1))
    ax.set_title("Perbandingan Performa CNN 1D dan GRU")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()


def save_confusion_matrices(cnn_matrix, gru_matrix):
    """Menyimpan confusion matrix untuk masing-masing model."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, matrix, title in [
        (axes[0], cnn_matrix, "CNN 1D"),
        (axes[1], gru_matrix, "GRU"),
    ]:
        image = axis.imshow(matrix, interpolation="nearest")
        axis.set_title(f"Confusion Matrix - {title}")
        axis.set_xlabel("Prediksi")
        axis.set_ylabel("Aktual")
        axis.set_xticks([0, 1], labels=["Negatif", "Positif"])
        axis.set_yticks([0, 1], labels=["Negatif", "Positif"])
        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main():
    tf.keras.utils.set_random_seed(SEED)

    print("Memuat dataset IMDB dari Keras...")
    (x_train_raw, y_train), (x_test_raw, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=MAX_WORDS
    )

    # Menyamakan panjang setiap ulasan agar bisa diproses oleh model.
    x_train = pad_sequences(
        x_train_raw,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )
    x_test = pad_sequences(
        x_test_raw,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Data training dibagi lagi menjadi training dan validation.
    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train,
    )

    print(f"Data training   : {len(x_train_final)}")
    print(f"Data validation: {len(x_val)}")
    print(f"Data testing    : {len(x_test)}")
    print(f"Panjang sequence: {MAX_LEN}")

    cnn_metrics, cnn_matrix = train_and_evaluate(
        "CNN_1D",
        build_cnn_model(),
        x_train_final,
        y_train_final,
        x_val,
        y_val,
        x_test,
        y_test,
    )

    gru_metrics, gru_matrix = train_and_evaluate(
        "GRU",
        build_gru_model(),
        x_train_final,
        y_train_final,
        x_val,
        y_val,
        x_test,
        y_test,
    )

    results_df = pd.DataFrame([cnn_metrics, gru_metrics])
    results_path = RESULTS_DIR / "comparison_metrics.csv"
    results_df.to_csv(results_path, index=False)

    save_comparison_chart(results_df)
    save_confusion_matrices(cnn_matrix, gru_matrix)

    print("\nRingkasan perbandingan:")
    print(results_df.to_string(index=False))
    print(f"\nHasil tersimpan di: {RESULTS_DIR}")
    print(f"Model tersimpan di : {MODELS_DIR}")
    print("Catatan: gunakan angka pada comparison_metrics.csv untuk mengisi laporan.")


if __name__ == "__main__":
    main()
