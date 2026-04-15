import json
import math
import pathlib
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import layers


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
MAX_EPOCHS_A = 4
MAX_EPOCHS_B_HEAD = 6
MAX_EPOCHS_B_FINETUNE = 3
PATIENCE = 2
MAX_EPOCHS_SWEEP = 2
RUN_DROPOUT_SWEEP = False
FAST_MODE = False
FAST_TRAIN_BATCHES = 0
FAST_VAL_BATCHES = 0
FAST_TEST_BATCHES = 0

BASE_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def locate_flower_dataset() -> pathlib.Path:
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    archive_path = tf.keras.utils.get_file(
        fname="flower_photos.tgz",
        origin=dataset_url,
        extract=True,
    )

    archive_path = pathlib.Path(archive_path)
    keras_cache = archive_path.parent

    candidates = [
        keras_cache / "flower_photos",
        archive_path.with_suffix("") / "flower_photos",
        archive_path.with_suffix("").with_suffix(""),
    ]

    for match in keras_cache.rglob("flower_photos"):
        if match.is_dir():
            candidates.insert(0, match)

    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c

    raise FileNotFoundError("Nao foi possivel encontrar o diretorio flower_photos extraido.")


def make_datasets(data_dir: pathlib.Path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )

    valtest_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names

    valtest_batches = tf.data.experimental.cardinality(valtest_ds).numpy()
    val_batches = max(1, valtest_batches // 2)
    test_batches = valtest_batches - val_batches

    val_ds = valtest_ds.take(val_batches)
    test_ds = valtest_ds.skip(val_batches)

    print(f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Val batches  : {val_batches}")
    print(f"Test batches : {test_batches}")

    autotune = tf.data.AUTOTUNE

    train_ds_n = train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds_n = val_ds.cache().prefetch(autotune)
    test_ds_n = test_ds.cache().prefetch(autotune)

    if FAST_MODE:
        train_ds_n = train_ds_n.take(FAST_TRAIN_BATCHES)
        val_ds_n = val_ds_n.take(FAST_VAL_BATCHES)
        test_ds_n = test_ds_n.take(FAST_TEST_BATCHES)
        print("FAST_MODE ativo: usando subconjunto de batches para execucao rapida.")

    return train_ds_n, val_ds_n, test_ds_n, class_names


@dataclass
class EvalMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion: list
    report: dict


def build_model_a(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_b(num_classes: int, dropout_rate: float = 0.4):
    aug = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.15, seed=SEED),
            layers.RandomZoom(0.15, seed=SEED),
            layers.RandomContrast(0.1, seed=SEED),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def evaluate_model(model: tf.keras.Model, test_ds, class_names):
    y_true = []
    y_pred = []

    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = EvalMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        precision_weighted=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall_weighted=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
        report=classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    )

    return metrics, y_true, y_pred


def save_loss_plot(hist_a, hist_b, stopped_epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(hist_a.history["loss"], label="Modelo A - train loss", color="#d62728")
    plt.plot(hist_a.history["val_loss"], label="Modelo A - val loss", color="#ff9896")

    plt.plot(hist_b["loss"], label="Modelo B - train loss", color="#1f77b4")
    plt.plot(hist_b["val_loss"], label="Modelo B - val loss", color="#9ecae1")

    if stopped_epoch is not None:
        plt.axvline(stopped_epoch - 1, color="black", linestyle="--", linewidth=1.2, label=f"EarlyStopping epoch={stopped_epoch}")

    plt.title("Comparacao de curvas de loss (A vs B)")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_a_vs_b.png", dpi=150)
    plt.close()


def save_confusion_heatmap(cm, class_names, title, out_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=150)
    plt.close()


def run_dropout_sweep(train_ds, val_ds, test_ds, class_names, rates=(0.3, 0.5, 0.7)):
    rows = []
    for r in rates:
        print(f"[Dropout Sweep] Treinando com dropout={r}")
        model, _ = build_model_b(len(class_names), dropout_rate=r)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=MAX_EPOCHS_SWEEP,
            verbose=0,
            callbacks=callbacks,
        )

        metrics, _, _ = evaluate_model(model, test_ds, class_names)
        rows.append(
            {
                "dropout": r,
                "epochs_ran": len(history.history["loss"]),
                "f1_macro": metrics.f1_macro,
                "accuracy": metrics.accuracy,
            }
        )

    return rows


def to_dict(metrics: EvalMetrics):
    return {
        "accuracy": metrics.accuracy,
        "precision_macro": metrics.precision_macro,
        "recall_macro": metrics.recall_macro,
        "f1_macro": metrics.f1_macro,
        "precision_weighted": metrics.precision_weighted,
        "recall_weighted": metrics.recall_weighted,
        "f1_weighted": metrics.f1_weighted,
        "confusion": metrics.confusion,
        "report": metrics.report,
    }


def merge_histories(*histories):
    merged = {"loss": [], "val_loss": []}
    for h in histories:
        merged["loss"].extend(h.history.get("loss", []))
        merged["val_loss"].extend(h.history.get("val_loss", []))
    return merged


def main():
    data_dir = locate_flower_dataset()
    print(f"Dataset localizado em: {data_dir}")

    train_ds, val_ds, test_ds, class_names = make_datasets(data_dir)
    print(f"Classes: {class_names}")

    model_a = build_model_a(len(class_names))
    hist_a = model_a.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS_A, verbose=2)
    metrics_a, _, _ = evaluate_model(model_a, test_ds, class_names)

    model_b, base_model = build_model_b(len(class_names), dropout_rate=0.4)

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    hist_b_head = model_b.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS_B_HEAD,
        verbose=2,
        callbacks=[early, reduce_lr],
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model_b.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_ft = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr_ft = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )

    hist_b_ft = model_b.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS_B_FINETUNE,
        verbose=2,
        callbacks=[early_ft, reduce_lr_ft],
    )

    hist_b = merge_histories(hist_b_head, hist_b_ft)
    stopped_epoch = len(hist_b["loss"])
    metrics_b, _, _ = evaluate_model(model_b, test_ds, class_names)

    save_loss_plot(hist_a, hist_b, stopped_epoch)
    save_confusion_heatmap(np.array(metrics_a.confusion), class_names, "Matriz de Confusao - Modelo A", "cm_modelo_a.png")
    save_confusion_heatmap(np.array(metrics_b.confusion), class_names, "Matriz de Confusao - Modelo B", "cm_modelo_b.png")

    sweep_rows = []
    if RUN_DROPOUT_SWEEP:
        sweep_rows = run_dropout_sweep(train_ds, val_ds, test_ds, class_names, rates=(0.3, 0.5, 0.7))

    result = {
        "class_names": class_names,
        "modelo_a": to_dict(metrics_a),
        "modelo_b": to_dict(metrics_b),
        "modelo_b_early_stopping_stopped_epoch": stopped_epoch,
        "dropout_sweep": sweep_rows,
        "arquivos_gerados": [
            str(OUT_DIR / "loss_a_vs_b.png"),
            str(OUT_DIR / "cm_modelo_a.png"),
            str(OUT_DIR / "cm_modelo_b.png"),
        ],
    }

    with open(OUT_DIR / "metricas_flores.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== RESULTADOS RESUMIDOS ===")
    print(f"Modelo A - acc: {metrics_a.accuracy:.4f}, macro_f1: {metrics_a.f1_macro:.4f}, weighted_f1: {metrics_a.f1_weighted:.4f}")
    print(f"Modelo B - acc: {metrics_b.accuracy:.4f}, macro_f1: {metrics_b.f1_macro:.4f}, weighted_f1: {metrics_b.f1_weighted:.4f}")
    print(f"EarlyStopping parou em: epoca {stopped_epoch}")
    print(f"Arquivo de metricas: {OUT_DIR / 'metricas_flores.json'}")


if __name__ == "__main__":
    main()
