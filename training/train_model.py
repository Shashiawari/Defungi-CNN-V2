import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras


IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export a fungi image classifier.")
    parser.add_argument("--data-dir", default="archive", help="Root directory containing one subfolder per class.")
    parser.add_argument("--epochs", type=int, default=10, help="Initial frozen-base training epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=6, help="Fine-tuning epochs after unfreezing part of the backbone.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5, help="Learning rate for the fine-tuning stage.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--export-dir", default="web-model", help="Output directory for ONNX assets.")
    return parser.parse_args()


def prepare_datasets(data_dir: Path, batch_size: int, validation_split: float, seed: int):
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )

    class_names = train_ds.class_names

    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.12),
            keras.layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )

    def prepare_for_training(images, labels):
        images = augmentation(images)
        return images, labels

    def prepare_for_validation(images, labels):
        return images, labels

    train_ds = train_ds.map(prepare_for_training, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(prepare_for_validation, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def compute_weights(data_dir: Path, class_names: list[str]) -> dict[int, float]:
    labels = []
    for index, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        count = sum(1 for path in class_dir.rglob("*") if path.is_file())
        labels.extend([index] * count)

    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=np.asarray(labels))
    return {index: float(weight) for index, weight in enumerate(weights)}


def build_model(num_classes: int, learning_rate: float) -> tuple[keras.Model, keras.Model]:
    base_model = keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE + (3,))
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    export_dir = Path(args.export_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_ds, val_ds, class_names = prepare_datasets(
        data_dir=data_dir,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    model, base_model = build_model(num_classes=len(class_names), learning_rate=args.learning_rate)
    class_weight = compute_weights(data_dir, class_names)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    initial_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.fine_tune_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=len(initial_history.history.get("loss", [])),
        epochs=len(initial_history.history.get("loss", [])) + args.fine_tune_epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    history = {
        key: initial_history.history.get(key, []) + fine_tune_history.history.get(key, [])
        for key in set(initial_history.history) | set(fine_tune_history.history)
    }

    export_dir.mkdir(parents=True, exist_ok=True)
    keras_model_dir = export_dir / "_keras"
    keras_model_dir.mkdir(parents=True, exist_ok=True)

    keras_model_path = keras_model_dir / "model.keras"
    model.save(keras_model_path)
    input_signature = [
        tf.TensorSpec([None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], tf.float32, name="input")
    ]
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=str(export_dir / "model.onnx"),
    )

    labels_path = export_dir / "labels.json"
    labels_path.write_text(json.dumps({"labels": class_names}, indent=2), encoding="utf-8")

    metrics_path = export_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_train_accuracy": float(max(history.get("accuracy", [0.0]))),
                "best_val_accuracy": float(max(history.get("val_accuracy", [0.0]))),
                "epochs_ran": len(history.get("loss", [])),
                "image_size": list(IMAGE_SIZE),
                "class_names": class_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Training complete. Exported ONNX model to {export_dir}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Best validation accuracy: {max(history.get('val_accuracy', [0.0])):.4f}")


if __name__ == "__main__":
    main()
