const MODEL_PATH = "./web-model/model.onnx";
const LABELS_PATH = "./web-model/labels.json";
const METRICS_PATH = "./web-model/metrics.json";
const IMAGE_SIZE = 224;

const state = {
  model: null,
  labels: [],
  metrics: null,
  imageFile: null,
  imageElement: null,
};

const imageInput = document.getElementById("imageInput");
const dropzone = document.getElementById("dropzone");
const previewImage = document.getElementById("previewImage");
const emptyPreview = document.getElementById("emptyPreview");
const predictButton = document.getElementById("predictButton");
const resetButton = document.getElementById("resetButton");
const statusBox = document.getElementById("statusBox");
const predictionLabel = document.getElementById("predictionLabel");
const predictionConfidence = document.getElementById("predictionConfidence");
const resultsList = document.getElementById("resultsList");
const classesCount = document.getElementById("classesCount");

function setStatus(message, type = "") {
  statusBox.textContent = message;
  statusBox.className = "status-box";
  if (type) {
    statusBox.classList.add(type);
  }
}

function renderRankings(predictions) {
  resultsList.innerHTML = "";

  predictions.forEach(({ label, score }) => {
    const item = document.createElement("div");
    item.className = "result-item";

    const labelRow = document.createElement("div");
    labelRow.className = "result-item-label";
    labelRow.innerHTML = `<span>${label}</span><strong>${(score * 100).toFixed(2)}%</strong>`;

    const meter = document.createElement("div");
    meter.className = "meter";

    const bar = document.createElement("div");
    bar.className = "meter-bar";
    bar.style.width = `${Math.max(score * 100, 2)}%`;

    meter.appendChild(bar);
    item.appendChild(labelRow);
    item.appendChild(meter);
    resultsList.appendChild(item);
  });
}

function resetUi() {
  state.imageFile = null;
  state.imageElement = null;
  imageInput.value = "";
  previewImage.src = "";
  previewImage.style.display = "none";
  emptyPreview.style.display = "grid";
  predictButton.disabled = !state.model;
  predictionLabel.textContent = state.model ? "Ready for analysis" : "Awaiting model";
  predictionConfidence.textContent = state.model
    ? "Upload an image to run a prediction."
    : "Train and export the model first.";
  resultsList.innerHTML = "";
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Unable to load ${path}`);
  }

  return response.json();
}

async function bootstrap() {
  try {
    const [labelsPayload, metricsPayload] = await Promise.all([
      loadJson(LABELS_PATH),
      loadJson(METRICS_PATH).catch(() => null),
    ]);

    state.labels = labelsPayload.labels || [];
    state.metrics = metricsPayload;
    classesCount.textContent = `${state.labels.length} classes`;

    if (!state.labels.length) {
      throw new Error("No class labels found.");
    }

    state.model = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });
    predictButton.disabled = false;
    predictionLabel.textContent = "Ready for analysis";
    predictionConfidence.textContent = "Upload an image to run a prediction.";

    const validationAccuracy = state.metrics?.best_val_accuracy;
    const heroAccuracy = document.getElementById("heroAccuracy");

    if (typeof validationAccuracy === "number") {
      const pct = (validationAccuracy * 100).toFixed(1);
      setStatus(`Model loaded · ${pct}% validation accuracy`, "success");
      if (heroAccuracy) {
        heroAccuracy.textContent = `Model accuracy: ${pct}%`;
      }
    } else {
      setStatus("Model loaded and ready.", "success");
    }
  } catch (error) {
    console.error(error);
    predictButton.disabled = true;
    classesCount.textContent = "0 classes";
    setStatus("Unable to load the classification model. Please try again later.", "error");
  }
}

function loadPreview(file) {
  const objectUrl = URL.createObjectURL(file);
  previewImage.onload = () => URL.revokeObjectURL(objectUrl);
  previewImage.src = objectUrl;
  previewImage.style.display = "block";
  emptyPreview.style.display = "none";
}

async function classifyImage() {
  if (!state.model || !state.imageFile) {
    return;
  }

  try {
    setStatus("Running prediction...", "success");

    const bitmap = await createImageBitmap(state.imageFile);
    const canvas = document.createElement("canvas");
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    const context = canvas.getContext("2d");
    context.drawImage(bitmap, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

    const { data } = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const floatData = new Float32Array(IMAGE_SIZE * IMAGE_SIZE * 3);

    // Pass raw [0, 255] pixel values — the ONNX model already contains
    // Rescaling(1/255) and MobileNetV2 preprocess_input layers.
    for (let sourceIndex = 0, targetIndex = 0; sourceIndex < data.length; sourceIndex += 4) {
      floatData[targetIndex++] = data[sourceIndex];
      floatData[targetIndex++] = data[sourceIndex + 1];
      floatData[targetIndex++] = data[sourceIndex + 2];
    }

    const inputTensor = new ort.Tensor("float32", floatData, [1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    const inputName = state.model.inputNames[0];
    const outputName = state.model.outputNames[0];
    const outputs = await state.model.run({ [inputName]: inputTensor });
    const probabilities = Array.from(outputs[outputName].data);

    bitmap.close();

    const ranked = probabilities
      .map((score, index) => ({
        label: state.labels[index] || `Class ${index}`,
        score,
      }))
      .sort((left, right) => right.score - left.score)
      .slice(0, 5);

    if (!ranked.length) {
      throw new Error("Prediction returned no classes.");
    }

    predictionLabel.textContent = ranked[0].label;
    predictionConfidence.textContent = `Confidence: ${(ranked[0].score * 100).toFixed(2)}%`;
    renderRankings(ranked);
    setStatus("Prediction complete.", "success");
  } catch (error) {
    console.error(error);
    setStatus("Prediction failed. Check that the exported ONNX model matches the expected input shape.", "error");
  }
}

function handleFileSelection(file) {
  if (!file) {
    return;
  }

  if (!file.type.startsWith("image/")) {
    setStatus("Please choose a valid image file.", "error");
    return;
  }

  state.imageFile = file;
  loadPreview(file);
  predictButton.disabled = !state.model;
  predictionLabel.textContent = "Image loaded";
  predictionConfidence.textContent = "Press Analyze Image to classify it.";
  resultsList.innerHTML = "";
  setStatus("Image ready for prediction.", "success");
}

imageInput.addEventListener("change", (event) => {
  handleFileSelection(event.target.files?.[0] || null);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0] || null;
  handleFileSelection(file);
});

predictButton.addEventListener("click", classifyImage);
resetButton.addEventListener("click", resetUi);

bootstrap();
resetUi();
