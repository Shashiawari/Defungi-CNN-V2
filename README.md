# FungiScan AI

This project has been rebuilt from a desktop Tkinter prototype into a Vercel-friendly web application.

## What changed

- The old Python desktop UI was replaced with a modern HTML, CSS, and JavaScript frontend.
- Model training is now handled by dedicated Python scripts in `training/`.
- The frontend expects an ONNX export inside `web-model/`.
- The site can be deployed as a static app on Vercel.

## Project structure

- `index.html`, `styles.css`, `app.js`: new browser UI and client-side inference.
- `training/download_dataset.py`: downloads and extracts a Kaggle dataset.
- `training/train_model.py`: trains a transfer-learning image classifier and exports ONNX assets.
- `web-model/`: generated model files for browser inference.
- `main.py`: compatibility entry point that points you to the new workflow.

## Local workflow

1. Install Python dependencies:

   ```bash
   pip install -r training/requirements.txt
   ```

   Use Python 3.10 for the training environment on Windows because TensorFlow 2.15 is the most stable option for this setup.

2. Configure Kaggle credentials:

   - Create `%USERPROFILE%\\.kaggle\\kaggle.json`
   - Or set `KAGGLE_USERNAME` and `KAGGLE_KEY`

3. Use the local dataset:

   The project now trains directly from [archive](G:\Defungi-CNN\Defungi V-2\Fungi\archive), which already contains the class folders `H1`, `H2`, `H3`, `H5`, and `H6`.

4. Optional Kaggle download:

   ```bash
   python training/download_dataset.py
   ```

   Default Kaggle dataset:
   [anshtanwar/microscopic-fungi-images](https://www.kaggle.com/datasets/anshtanwar/microscopic-fungi-images)

5. Train and export the model:

   ```bash
   python training/train_model.py --data-dir archive --epochs 8 --fine-tune-epochs 4
   ```

6. Preview locally:

   ```bash
   python -m http.server 8000
   ```

   Then open `http://localhost:8000`.

## Vercel deployment

1. Make sure `web-model/model.onnx` and `web-model/labels.json` exist.
2. Deploy the repo:

   ```bash
   vercel --prod
   ```

The app is static, so Vercel will serve the HTML frontend and model artifacts directly.

## Notes on accuracy

Accuracy depends mostly on the dataset and label quality. The new training pipeline improves on the original code by using:

- Transfer learning with MobileNetV2
- Data augmentation
- Validation split
- Early stopping and learning-rate reduction
- Class weighting for imbalanced datasets


