# Training workflow

Use these scripts to prepare a proper fungi classifier before deploying to Vercel.

Use Python 3.10 for the training environment on Windows.

## 1. Use the local archive dataset

The current project already includes training data in [archive](G:\Defungi-CNN\Defungi V-2\Fungi\archive).

## 2. Optional: download a dataset from Kaggle

```bash
python training/download_dataset.py
```

Default Kaggle dataset:
[anshtanwar/microscopic-fungi-images](https://www.kaggle.com/datasets/anshtanwar/microscopic-fungi-images)

Your dataset should end up in a structure like this:

```text
dataset/
  amanita/
    image1.jpg
  chanterelle/
    image2.jpg
  oyster/
    image3.jpg
```

## 3. Train and export the browser model

```bash
python training/train_model.py --data-dir archive --epochs 8 --fine-tune-epochs 4 --batch-size 32
```

This script exports:

- `web-model/model.onnx`
- `web-model/labels.json`
- `web-model/metrics.json`

## 4. Deploy

After export, deploy the repository to Vercel. The frontend reads the model from `web-model/`.
