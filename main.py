"""
Legacy entry point kept for compatibility with the original project layout.

The desktop Tkinter application has been replaced by a Vercel-friendly web app.
Use the HTML frontend from the repository root and the scripts in `training/`
to download data from Kaggle, train the classifier, and export the browser model.
"""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    print("Desktop UI has been retired.")
    print(f"Open {root / 'index.html'} in a browser for the new web UI.")
    print(f"Use the training scripts in {root / 'training'} to train and export the model.")


if __name__ == "__main__":
    main()
