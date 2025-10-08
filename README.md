# Multi-Database Connection with Vision (Clean Start)

A minimal setup for verifying ID images against curated references using PyTorch embeddings. Includes a simple CLI and an optional Streamlit UI.

## Prerequisites
- Python 3.10+
- Git (optional)

## Setup (fresh virtual environment)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision pillow numpy streamlit
```

If you prefer CPU-only wheels for PyTorch, add the extra index:
```powershell
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
```

## Data
Curated reference images live under `id_cards/curated/` with subfolders:
- `drivers_licence/`
- `ghana_card/`
- `voter_id/`

Populate each with representative samples (PNG/JPEG).

## Run the CLI
```powershell
python tools/id_card_matcher_ml.py path\to\your_image.jpg
```

## Run the Streamlit App
```powershell
streamlit run tools/streamlit_login.py
```

Use the "Curated reference folder" input to point to your `id_cards/curated` directory if needed.

## Notes
- The repo has been cleaned of environment checks and auxiliary SQL demo code.
- No automatic package installation occurs in the UI; manage dependencies via pip.
