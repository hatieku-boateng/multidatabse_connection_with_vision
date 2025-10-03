# Multi-Database Connection with Vision

## Overview
This project combines computer vision and database connectivity to verify government ID cards. It provides a PyTorch-based embedding matcher that compares uploaded ID images against curated reference samples, a Streamlit UI for interactive verification, and a SQL Server connectivity demo that can be triggered after a successful match. The toolkit is designed for rapid experiments with small ID datasets and can serve as a foundation for building richer authentication workflows.

## Key Features
- ResNet18-based embedding matcher for drivers licence, Ghana card, and voter ID samples.
- Streamlit web app that guides users through uploading an ID card and viewing match metrics.
- Optional SQL Server connectivity check using `pyodbc` and `pandas` after a successful drivers licence match.
- Utility scripts for generating and refreshing CSV/Excel manifests of curated images.
- CLI fallback for running image classification outside of Streamlit.

## Project Layout
```
.
|-- id_cards/
|   |-- curated/                 # Reference images grouped by class
|   |   |-- drivers_licence/
|   |   |-- ghana_card/
|   |   `-- voter_id/
|   |-- manifests/               # Auto-generated CSV and Excel manifests
|   `-- splits/                  # (Optional) text files with dataset splits
|-- tools/
|   |-- id_card_matcher_ml.py    # Core embedding matcher and CLI entry point
|   |-- streamlit_login.py       # Streamlit app for visual verification
|   |-- sql_server_connect.py    # SQL Server connectivity demo
|   |-- create_labels_csv.py     # Build a shuffled CSV manifest
|   |-- create_labels-excel.py   # Build a shuffled Excel manifest
|   |-- update_labels_csv.py     # Refresh CSV manifest with new files
|   |-- update_labels_excel.py   # Refresh Excel manifest with new files
|   |-- folder_infer.py          # Quick torchvision dataset sanity check
|   `-- prepare_dataset.py       # Placeholder for future preprocessing
`-- tools/user_image_store/      # Sample images for testing
```

## Requirements
- Python 3.10 or newer (recommended for PyTorch builds).
- [PyTorch](https://pytorch.org/) with torchvision (CPU or GPU build).
- Python packages: `numpy`, `Pillow`, `streamlit`, `pandas`, `pyodbc` (for SQL demo), `torch`, `torchvision`.
- Optional: Microsoft ODBC Driver 17 (or later) for SQL Server to enable the `pyodbc` connection.

Install dependencies inside a virtual environment:
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # adjust for your platform
pip install streamlit numpy Pillow pandas pyodbc
```

## Preparing Curated References
1. Place representative ID images into `id_cards/curated/<class_name>/` for each supported class (`drivers_licence`, `ghana_card`, `voter_id`).
2. Ensure images are clear, cropped, and in common formats (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`).
3. (Optional) Generate manifests for quick inspection:
   ```bash
   python tools/create_labels_csv.py
   python tools/create_labels-excel.py
   ```
   The `update_*.py` variants keep existing manifests in sync after adding new images.

If any class folder is empty or missing, the matcher will raise `FileNotFoundError`. Verify the curated directory before running the app.

## Running the Embedding Matcher (CLI)
```bash
python tools/id_card_matcher_ml.py path/to/image.png --references id_cards/curated
```
The script prints the best matching class, the cosine-normalized L2 distance, and the runner-up distance. Smaller distances indicate closer matches. When launched without arguments, the script prompts for an image path using a file dialog (if `tkinter` is available) or falls back to console input.

## Launching the Streamlit Login Demo
```bash
streamlit run tools/streamlit_login.py
```
Within the app you can:
- Set the path to the curated reference directory (defaults to `id_cards/curated`).
- Upload an ID image and preview it in the UI.
- Adjust the match threshold slider (default 0.60).
- Optionally toggle the SQL Server connectivity check that runs after a verified drivers licence match.

The app displays the best-match label, the embedding distance, and the runner-up distance to help you judge classification confidence.

## SQL Server Connectivity Demo
The Streamlit app can run `tools/sql_server_connect.py` once a drivers licence image is verified. Update the connection string in that file to point to your SQL Server instance:
```python
server = r"YOUR_HOST\SQLEXPRESS"
database = "YOUR_DATABASE"
```
The script uses Windows authentication. Install the Microsoft ODBC Driver and ensure the target database is reachable. When executed, the script fetches the first five rows from the `drivers` table and displays them in Streamlit (or prints them when run standalone).

Run the script directly to test connectivity:
```bash
python tools/sql_server_connect.py
```

## Utility Scripts
- `tools/folder_infer.py`: Quick sanity check that torchvision can read the curated folder and infer class mappings.
- `tools/prepare_dataset.py`: Placeholder for future preprocessing steps.
- `id_cards/splits/*.txt`: Optional files you can use to store train/val/test filenames if you extend the pipeline for model training.

## Troubleshooting
- **Missing PyTorch or torchvision:** Install the correct wheel for your Python version and platform from the PyTorch website.
- **`FileNotFoundError` for curated references:** Ensure each expected subfolder (`drivers_licence`, `ghana_card`, `voter_id`) contains at least one image.
- **`pyodbc` errors:** Verify the ODBC Driver for SQL Server is installed and the connection string is accurate. On Windows, the driver typically appears as `ODBC Driver 17 for SQL Server`.
- **Slow inference:** Running on CPU is supported but slower. If you have a CUDA-capable GPU, install the matching PyTorch build and enable it automatically.

## Next Steps
- Expand the curated dataset for improved coverage and consider fine-tuning the backbone model.
- Replace the placeholder `prepare_dataset.py` with preprocessing routines or automated data augmentation.
- Extend the SQL demo to include write operations or integration with additional databases.
