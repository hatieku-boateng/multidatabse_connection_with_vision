import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    st = None

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torchvision.models import ResNet18_Weights, resnet18
except ModuleNotFoundError as exc:  # pragma: no cover - surface install issue in runtime path
    torch = None
    Tensor = None  # type: ignore
    nn = None
    F = None
    ResNet18_Weights = None  # type: ignore
    resnet18 = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

CURATED_ROOT = (Path(__file__).resolve().parent.parent / "id_cards" / "curated").resolve()
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
EXPECTED_CLASS_LABELS: Dict[str, str] = {
    "drivers_licence": "Drivers Licence",
    "ghana_card": "Ghana Card",
    "voter_id": "Voter ID",
}


@dataclass
class MatchResult:
    label: str
    distance: float
    runner_up_distance: float


class IDCardEmbeddingMatcher:
    def __init__(self, reference_root: Path = CURATED_ROOT) -> None:
        if torch is None or resnet18 is None or nn is None:
            details = f" Underlying import error: {TORCH_IMPORT_ERROR}" if TORCH_IMPORT_ERROR else ""
            raise ModuleNotFoundError(
                "PyTorch with torchvision is required. Install via 'pip install torch torchvision'." + details
            ) from TORCH_IMPORT_ERROR

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.preprocess = weights.transforms()
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(self.device)

        self.reference_root = reference_root
        self.references = self._load_reference_embeddings()
        if not self.references:
            raise FileNotFoundError(
                f"No reference images found under '{self.reference_root}'. "
                "Ensure curated drivers licence, Ghana card, and voter ID samples are present."
            )

    def _load_reference_embeddings(self) -> Dict[str, List[np.ndarray]]:
        refs: Dict[str, List[np.ndarray]] = {}
        missing: List[str] = []

        for folder_name, friendly_label in EXPECTED_CLASS_LABELS.items():
            class_dir = self.reference_root / folder_name
            if not class_dir.is_dir():
                missing.append(friendly_label)
                continue

            embeddings: List[np.ndarray] = []
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue
                try:
                    embedding = self._embed_image(path)
                except Exception:
                    continue
                embeddings.append(embedding)

            if embeddings:
                refs[friendly_label] = embeddings
            else:
                missing.append(friendly_label)

        if missing:
            raise FileNotFoundError(
                "Curated reference images missing for: " + ", ".join(sorted(missing))
            )

        return refs

    def _embed_image(self, source: Path | io.BytesIO | bytes) -> np.ndarray:
        if isinstance(source, Path):
            image = Image.open(source)
        elif isinstance(source, io.BytesIO):
            image = Image.open(source)
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(source))
        else:
            raise TypeError("Unsupported image source type.")

        image = image.convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding: Tensor = self.model(tensor)  # type: ignore[assignment]
        embedding = F.normalize(embedding, dim=1)
        return embedding.squeeze(0).cpu().numpy()

    def match(self, candidate_bytes: bytes) -> MatchResult:
        candidate_embedding = self._embed_image(candidate_bytes)
        best_label = "Unknown"
        best_distance = float("inf")
        runner_up = float("inf")
        for label, embeddings in self.references.items():
            class_distance = min(np.linalg.norm(embedding - candidate_embedding) for embedding in embeddings)
            if class_distance < best_distance:
                runner_up = best_distance
                best_distance = class_distance
                best_label = label
            elif class_distance < runner_up:
                runner_up = class_distance
        return MatchResult(best_label, best_distance, runner_up)


@lru_cache(maxsize=2)
def _matcher_for(reference_root: str) -> IDCardEmbeddingMatcher:
    return IDCardEmbeddingMatcher(Path(reference_root))


if st is not None:
    @st.cache_resource(show_spinner=False)
    def _streamlit_matcher(root: str) -> IDCardEmbeddingMatcher:
        return IDCardEmbeddingMatcher(Path(root))
else:
    _streamlit_matcher = None


def _resolve_matcher(reference_root: Path) -> IDCardEmbeddingMatcher:
    root_str = str(reference_root.resolve())
    if _streamlit_matcher is not None:
        return _streamlit_matcher(root_str)
    return _matcher_for(root_str)


def classify_id(image_bytes: bytes, reference_root: Path | None = None) -> MatchResult:
    root = (reference_root or CURATED_ROOT).resolve()
    matcher = _resolve_matcher(root)
    return matcher.match(image_bytes)


def render_streamlit_app() -> None:
    assert st is not None
    st.set_page_config(page_title="ID Matcher", page_icon="??", layout="centered")
    st.title("Government ID Matcher (PyTorch)")
    st.caption(
        "Upload a card image and compare it against curated references for drivers licences, Ghana cards, and voter IDs."
    )

    reference_root_str = st.text_input(
        "Curated reference folder",
        value=str(CURATED_ROOT),
        help="Folder must contain subfolders drivers_licence, ghana_card, and voter_id.",
    )
    reference_root = Path(reference_root_str).expanduser().resolve()

    try:
        matcher = _resolve_matcher(reference_root)
        st.success("Reference embeddings loaded successfully.")
    except Exception as exc:
        st.error(f"Unable to load references: {exc}")
        return

    uploaded_file = st.file_uploader(
        "Upload ID image",
        type=[ext.strip('.') for ext in ALLOWED_EXTENSIONS],
        accept_multiple_files=False,
        help="Click to upload the ID card image you want to classify.",
    )

    if uploaded_file is None:
        st.info("Choose an ID image to begin.")
        return

    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Uploaded image", use_container_width=True)

    threshold = st.slider(
        "Match threshold",
        min_value=0.10,
        max_value=1.50,
        value=0.60,
        step=0.05,
        help="Lower values require closer matches. Typical matches fall below ~0.6.",
    )

    if st.button("Classify", type="primary"):
        try:
            result = matcher.match(image_bytes)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            st.error(f"Failed to classify image: {exc}")
            return

        st.metric("Best match distance", f"{result.distance:.4f}")
        if result.runner_up_distance < float("inf"):
            st.caption(f"Runner up distance: {result.runner_up_distance:.4f}")

        if result.distance <= threshold:
            st.success(f"Prediction: {result.label}")
        else:
            st.error("The uploaded image does not closely match any curated references.")
            st.info(f"Closest class: {result.label} (distance {result.distance:.4f}).")


def _running_with_streamlit() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return bool(os.environ.get("STREAMLIT_SERVER_RUNNING"))


def _prompt_for_image_path() -> Path | None:
    try:
        from tkinter import Tk, filedialog
    except Exception:
        raw = input("Enter the path to the ID image: ").strip()
        return Path(raw) if raw else None

    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Select ID image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")],
    )
    root.destroy()
    if not filename:
        return None
    return Path(filename)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Classify an ID image using PyTorch embeddings")
    parser.add_argument("image", nargs="?", type=Path, help="Path to the ID image to classify")
    parser.add_argument(
        "--references",
        type=Path,
        default=CURATED_ROOT,
        help="Path to curated reference root (defaults to id_cards/curated)",
    )
    args = parser.parse_args()

    image_path = args.image or _prompt_for_image_path()
    if image_path is None:
        print("No image selected. Exiting.")
        return

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    with image_path.open("rb") as handle:
        result = classify_id(handle.read(), reference_root=args.references)
    print(f"Best match: {result.label} (distance {result.distance:.4f})")
    if result.runner_up_distance < float("inf"):
        print(f"Runner up distance: {result.runner_up_distance:.4f}")


if __name__ == "__main__":
    if _running_with_streamlit():
        render_streamlit_app()
    else:
        _cli()
