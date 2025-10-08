import contextlib
import importlib.util
import importlib.metadata
from pathlib import Path

import streamlit as st

from id_card_matcher_ml import CURATED_ROOT, MatchResult, classify_id

SQL_CONNECT_SCRIPT = Path(__file__).resolve().parent / "sql_server_connect.py"
MATCH_THRESHOLD = 0.60  # tuned for ResNet18 embeddings with L2 distance on normalized vectors


def run_sql_server_demo() -> str:
    if not SQL_CONNECT_SCRIPT.exists():
        raise FileNotFoundError(f"SQL connector script not found at '{SQL_CONNECT_SCRIPT}'.")

    if importlib.util.find_spec('pyodbc') is None:
        raise ModuleNotFoundError("'pyodbc' is not installed in this environment.")

    import io
    import runpy

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        runpy.run_path(str(SQL_CONNECT_SCRIPT), run_name='__main__')
    return buffer.getvalue().strip()


def main() -> None:
    st.set_page_config(page_title="Image Login", page_icon="[img]", layout="centered")
    st.title("Government ID Verification")
    st.caption("Upload an ID card and we will compare it against curated drivers licence, Ghana card, and voter ID samples.")

    # Pre-warm torchvision weights on Streamlit Cloud so first inference is fast.
    # This is cached and safe if torchvision isn't available (e.g., partial installs).
    @st.cache_resource(show_spinner=False)
    def _prewarm_model() -> bool:
        try:
            from torchvision.models import ResNet18_Weights, resnet18  # type: ignore
            weights = ResNet18_Weights.IMAGENET1K_V1
            _ = resnet18(weights=weights).eval()
            return True
        except Exception:
            return False

    with st.spinner("Preparing model (first run may take a moment)..."):
        _prewarm_model()

    # Environment diagnostics + self-heal option if PyTorch is missing on Cloud.
    with st.expander("Environment Check", expanded=False):
        pyver = importlib.metadata.version
        def _ver(pkg: str) -> str:
            try:
                return pyver(pkg)
            except Exception:
                return "not installed"

        st.write({
            "python": _ver("pip").split(" ")[0] if True else "",
            "torch": _ver("torch"),
            "torchvision": _ver("torchvision"),
            "numpy": _ver("numpy"),
            "pillow": _ver("Pillow"),
            "pandas": _ver("pandas"),
            "streamlit": _ver("streamlit"),
        })

        if importlib.util.find_spec('torch') is None or importlib.util.find_spec('torchvision') is None:
            st.warning("PyTorch or torchvision is missing. If this is Streamlit Cloud and the build skipped, install CPU wheels below.")
            if st.button("Install PyTorch CPU now", type="secondary"):
                import subprocess, sys
                cmd = [sys.executable, "-m", "pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cpu", "torch==2.4.1", "torchvision==0.19.1"]
                with st.spinner("Installing torch + torchvision (CPU)... this may take a few minutes"):
                    try:
                        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                        st.code(out, language="bash")
                        st.success("Installed. Click 'Rerun' from the Streamlit menu.")
                    except subprocess.CalledProcessError as exc:
                        st.error("Install failed. See logs below.")
                        st.code(exc.output or str(exc), language="bash")

    reference_root_str = st.text_input(
        "Curated reference folder",
        value=str(CURATED_ROOT),
        help="Folder must contain subfolders drivers_licence, ghana_card, and voter_id.",
    )
    reference_root = Path(reference_root_str).expanduser().resolve()

    uploaded_image = st.file_uploader(
        "Upload your ID image",
        type=["png", "jpg", "jpeg", "bmp", "gif"],
        accept_multiple_files=False,
    )

    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        st.image(image_bytes, caption="Uploaded image", use_container_width=True)
    else:
        image_bytes = None

    col1, col2 = st.columns([1, 1])
    with col1:
        threshold = st.slider(
            "Match threshold",
            min_value=0.10,
            max_value=1.50,
            value=MATCH_THRESHOLD,
            step=0.05,
            help="Lower values require closer matches. Typical matches fall below ~0.6.",
        )
    with col2:
        has_pyodbc = importlib.util.find_spec('pyodbc') is not None
        run_sql = st.toggle(
            "Run SQL check when Drivers Licence matches",
            value=has_pyodbc,
            help="Runs tools/sql_server_connect.py after a successful Drivers Licence match.",
        )

    if st.button("Log in", type="primary"):
        if image_bytes is None:
            st.error("Upload an ID image before logging in.")
            return

        try:
            result: MatchResult = classify_id(image_bytes, reference_root=reference_root)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except ModuleNotFoundError as exc:
            st.error(f"Missing dependency: {exc}")
            return
        except Exception as exc:  # pragma: no cover - safety net
            st.error(f"Classification failed: {exc}")
            return

        st.metric("Best match distance", f"{result.distance:.3f}", help="Lower numbers mean closer matches.")
        if result.runner_up_distance < float("inf"):
            st.caption(f"Runner-up distance: {result.runner_up_distance:.3f}")

        if result.distance <= threshold:
            st.success(f"Verification passed. Best match: {result.label}.")
            if run_sql and result.label.lower() == "drivers licence":
                st.info("Drivers licence detected. Running SQL Server connectivity check...")
                try:
                    output = run_sql_server_demo()
                    if output:
                        st.code(output, language="text")
                    else:
                        st.write("SQL script executed but returned no output.")
                except Exception as exc:
                    st.error(f"SQL Server check failed: {exc}")
        else:
            st.error("Verification failed. Uploaded ID does not closely match the curated references.")
            st.info(f"Closest detected class: {result.label} (distance {result.distance:.3f}).")


if __name__ == "__main__":
    main()
