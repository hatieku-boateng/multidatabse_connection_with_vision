from pathlib import Path

import streamlit as st

from id_card_matcher_ml import CURATED_ROOT, MatchResult, classify_id

SQL_CONNECT_SCRIPT = Path(__file__).resolve().parent / "sql_server_connect.py"
MATCH_THRESHOLD = 0.60  # tuned for ResNet18 embeddings with L2 distance on normalized vectors


def run_sql_server_demo() -> str:
    if not SQL_CONNECT_SCRIPT.exists():
        raise FileNotFoundError(f"SQL connector script not found at '{SQL_CONNECT_SCRIPT}'.")

    import io
    import runpy
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        runpy.run_path(str(SQL_CONNECT_SCRIPT), run_name='__main__')
    return buffer.getvalue().strip()


def main() -> None:
    st.set_page_config(page_title="Image Login", page_icon="[img]", layout="centered")
    st.title("Government ID Verification")
    st.caption("Upload an ID card and we will compare it against curated drivers licence, Ghana card, and voter ID samples.")


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
        # Default the toggle based on whether pyodbc appears installed
        has_pyodbc = False
        try:
            import importlib.util
            has_pyodbc = importlib.util.find_spec('pyodbc') is not None
        except Exception:
            has_pyodbc = False
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
