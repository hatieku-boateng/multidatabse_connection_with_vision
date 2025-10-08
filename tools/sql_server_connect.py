import contextlib
import pyodbc
import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:  # allow running outside Streamlit
    st = None

# Replace with your actual server and database
server = r"HAB\SQLEXPRESS"     # e.g., "DESKTOP-123ABC\SQLEXPRESS"
database = "dvla"                # your database name

# Connection string for Windows Authentication
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={server};"
    f"DATABASE={database};"
    "Trusted_Connection=yes;"
)


def notify(message: str) -> None:
    if st is not None:
        st.info(message)
    else:
        print(message)


def show_success(message: str) -> None:
    if st is not None:
        st.success(message)
    else:
        print(message)


def show_error(message: str) -> None:
    if st is not None:
        st.error(message)
    else:
        print(message)


def show_table(df: pd.DataFrame, title: str = "Query Results") -> None:
    if st is not None:
        st.markdown("---")
        st.subheader(title)
        st.caption(f"Showing {len(df)} rows")
        height = min(600, 120 + len(df) * 32)
        left_pad, centre, right_pad = st.columns([1, 6, 1])
        with centre:
            st.dataframe(df, use_container_width=True, height=height)
    else:
        print(df)


def main() -> None:
    try:
        notify("Connecting to DVLA Database...")
        with contextlib.closing(pyodbc.connect(conn_str)) as conn:
            show_success("Connected successfully to DVLA Database.")
            query = "SELECT TOP 5 * FROM drivers;"
            df = pd.read_sql(query, conn)
            show_success("Retrieved data. Displaying interactive table below.")
            show_table(df, title="Drivers Table Preview")
    except Exception as exc:  # broad catch to surface errors in Streamlit UI
        show_error(f"Connection failed: {exc}")


if __name__ == "__main__":
    main()

