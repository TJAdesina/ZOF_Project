import streamlit as st
import pandas as pd
import ZOF_CLI as zof


# -------------------------------------------
# PAGE SETTINGS
# -------------------------------------------
st.set_page_config(page_title="ZOF Solver", layout="centered")

st.title("📘 Zero of Functions (ZOF) Solver")
st.write("Select a method, enter parameters, and compute the root.")


# -------------------------------------------
# METHOD SELECTION
# -------------------------------------------
methods = [
    "Bisection Method",
    "Regula Falsi (False Position)",
    "Secant Method",
    "Newton–Raphson Method",
    "Fixed Point Iteration",
    "Modified Secant Method"
]

method = st.selectbox("Choose a Numerical Method", methods)


# -------------------------------------------
# INPUT FIELDS
# -------------------------------------------
function = st.text_input("Enter function f(x):", "x**2 - 4")

col1, col2 = st.columns(2)

with col1:
    x0 = st.number_input("x0 (Initial guess / lower bound)", value=0.0)

with col2:
    x1 = st.number_input("x1 (Second guess / upper bound)", value=2.0)

tol = st.number_input("Tolerance", value=1e-6, format="%.8f")
max_iter = st.number_input("Max Iterations", value=20, step=1)


# Extra input for Modified Secant
delta = None
if method == "Modified Secant Method":
    delta = st.number_input("Delta (e.g. 0.01)", value=0.01)


st.markdown("---")


# -------------------------------------------
# RUN BUTTON
# -------------------------------------------
if st.button("Compute"):

    try:
        # RUN THE SELECTED METHOD
        if method == "Bisection Method":
            rows = zof.bisection(function, x0, x1, tol, max_iter)
            df = pd.DataFrame(rows, columns=["Iter", "a", "b", "c", "f(a)", "f(c)", "Error"])

        elif method == "Regula Falsi (False Position)":
            rows = zof.regula_falsi(function, x0, x1, tol, max_iter)
            df = pd.DataFrame(rows, columns=["Iter", "a", "b", "c", "f(a)", "f(b)", "f(c)", "Error"])

        elif method == "Secant Method":
            rows = zof.secant(function, x0, x1, tol, max_iter)
            df = pd.DataFrame(rows, columns=["Iter", "x0", "x1", "f(x0)", "f(x1)", "x_new", "Error"])

        elif method == "Newton–Raphson Method":
            rows = zof.newton_raphson(function, x0, tol, max_iter)
            df = pd.DataFrame(rows, columns=["Iter", "x", "f(x)", "f'(x)", "x_new", "Error"])

        elif method == "Fixed Point Iteration":
            rows = zof.fixed_point(function, x0, tol, max_iter)
            df = pd.DataFrame(rows, columns=["Iter", "x_old", "x_new", "Error"])

        elif method == "Modified Secant Method":
            rows = zof.modified_secant(function, x0, delta, tol, max_iter)
            df = pd.DataFrame(rows, columns=["Iter", "x", "f(x)", "Derivative", "x_new", "Error"])


        # -------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------
        st.subheader("📊 Iteration Table")
        st.dataframe(df, use_container_width=True)

        # Show final root
        final_root = df.iloc[-1, -2]   # x_new column
        st.success(f"Estimated Root: **{final_root}**")

        # CSV DOWNLOAD
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "ZOF_results.csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
