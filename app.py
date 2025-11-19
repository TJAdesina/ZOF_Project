import streamlit as st
import importlib
import ZOF_CLI as zof

st.set_page_config(page_title="ZOF Solver", layout="centered")

st.title("📘 Zero-Order, First-Order & Newton-Raphson Solver")

# Reload module dynamically (useful while editing)
importlib.reload(zof)

st.subheader("Enter Function")
function = st.text_input("f(x) =", "x**2 - 4")

st.subheader("Initial Values")
x0 = st.number_input("Initial guess x0:", value=1.0)

st.write("Optional Inputs (Only used by some methods)")
x1 = st.number_input("Second guess x1 (for methods that need it):", value=2.0)
max_iter = st.number_input("Maximum iterations:", value=20)
tol = st.number_input("Error tolerance:", value=1e-6)

st.subheader("Choose Method")
method = st.selectbox(
    "Method",
    [
        "Zero-Order (Trial-and-Error)",
        "First-Order (Bisection / False Position)",
        "Newton-Raphson"
    ]
)

if st.button("Compute"):
    st.write("### 🔍 Results")

    try:
        if method == "Zero-Order (Trial-and-Error)":
            output = zof.zero_order_method(function, x0, x1, max_iter)
            st.code(output)

        elif method == "First-Order (Bisection / False Position)":
            output = zof.first_order_method(function, x0, x1, max_iter)
            st.code(output)

        elif method == "Newton-Raphson":
            output = zof.newton_raphson(function, x0, max_iter, tol)
            st.code(output)

    except Exception as e:
        st.error(f"Error: {e}")
