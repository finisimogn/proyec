import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

st.set_page_config(
    page_title="Analizador de Series de Fourier", layout="wide"
)

st.title("📊 Analizador y Modificador de Series de Fourier")
st.markdown(
    "Ingresa una función y sus rangos para calcular, observar, comparar y modificar su aproximación por Series de Fourier."
)

# --- PANEL LATERAL: ENTRADA DE DATOS ---
st.sidebar.header("1. Configuración de la Función")

# Entrada de la función
func_str = st.sidebar.text_input(
    "Función f(x):",
    value="x**2",
    help="Usa sintaxis de Python/SymPy. Ejemplos: x, x**2, sin(x), exp(-x), sign(x)",
)

# Rangos (CORREGIDO: uso correcto de columnas en sidebar)
col1, col2 = st.sidebar.columns(2)
with col1:
    a = st.number_input("Límite inferior (a):", value=-float(np.pi))
with col2:
    b = st.number_input("Límite superior (b):", value=float(np.pi))

st.sidebar.header("2. Parámetros de Fourier")
N_harmonics = st.sidebar.slider(
    "Número de armónicos (N):", min_value=1, max_value=50, value=10
)

# Opción para modificar coeficientes manualmente
st.sidebar.header("3. Modificación de Armónicos")
modify_coeffs = st.sidebar.checkbox("Modificar coeficientes manualmente")


# --- CÁLCULO DE LA SERIE DE FOURIER ---
def calculate_fourier(func_input, a_val, b_val, n_terms):
    x = sp.Symbol("x")
    try:
        f_expr = sp.sympify(func_input)
    except Exception as e:
        st.error(f"Error en la expresión matemática: {e}")
        return None

    L = (b_val - a_val) / 2.0
    x_mid = (a_val + b_val) / 2.0

    # Coeficiente a0
    try:
        a0_integral = sp.integrate(f_expr, (x, a_val, b_val))
        a0 = float((1.0 / L) * a0_integral.evalf())
    except Exception:
        a0 = 0.0

    an_list = []
    bn_list = []

    # Cálculo acelerado de coeficientes an y bn
    for n in range(1, n_terms + 1):
        try:
            expr_an = f_expr * sp.cos(n * sp.pi * (x - x_mid) / L)
            an_val = float((1.0 / L) * sp.integrate(expr_an, (x, a_val, b_val)).evalf())
        except Exception:
            an_val = 0.0

        try:
            expr_bn = f_expr * sp.sin(n * sp.pi * (x - x_mid) / L)
            bn_val = float((1.0 / L) * sp.integrate(expr_bn, (x, a_val, b_val)).evalf())
        except Exception:
            bn_val = 0.0

        an_list.append(an_val)
        bn_list.append(bn_val)

    return f_expr, a0, an_list, bn_list, L, x_mid


if a >= b:
    st.error("El límite inferior (a) debe ser menor que el límite superior (b).")
else:
    result = calculate_fourier(func_str, a, b, N_harmonics)

    if result:
        f_expr, a0, an_list, bn_list, L, x_mid = result

        # Modificación manual de coeficientes si el usuario lo activa
        if modify_coeffs:
            st.sidebar.subheader("Ajuste Manual")
            a0 = st.sidebar.number_input("Modificar a0:", value=a0)
            for i in range(min(5, N_harmonics)):
                an_list[i] = st.sidebar.number_input(
                    f"Modificar a_{i+1}:", value=an_list[i]
                )
                bn_list[i] = st.sidebar.number_input(
                    f"Modificar b_{i+1}:", value=bn_list[i]
                )

        # --- EVALUACIÓN NUMÉRICA ---
        x_vals = np.linspace(a - L, b + L, 1000)
        f_num = sp.lambdify(sp.Symbol("x"), f_expr, "numpy")

        try:
            y_original = f_num(x_vals)
            if np.isscalar(y_original):
                y_original = np.full_like(x_vals, y_original)
        except Exception:
            y_original = np.zeros_like(x_vals)

        # Reconstrucción de la Serie de Fourier
        fourier_series = np.full_like(x_vals, a0 / 2.0)
        for n in range(1, N_harmonics + 1):
            fourier_series += an_list[n - 1] * np.cos(
                n * np.pi * (x_vals - x_mid) / L
            ) + bn_list[n - 1] * np.sin(n * np.pi * (x_vals - x_mid) / L)

        # --- VISUALIZACIÓN Y COMPARACIÓN ---
        st.subheader("📈 Observar y Comparar: Función Original vs. Serie de Fourier")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_original,
                mode="lines",
                name="Función Original f(x)",
                line=dict(color="blue", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=fourier_series,
                mode="lines",
                name=f"Aproximación de Fourier (N={N_harmonics})",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            xaxis_title="x",
            yaxis_title="f(x)",
            margin=dict(l=20, r=20, t=30, b=20),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width="stretch")

        # --- ANÁLISIS DE COEFICIENTES Y ERROR ---
        st.subheader("🔍 Análisis Métrico y Coeficientes")

        col_left, col_right = st.columns(2)

        with col_left:
            st.write("**Coeficientes Calculados:**")
            st.write(f"**a₀ / 2:** `{a0 / 2.0:.4f}`")
            coeffs_data = {
                "n": list(range(1, N_harmonics + 1)),
                "a_n (Cosenos)": [round(val, 4) for val in an_list],
                "b_n (Senos)": [round(val, 4) for val in bn_list],
            }
            st.dataframe(coeffs_data, height=250)

        with col_right:
            st.write("**Espectro de Amplitud (Magnitud de Armónicos):**")
            magnitudes = [
                float(np.sqrt(a**2 + b**2)) for a, b in zip(an_list, bn_list)
            ]

            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=list(range(1, N_harmonics + 1)),
                        y=magnitudes,
                        marker_color="purple",
                    )
                ]
            )
            fig_bar.update_layout(
                xaxis_title="Armónico (n)",
                yaxis_title="Magnitud √(aₙ² + bₙ²)",
                margin=dict(l=20, r=20, t=30, b=20),
                template="plotly_white",
            )
            st.plotly_chart(fig_bar, use_container_width="stretch")