import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.integrate as integrate
import streamlit as st
import sympy as sp

# Configuración de la página
st.set_page_config(page_title="Calculadora de Fourier", layout="wide")

st.title("⚡ Calculadora de Serie de Fourier High-Performance")
st.markdown("Potenciada con **SciPy**, **NumPy**, **SymPy** y **Plotly** para $N > 1000$ armónicos.")

# --- PANEL LATERAL ---
st.sidebar.header("1. Configuración de la Función")
var_choice = st.sidebar.radio("Variable independiente:", ["x", "t"], horizontal=True)

func_str = st.sidebar.text_input(
    f"Función f({var_choice}):",
    value=f"{var_choice}",
    help="Usa sintaxis estándar de Python/SymPy. Ej: x, x**2, sin(x), pi*x"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    a_str = st.sidebar.text_input("Límite inferior (a):", value="-pi")
with col2:
    b_str = st.sidebar.text_input("Límite superior (b):", value="pi")

axis_format = st.sidebar.radio(
    "Formato de ejes:",
    ["Múltiplos de Pi (ej. -π, 0, π)", "Decimales Directos"],
    index=0
)

st.sidebar.header("2. Parámetros de Fourier")
N_harmonics = st.sidebar.slider(
    "Número de armónicos (N):", min_value=1, max_value=1000, value=50, step=10
)

# --- PARSEO SIMBÓLICO Y CONVERSIÓN ---
v_sym = sp.Symbol(var_choice)
n_sym = sp.Symbol('n', integer=True, positive=True)

try:
    a_sym = sp.sympify(a_str.replace("π", "pi"))
    b_sym = sp.sympify(b_str.replace("π", "pi"))
    a_val = float(a_sym.evalf())
    b_val = float(b_sym.evalf())
    f_expr = sp.sympify(func_str.replace("π", "pi"))
except Exception as e:
    st.error(f"Error al procesar la función o límites: {e}")
    st.stop()

if a_val >= b_val:
    st.error("El límite inferior (a) debe ser menor que el límite superior (b).")
    st.stop()

T = b_val - a_val
L = T / 2.0
w0 = (2.0 * np.pi) / T

# --- CÁLCULO CIENTÍFICO VECTORIZADO ---
@st.cache_data(show_spinner=False)
def compute_fourier_engine(clean_expr_str, var_name, a_v, b_v, n_terms):
    v = sp.Symbol(var_name)
    f = sp.sympify(clean_expr_str)
    
    # 1. Detección de Simetría con SymPy
    is_even = sp.simplify(f.subs(v, -v) - f) == 0
    is_odd = sp.simplify(f.subs(v, -v) + f) == 0
    
    # 2. Intento de Integración Analítica Exacta (SymPy)
    w0_sym = 2 * sp.pi / (b_sym - a_sym)
    try:
        a0_sym = (2 / (b_sym - a_sym)) * sp.integrate(f, (v, a_sym, b_sym)) if not is_odd else sp.Integer(0)
        an_sym = (2 / (b_sym - a_sym)) * sp.integrate(f * sp.cos(n_sym * w0_sym * v), (v, a_sym, b_sym)) if not is_odd else sp.Integer(0)
        bn_sym = (2 / (b_sym - a_sym)) * sp.integrate(f * sp.sin(n_sym * w0_sym * v), (v, a_sym, b_sym)) if not is_even else sp.Integer(0)
    except Exception:
        a0_sym, an_sym, bn_sym = None, None, None

    # 3. Muestreo Vectorial en NumPy
    f_num = sp.lambdify(v, f, modules=["numpy", {"sign": np.sign}])
    
    def safe_f(val):
        res = f_num(val)
        return res if isinstance(res, np.ndarray) else np.full_like(val, res)

    # Coeficiente a0
    if is_odd:
        a0_val = 0.0
    else:
        a0_int, _ = integrate.quad(lambda x: float(f_num(x)), a_v, b_v, limit=200)
        a0_val = (2.0 / T) * a0_int

    # Coeficientes an y bn vectorizados con NumPy
    n_vec = np.arange(1, n_terms + 1)
    x_samp = np.linspace(a_v, b_v, 5000)
    y_samp = safe_f(x_samp)

    if is_odd:
        an_vals = np.zeros(n_terms)
    else:
        cos_mat = np.cos(n_vec.reshape(-1, 1) * w0 * x_samp)
        an_vals = (2.0 / T) * np.trapezoid(y_samp * cos_mat, x_samp, axis=1)

    if is_even:
        bn_vals = np.zeros(n_terms)
    else:
        sin_mat = np.sin(n_vec.reshape(-1, 1) * w0 * x_samp)
        bn_vals = (2.0 / T) * np.trapezoid(y_samp * sin_mat, x_samp, axis=1)

    return a0_val, an_vals, bn_vals, is_even, is_odd, a0_sym, an_sym, bn_sym

a0_calc, an_calc, bn_calc, is_even, is_odd, a0_sym, an_sym, bn_sym = compute_fourier_engine(
    str(f_expr), var_choice, a_val, b_val, N_harmonics
)

# Reconstrucción Vectorial Ultra Rápida con NumPy (Broadcasting 2D)
x_plot = np.linspace(a_val, b_val, 2000)
f_num_plot = sp.lambdify(v_sym, f_expr, modules=["numpy", {"sign": np.sign}])
y_orig = f_num_plot(x_plot)
if np.isscalar(y_orig):
    y_orig = np.full_like(x_plot, y_orig)

n_mat = np.arange(1, N_harmonics + 1).reshape(-1, 1)
cos_terms = np.cos(n_mat * w0 * x_plot)
sin_terms = np.sin(n_mat * w0 * x_plot)

fourier_series = (a0_calc / 2.0) + np.sum(
    an_calc.reshape(-1, 1) * cos_terms + bn_calc.reshape(-1, 1) * sin_terms, axis=0
)

# --- VISUALIZACIÓN INTERACTIVA ---
tab1, tab2, tab3, tab4 = st.tabs([
    " Gráfica Principal", 
    " Desarrollo Analítico Paso a Paso", 
    " Tabla de Coeficientes", 
    " Espectro de Potencia"
])

# --- TAB 1: GRÁFICA ---
with tab1:
    st.subheader(f"Serie de Fourier: f({var_choice}) = {func_str}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=y_orig, mode="lines", name=f"f({var_choice}) Original", line=dict(color="#1f77b4", width=2.5)))
    fig.add_trace(go.Scatter(x=x_plot, y=fourier_series, mode="lines", name=f"Serie de Fourier (N={N_harmonics})", line=dict(color="#d62728", width=2, dash="dash")))

    xaxis_config = dict(title=f"Variable {var_choice}")
    if "Múltiplos de Pi" in axis_format:
        xaxis_config.update(dict(
            tickmode="array",
            tickvals=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
            ticktext=["-π", "-π/2", "0", "π/2", "π"]
        ))

    fig.update_layout(
        xaxis=xaxis_config,
        yaxis_title="Amplitud",
        template="plotly_white",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: PASO A PASO ---
with tab2:
    st.subheader("SOLUCIÓN DE LA SERIE DE FOURIER LISTA PARA COPIAR")
    st.code("==========================================================================")
    
    sim_type = "PAR" if is_even else ("IMPAR" if is_odd else "SIN SIMETRÍA")
    st.text(f"Función seleccionada: f({var_choice}) = {func_str}")
    st.text(f"Intervalo de integración: [{a_val:.6f}, {b_val:.6f}]")
    
    st.markdown("#### PASO 1. IDENTIFICAR EL PERIODO")
    st.text(f"T = {T:.10f}")
    st.text(f"El intervalo usado para integrar es [-T/2, T/2] = [{a_val:.10f}, {b_val:.10f}]")
    
    st.markdown("#### PASO 2. CALCULAR LA FRECUENCIA ANGULAR FUNDAMENTAL")
    st.text(f"omega_0 = 2*pi / T = {w0:.6f} rad/unidad")

    st.markdown("#### PASO 3. ESCRIBIR LA FORMA GENERAL")
    st.latex(f"f({var_choice}) = \\frac{{a_0}}{{2}} + \\sum_{{n=1}}^{{\\infty}} \\left[ a_n \\cos(n \\omega_0 {var_choice}) + b_n \\sin(n \\omega_0 {var_choice}) \\right]")

    st.markdown("#### PASO 4. ANALIZAR LA SIMETRÍA")
    st.text(f"Simetría detectada: {sim_type}")
    if is_odd:
        st.text("La función satisface f(-x) = -f(x). Componente DC a0 = 0 y an = 0.")
    elif is_even:
        st.text("La función satisface f(-x) = f(x). Coeficientes bn = 0.")

    st.markdown("#### PASO 5. FORMULACIÓN ANALÍTICA EXACTA (SYMPY)")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.latex(f"a_0 = {sp.latex(a0_sym) if a0_sym is not None else round(a0_calc, 6)}")
    with col_b:
        st.latex(f"a_n = {sp.latex(an_sym) if an_sym is not None else '0'}")
    with col_c:
        st.latex(f"b_n = {sp.latex(bn_sym) if bn_sym is not None else '0'}")

# --- TAB 3: TABLA PANDAS ---
with tab3:
    st.subheader("Coeficientes Calculados")
    df_coeffs = pd.DataFrame({
        "Armónico (n)": np.arange(1, N_harmonics + 1),
        "a_n (Cosenos)": an_calc,
        "b_n (Senos)": bn_calc,
        "Magnitud |C_n|": np.sqrt(an_calc**2 + bn_calc**2)
    })
    st.dataframe(df_coeffs, height=400, use_container_width=True)

# --- TAB 4: ESPECTRO ---
with tab4:
    st.subheader("Espectro de Magnitud |C_n|")
    magnitudes = np.sqrt(an_calc**2 + bn_calc**2)
    fig_spec = go.Figure(data=[go.Bar(x=list(range(1, N_harmonics + 1)), y=magnitudes, marker_color="#6366f1")])
    fig_spec.update_layout(xaxis_title="Número de Armónico (n)", yaxis_title="Amplitud", template="plotly_white")
    st.plotly_chart(fig_spec, use_container_width=True)
