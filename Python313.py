import re
import numpy as np
import plotly.graph_objects as go
import scipy.integrate as integrate
import streamlit as st
import sympy as sp

st.set_page_config(page_title="Calculadora de Fourier", layout="wide")

st.title("Calculadora de Fourier")
st.markdown(
    "Herramienta interactiva para el cálculo, análisis de error y modificación de Series de Fourier en el dominio del tiempo."
)

# --- FUNCIONES DE LIMPIEZA Y PROCESAMIENTO DE CADENAS ---
def clean_expression(expr_str):
    """Reemplaza caracteres como π por pi y normaliza la sintaxis."""
    if not expr_str:
        return "0"
    s = str(expr_str).replace("π", "pi")
    # Permite sintaxis implícita como 2pi -> 2*pi
    s = re.sub(r'(\d)\s*pi', r'\1*pi', s)
    return s

# --- PANEL LATERAL: ENTRADA DE DATOS ---
st.sidebar.header("1. Configuración de la Función")

func_input_raw = st.sidebar.text_input(
    "Función f(t):",
    value="t",
    help="Usa sintaxis de SymPy. Ejemplos: t, t**2, sin(t), exp(-t), sign(t), pi*t",
)

col1, col2 = st.sidebar.columns(2)
with col1:
    a_input_raw = st.sidebar.text_input("Límite inferior (a):", value="-π")
with col2:
    b_input_raw = st.sidebar.text_input("Límite superior (b):", value="π")

st.sidebar.header("2. Parámetros de Fourier")
N_harmonics = st.sidebar.slider(
    "Número de armónicos (N):", min_value=1, max_value=50, value=10
)

# Parseo seguro de los límites a y b
try:
    a_clean = clean_expression(a_input_raw)
    b_clean = clean_expression(b_input_raw)
    a_val = float(sp.sympify(a_clean).evalf())
    b_val = float(sp.sympify(b_clean).evalf())
except Exception as e:
    st.error(f"Error al evaluar los límites a o b: Ingrese una expresión válida como -pi, pi, -3.14 o π. ({e})")
    st.stop()

if a_val >= b_val:
    st.error("El límite inferior (a) debe ser estrictamente menor que el límite superior (b).")
    st.stop()


# --- CÁLCULO DE LA SERIE DE FOURIER ---
def calculate_fourier_analysis(func_str, a_v, b_v, n_terms):
    t = sp.Symbol("t")
    clean_func = clean_expression(func_str)
    try:
        f_expr = sp.sympify(clean_func)
        f_num = sp.lambdify(t, f_expr, modules=["numpy", {"sign": np.sign}])
    except Exception as e:
        st.error(f"Error en la expresión sintáctica de f(t): {e}")
        return None

    L = (b_v - a_v) / 2.0
    T = b_v - a_v
    w0 = (2 * np.pi) / T

    def safe_f(val):
        try:
            res = f_num(val)
            return float(res) if np.isscalar(res) else float(res[0])
        except Exception:
            return 0.0

    # Coeficiente a0
    a0_int, _ = integrate.quad(safe_f, a_v, b_v)
    a0 = (1.0 / L) * a0_int

    an_list = []
    bn_list = []

    for n in range(1, n_terms + 1):
        def integrand_an(val):
            return safe_f(val) * np.cos(n * w0 * (val - a_v - L))

        def integrand_bn(val):
            return safe_f(val) * np.sin(n * w0 * (val - a_v - L))

        an_val, _ = integrate.quad(integrand_an, a_v, b_v)
        bn_val, _ = integrate.quad(integrand_bn, a_v, b_v)

        an_list.append((1.0 / L) * an_val)
        bn_list.append((1.0 / L) * bn_val)

    return f_expr, f_num, safe_f, a0, an_list, bn_list, L, T, w0


analysis_result = calculate_fourier_analysis(func_input_raw, a_val, b_val, N_harmonics)

if analysis_result:
    f_expr, f_num, safe_f, a0_calc, an_calc, bn_calc, L, T, w0 = analysis_result

    # Panel de Modificación Manual
    with st.sidebar.expander("Modificación Manual de Coeficientes"):
        modify_coeffs = st.checkbox("Activar modificación manual")
        a0 = a0_calc
        an_list = list(an_calc)
        bn_list = list(bn_calc)

        if modify_coeffs:
            a0 = st.number_input("a0:", value=float(a0_calc), format="%.4f")
            for i in range(N_harmonics):
                col_a, col_b = st.columns(2)
                with col_a:
                    an_list[i] = st.number_input(f"a_{i+1}:", value=float(an_calc[i]), format="%.4f", key=f"an_{i}")
                with col_b:
                    bn_list[i] = st.number_input(f"b_{i+1}:", value=float(bn_calc[i]), format="%.4f", key=f"bn_{i}")

    # Configuración de ejes en términos de pi para los gráficos
    tick_vals = np.linspace(a_val - T, b_val + T, 9)
    tick_text = []
    for tv in tick_vals:
        ratio = tv / np.pi
        if np.isclose(ratio, 0):
            tick_text.append("0")
        elif np.isclose(ratio, 1):
            tick_text.append("π")
        elif np.isclose(ratio, -1):
            tick_text.append("-π")
        elif np.isclose(ratio, 2):
            tick_text.append("2π")
        elif np.isclose(ratio, -2):
            tick_text.append("-2π")
        else:
            tick_text.append(f"{ratio:.2f}π")

    # --- PESTAÑAS DE NAVEGACIÓN ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Comparación de Señales", 
        "Coeficientes y Espectro", 
        "Análisis de Error", 
        "Proceso Matemático"
    ])

    # Generación de vectores temporales
    t_vals = np.linspace(a_val - T, b_val + T, 1500)

    # Función Periódica
    def periodic_f(t_array):
        y_out = np.zeros_like(t_array)
        for idx, tv in enumerate(t_array):
            tv_shift = a_val + ((tv - a_val) % T)
            y_out[idx] = safe_f(tv_shift)
        return y_out

    y_original_periodic = periodic_f(t_vals)

    # Reconstrucción de la Serie de Fourier
    fourier_series = np.full_like(t_vals, a0 / 2.0)
    for n in range(1, N_harmonics + 1):
        fourier_series += an_list[n - 1] * np.cos(n * w0 * (t_vals - a_val - L)) + bn_list[n - 1] * np.sin(n * w0 * (t_vals - a_val - L))

    # --- PESTAÑA 1: COMPARACIÓN ---
    with tab1:
        st.subheader("Observar y Comparar: Señal Original f(t) vs. Serie de Fourier")
        st.write("""
        **¿Qué estás viendo aquí?**
        - **Línea azul (Señal Periódica):** Es la función matemática original repetida periódicamente a lo largo del tiempo.
        - **Línea roja discontinua (Aproximación de Fourier):** Es la reconstrucción obtenida al sumar la cantidad seleccionada de armónicos ($N$). A mayor número de armónicos, más exacta es la aproximación.
        """)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_vals, y=y_original_periodic, mode="lines", name="Señal Periódica f(t)", line=dict(color="blue", width=2)))
        fig.add_trace(go.Scatter(x=t_vals, y=fourier_series, mode="lines", name=f"Serie de Fourier (N={N_harmonics})", line=dict(color="red", width=2, dash="dash")))

        fig.update_layout(
            xaxis=dict(title="Tiempo (t)", tickmode="array", tickvals=tick_vals, ticktext=tick_text),
            yaxis_title="Amplitud",
            margin=dict(l=20, r=20, t=30, b=20),
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- PESTAÑA 2: COEFICIENTES Y ESPECTROS ---
    with tab2:
        st.subheader("Coeficientes y Espectros de Frecuencia")
        st.write("""
        **Explicación de Parámetros:**
        - **a0 / 2 (Componente DC):** Es el valor promedio de la señal en un periodo completo (desplazamiento vertical).
        - **a_n (Cosenos):** Amplitude de las componentes pares (armónicos cosenoidales).
        - **b_n (Senos):** Amplitude de las componentes impares (armónicos senoidales).
        - **Espectro de Amplitud $|C_n|$:** Representa la fuerza o magnitud total $\sqrt{a_n^2 + b_n^2}$ de cada frecuencia $n$.
        - **Espectro de Fase:** Muestra el desfase angular $\\theta_n = \\arctan(-b_n / a_n)$ de cada armónico.
        """)
        col_tb, col_sp1, col_sp2 = st.columns([1, 1, 1])

        with col_tb:
            st.write("**Tabla de Coeficientes:**")
            st.write(f"a0 / 2 = `{a0 / 2.0:.4f}`")
            coeffs_data = {
                "n": list(range(1, N_harmonics + 1)),
                "a_n": [round(val, 4) for val in an_list],
                "b_n": [round(val, 4) for val in bn_list],
            }
            st.dataframe(coeffs_data, height=350, use_container_width=True)

        magnitudes = [float(np.sqrt(a**2 + b**2)) for a, b in zip(an_list, bn_list)]
        fases = [float(np.arctan2(-b, a)) for a, b in zip(an_list, bn_list)]

        with col_sp1:
            st.write("**Espectro de Amplitud |C_n|:**")
            fig_mag = go.Figure(data=[go.Bar(x=list(range(1, N_harmonics + 1)), y=magnitudes, marker_color="indigo")])
            fig_mag.update_layout(xaxis_title="Armónico (n)", yaxis_title="Amplitud", template="plotly_white", margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_mag, use_container_width=True)

        with col_sp2:
            st.write("**Espectro de Fase (Rad):**")
            fig_phase = go.Figure(data=[go.Bar(x=list(range(1, N_harmonics + 1)), y=fases, marker_color="teal")])
            fig_phase.update_layout(xaxis_title="Armónico (n)", yaxis_title="Fase (Rad)", template="plotly_white", margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_phase, use_container_width=True)

    # --- PESTAÑA 3: ANÁLISIS DE ERROR ---
    with tab3:
        st.subheader("Métricas de Error de la Aproximación")
        st.write("""
        **¿Qué significan estos indicadores de error?**
        - **Error Cuadrático Medio (MSE):** Mide la diferencia promedio al cuadrado entre la función real y la aproximación de Fourier. Cuanto más cercano a 0, mejor es la aproximación.
        - **Porcentaje de Error Relativo ($L^2$):** Representa la proporción de energía o información de la señal original que no logra cubrir la suma de $N$ armónicos. Un valor del 0% indica reconstrucción perfecta.
        """)
        
        t_fund = np.linspace(a_val, b_val, 1000)
        y_fund_orig = np.vectorize(safe_f)(t_fund)
        
        y_fund_fourier = np.full_like(t_fund, a0 / 2.0)
        for n in range(1, N_harmonics + 1):
            y_fund_fourier += an_list[n - 1] * np.cos(n * w0 * (t_fund - a_val - L)) + bn_list[n - 1] * np.sin(n * w0 * (t_fund - a_val - L))

        mse = np.mean((y_fund_orig - y_fund_fourier) ** 2)
        norm_orig = np.mean(y_fund_orig ** 2)
        rel_error = (mse / norm_orig * 100) if norm_orig != 0 else 0.0

        c1, c2 = st.columns(2)
        c1.metric("Error Cuadrático Medio (MSE)", f"{mse:.6f}")
        c2.metric("Porcentaje de Error Relativo (L²)", f"{rel_error:.4f} %")

        # Ajuste de eje pi para la gráfica de error
        tick_vals_fund = np.linspace(a_val, b_val, 5)
        tick_text_fund = [f"{(tv/np.pi):.2f}π" if not np.isclose(tv, 0) else "0" for tv in tick_vals_fund]

        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=t_fund, y=np.abs(y_fund_orig - y_fund_fourier), mode="lines", name="Error Absoluto |f(t) - S_N(t)|", line=dict(color="orange")))
        fig_err.update_layout(
            xaxis=dict(title="Tiempo (t)", tickmode="array", tickvals=tick_vals_fund, ticktext=tick_text_fund),
            yaxis_title="Diferencia Absoluta", 
            template="plotly_white", 
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # --- PESTAÑA 4: PROCESO MATEMÁTICO ---
    with tab4:
        st.subheader("Desarrollo Matemático Detallado de la Serie de Fourier")
        
        st.write("### 1. Definición General de la Serie Trigonométrica")
        st.latex(r"f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(n \omega_0 t) + b_n \sin(n \omega_0 t) \right]")
        
        st.write("### 2. Parámetros Fundamentales Obtenidos")
        st.write(f"- **Intervalo de Integración:** $[a, b] = [{a_input_raw}, {b_input_raw}]$")
        st.write(f"- **Periodo ($T = b - a$):** ${T:.4f}$")
        st.write(f"- **Semiperiodo ($L = T / 2$):** ${L:.4f}$")
        st.write(f"- **Frecuencia Fundamental ($\omega_0 = 2\pi / T$):** ${w0:.4f} \\text{{ rad/s}}$")

        st.write("### 3. Fórmulas Generales e Integrales Sustituidas")
        st.write("**Componente Promedio ($a_0$):**")
        st.latex(f"a_0 = \\frac{{1}}{{{L:.4f}}} \\int_{{{a_val:.4f}}}^{{{b_val:.4f}}} ({sp.latex(f_expr)}) \, dt")
        st.write(f"Resultado de la integración: **a_0 = {a0_calc:.4f}**")

        st.write("**Coeficientes de Coseno ($a_n$):**")
        st.latex(f"a_n = \\frac{{1}}{{{L:.4f}}} \\int_{{{a_val:.4f}}}^{{{b_val:.4f}}} ({sp.latex(f_expr)}) \\cos\\left(n \\cdot {w0:.4f} \\cdot t\\right) dt")

        st.write("**Coeficientes de Seno ($b_n$):**")
        st.latex(f"b_n = \\frac{{1}}{{{L:.4f}}} \\int_{{{a_val:.4f}}}^{{{b_val:.4f}}} ({sp.latex(f_expr)}) \\sin\\left(n \\cdot {w0:.4f} \\cdot t\\right) dt")

        st.write("### 4. Sustitución en la Serie Parcial ($S_N(t)$)")
        st.write("Reemplazando los valores numéricos calculados para los primeros armónicos:")
        
        t_sym = sp.Symbol('t')
        fourier_sym_expr = sp.N(a0_calc / 2.0, 4)
        for n in range(1, min(6, N_harmonics + 1)):
            term_an = sp.N(an_calc[n - 1], 4) * sp.cos(sp.N(n * w0, 4) * t_sym)
            term_bn = sp.N(bn_calc[n - 1], 4) * sp.sin(sp.N(n * w0, 4) * t_sym)
            fourier_sym_expr += term_an + term_bn
            
        st.latex(f"S_{{N}}(t) \\approx " + sp.latex(fourier_sym_expr))
        if N_harmonics > 5:
            st.caption("Nota: Se despliegan explícitamente los primeros 5 armónicos en la fórmula truncada.")
