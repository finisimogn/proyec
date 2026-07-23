import numpy as np
import plotly.graph_objects as go
import scipy.integrate as integrate
import streamlit as st
import sympy as sp

st.set_page_config(page_title="Calculadora de Fourier", layout="wide")

st.title("Calculadora de Fourier")
st.markdown(
    """
Esta aplicación permite calcular, analizar y modificar la **Serie de Fourier** de una función periódica en el dominio del tiempo $t$.
"""
)

# --- PANEL LATERAL: ENTRADA DE DATOS ---
st.sidebar.header("1. Configuración de la Función")

func_str = st.sidebar.text_input(
    "Función f(t):",
    value="t",
    help="Usa sintaxis matemática de Python/SymPy. Ejemplos: t, t**2, sin(t), cos(t), pi, pi*t, exp(-t)",
)

col1, col2 = st.sidebar.columns(2)
with col1:
    a_str = st.sidebar.text_input("Límite inferior (a):", value="-pi")
with col2:
    b_str = st.sidebar.text_input("Límite superior (b):", value="pi")

st.sidebar.header("2. Parámetros de Fourier")
N_harmonics = st.sidebar.slider(
    "Número de armónicos (N):", min_value=1, max_value=50, value=10
)

# --- FUNCIÓN DE PARSEO DE EXPRESIONES ---
def parse_math_expression(expr_str):
    clean_str = str(expr_str).replace("π", "pi")
    expr_sym = sp.sympify(clean_str)
    num_val = float(expr_sym.evalf())
    return expr_sym, num_val

try:
    a_sym, a_val = parse_math_expression(a_str)
    b_sym, b_val = parse_math_expression(b_str)
except Exception as e:
    st.error(f"Error al procesar los límites de integración (a o b): {e}")
    st.stop()

if a_val >= b_val:
    st.error("El límite inferior (a) debe ser strictly menor que el límite superior (b).")
    st.stop()


# --- CÁLCULO NUMÉRICO DE COEFICIENTES (SOLO RETORNA DATOS SERIALIZABLES) ---
@st.cache_data(show_spinner=False)
def compute_fourier_coefficients(clean_func_str, a_v, b_v, n_terms):
    t = sp.Symbol("t")
    try:
        f_expr = sp.sympify(clean_func_str)
        f_num = sp.lambdify(t, f_expr, modules=["numpy", {"sign": np.sign}])
    except Exception as e:
        return None, None, None, None, None, None, str(e)

    L = (b_v - a_v) / 2.0
    T = b_v - a_v
    w0 = (2 * np.pi) / T

    def safe_eval(val):
        try:
            res = f_num(val)
            return float(res) if np.isscalar(res) else float(res[0])
        except Exception:
            return 0.0

    # Coeficiente a0
    a0_int, _ = integrate.quad(safe_eval, a_v, b_v, limit=100)
    a0 = (1.0 / L) * a0_int

    an_list = []
    bn_list = []

    for n in range(1, n_terms + 1):
        def integrand_an(val):
            return safe_eval(val) * np.cos(n * w0 * (val - a_v - L))

        def integrand_bn(val):
            return safe_eval(val) * np.sin(n * w0 * (val - a_v - L))

        an_val, _ = integrate.quad(integrand_an, a_v, b_v, limit=100)
        bn_val, _ = integrate.quad(integrand_bn, a_v, b_v, limit=100)

        an_list.append((1.0 / L) * an_val)
        bn_list.append((1.0 / L) * bn_val)

    return a0, an_list, bn_list, L, T, w0, None


# Reconstrucción fuera del caché
clean_func_input = str(func_str).replace("π", "pi")

try:
    t_sym = sp.Symbol("t")
    f_expr = sp.sympify(clean_func_input)
    f_num_fast = sp.lambdify(t_sym, f_expr, modules=["numpy", {"sign": np.sign}])
    
    def safe_f(val):
        try:
            res = f_num_fast(val)
            return float(res) if np.isscalar(res) else float(res[0])
        except Exception:
            return 0.0

except Exception as err:
    st.error(f"Error en la expresión matemática f(t): {err}")
    st.stop()

a0_calc, an_calc, bn_calc, L, T, w0, err_msg = compute_fourier_coefficients(
    clean_func_input, a_val, b_val, N_harmonics
)

if err_msg:
    st.error(f"Error al calcular los coeficientes de Fourier: {err_msg}")
    st.stop()

if a0_calc is not None:
    # Expansor lateral para modificación manual
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Comparación de Señales", 
        "Coeficientes y Espectro", 
        "Análisis de Error Explicado", 
        "Guía de Parámetros",
        "Desarrollo Matemático Paso a Paso"
    ])

    # Vectorización para gráficos
    t_vals = np.linspace(a_val - T, b_val + T, 1000)

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

    # Generación dinámica de etiquetas Pi sin sobrecarga
    def get_pi_ticks_dynamic(min_v, max_v):
        total_range = max_v - min_v
        
        if total_range <= 4 * np.pi:
            step = np.pi / 2
        elif total_range <= 12 * np.pi:
            step = np.pi
        elif total_range <= 30 * np.pi:
            step = 5 * np.pi
        else:
            step = 10 * np.pi

        start_k = np.floor(min_v / step)
        end_k = np.ceil(max_v / step)
        tick_vals = []
        tick_text = []
        
        for k in range(int(start_k), int(end_k) + 1):
            val = k * step
            if val < min_v or val > max_v:
                continue
            tick_vals.append(val)
            
            if abs(val) < 1e-5:
                tick_text.append("0")
            else:
                mult = val / np.pi
                if np.isclose(mult, 1.0):
                    tick_text.append("π")
                elif np.isclose(mult, -1.0):
                    tick_text.append("-π")
                elif np.isclose(mult, 0.5):
                    tick_text.append("π/2")
                elif np.isclose(mult, -0.5):
                    tick_text.append("-π/2")
                elif mult.is_integer():
                    tick_text.append(f"{int(mult)}π")
                else:
                    tick_text.append(f"{mult:.1f}π")
                
        return tick_vals, tick_text

    x_ticks, x_labels = get_pi_ticks_dynamic(a_val - T, b_val + T)

    # --- PESTAÑA 1: COMPARACIÓN ---
    with tab1:
        st.subheader("Observar y Comparar: Señal Original f(t) vs. Serie de Fourier")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_vals, y=y_original_periodic, mode="lines", name="Señal Periódica f(t)", line=dict(color="blue", width=2)))
        fig.add_trace(go.Scatter(x=t_vals, y=fourier_series, mode="lines", name=f"Aproximación (N={N_harmonics})", line=dict(color="red", width=2, dash="dash")))

        fig.update_layout(
            xaxis_title="Tiempo (t)",
            yaxis_title="Amplitud f(t)",
            xaxis=dict(
                tickmode="array",
                tickvals=x_ticks,
                ticktext=x_labels
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- PESTAÑA 2: COEFICIENTES Y ESPECTRO ---
    with tab2:
        st.subheader("Coeficientes y Espectros de Frecuencia")
        col_tb, col_sp1, col_sp2 = st.columns([1, 1, 1])

        with col_tb:
            st.write("**Coeficientes Calculados:**")
            st.write(f"Componente DC (a0 / 2): `{a0 / 2.0:.4f}`")
            coeffs_data = {
                "n": list(range(1, N_harmonics + 1)),
                "a_n (Cosenos)": [round(val, 4) for val in an_list],
                "b_n (Senos)": [round(val, 4) for val in bn_list],
            }
            st.dataframe(coeffs_data, height=350, use_container_width=True)

        magnitudes = [float(np.sqrt(a**2 + b**2)) for a, b in zip(an_list, bn_list)]
        fases = [float(np.arctan2(-b, a)) for a, b in zip(an_list, bn_list)]

        with col_sp1:
            st.write("**Espectro de Amplitud |C_n|:**")
            fig_mag = go.Figure(data=[go.Bar(x=list(range(1, N_harmonics + 1)), y=magnitudes, marker_color="indigo")])
            fig_mag.update_layout(
                xaxis_title="Número de Armónico (n)",
                yaxis_title="Magnitud √(a_n² + b_n²)",
                template="plotly_white",
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig_mag, use_container_width=True)

        with col_sp2:
            st.write("**Espectro de Fase (Radianes):**")
            fig_phase = go.Figure(data=[go.Bar(x=list(range(1, N_harmonics + 1)), y=fases, marker_color="teal")])
            fig_phase.update_layout(
                xaxis_title="Número de Armónico (n)",
                yaxis_title="Fase θ_n (rad)",
                template="plotly_white",
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig_phase, use_container_width=True)

    # --- PESTAÑA 3: ANÁLISIS DE ERROR EXPLICADO ---
    with tab3:
        st.subheader("Explicación y Cálculo del Error de Aproximación")
        
        t_fund = np.linspace(a_val, b_val, 1000)
        y_fund_orig = np.vectorize(safe_f)(t_fund)
        
        y_fund_fourier = np.full_like(t_fund, a0 / 2.0)
        for n in range(1, N_harmonics + 1):
            y_fund_fourier += an_list[n - 1] * np.cos(n * w0 * (t_fund - a_val - L)) + bn_list[n - 1] * np.sin(n * w0 * (t_fund - a_val - L))

        mse = np.mean((y_fund_orig - y_fund_fourier) ** 2)
        norm_orig = np.mean(y_fund_orig ** 2)
        rel_error = (mse / norm_orig * 100) if norm_orig != 0 else 0.0

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.metric("Error Cuadrático Medio (MSE)", f"{mse:.6f}")
            st.markdown("Calcula el promedio de las diferencias al cuadrado entre $f(t)$ y la Serie de Fourier.")
        with col_e2:
            st.metric("Porcentaje de Error Relativo (L²)", f"{rel_error:.4f} %")
            st.markdown("Compara la energía del error con la energía total de la señal original.")

        st.subheader("Gráfica del Error Absoluto Punto a Punto")
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=t_fund, y=np.abs(y_fund_orig - y_fund_fourier), mode="lines", name="|Error|", line=dict(color="orange")))
        
        x_ticks_f, x_labels_f = get_pi_ticks_dynamic(a_val, b_val)
        fig_err.update_layout(
            xaxis_title="Tiempo (t)",
            yaxis_title="Diferencia Absoluta |f(t) - S_N(t)|",
            xaxis=dict(tickmode="array", tickvals=x_ticks_f, ticktext=x_labels_f),
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # --- PESTAÑA 4: GUÍA DE PARÁMETROS ---
    with tab4:
        st.subheader("Guía Explicativa de Parámetros y Conceptos")
        st.markdown(
            """
- **$t$ (Tiempo):** Variable independiente en el dominio del tiempo.
- **$f(t)$ (Función / Señal):** Onda que se analiza.
- **$T = b - a$ (Periodo):** Duración de un ciclo completo de la onda.
- **$\omega_0 = \\frac{2\\pi}{T}$ (Frecuencia fundamental):** Frecuencia angular base en rad/s.
- **$N$ (Número de Armónicos):** Cantidad de componentes armónicas sumadas.
            """
        )

    # --- PESTAÑA 5: DESARROLLO MATEMÁTICO ---
    with tab5:
        st.subheader("Desarrollo Matemático Analítico (Paso a Paso)")

        st.markdown("### 1. Definición Formal")
        st.latex(r"f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(n \omega_0 t) + b_n \sin(n \omega_0 t) \right]")

        st.markdown("### 2. Parámetros del Intervalo")
        st.write(f"- Periodo $T = b - a =$ **{T:.4f}**")
        st.write(f"- Semiperiodo $L = T / 2 =$ **{L:.4f}**")
        st.write(f"- Frecuencia fundamental $\\omega_0 =$ **{w0:.4f} rad/s**")

        st.markdown("### 3. Expresión Reconstruida de la Serie")
        series_terms = [f"{a0/2.0:.4f}"]
        for n_idx in range(1, min(6, N_harmonics + 1)):
            an_v = an_list[n_idx - 1]
            bn_v = bn_list[n_idx - 1]
            if abs(an_v) > 1e-4:
                series_terms.append(f"{an_v:+.4f}\\cos({n_idx}\\omega_0 t)")
            if abs(bn_v) > 1e-4:
                series_terms.append(f"{bn_v:+.4f}\\sin({n_idx}\\omega_0 t)")

        st.latex(r"S_N(t) \approx " + " ".join(series_terms))
