import numpy as np
import plotly.graph_objects as go
import scipy.integrate as integrate
import streamlit as st
import sympy as sp

st.set_page_config(page_title="Calculadora de Fourier", layout="wide")

# --- TÍTULO PRINCIPAL SIN EMOTICONES ---
st.title("Calculadora de Fourier")
st.markdown(
    """
Esta aplicación permite calcular, analizar y modificar la **Serie de Fourier** de una función periódica en el dominio del tiempo $t$.
A continuación, puedes configurar los parámetros de entrada y explorar los resultados gráficos y el desarrollo matemático detallado.
"""
)

# --- PANEL LATERAL: ENTRADA DE DATOS ---
st.sidebar.header("1. Configuración de la Función")

func_str = st.sidebar.text_input(
    "Función f(t):",
    value="t",
    help="Usa sintaxis matemática de Python/SymPy. Puedes usar: t, t**2, sin(t), cos(t), pi, pi*t, exp(-t), etc.",
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

# --- FUNCIÓN DE PARSEO DE EXPRESIONES (SOPORTE PARA pi, π, e, etc.) ---
def parse_math_expression(expr_str):
    """
    Sustituye símbolos comunes como 'π' por 'pi' y evalúa la expresión simbólicamente
    para obtener un valor numérico seguro.
    """
    clean_str = str(expr_str).replace("π", "pi")
    expr_sym = sp.sympify(clean_str)
    num_val = float(expr_sym.evalf())
    return expr_sym, num_val


# --- EVALUACIÓN Y PARSEO DE LÍMITES ---
try:
    a_sym, a_val = parse_math_expression(a_str)
    b_sym, b_val = parse_math_expression(b_str)
except Exception as e:
    st.error(f"Error al procesar los límites de integración (a o b): {e}")
    st.stop()

if a_val >= b_val:
    st.error("El límite inferior (a) debe ser estrictamente menor que el límite superior (b).")
    st.stop()


# --- FUNCIÓN DE CÁLCULO NUMÉRICO Y SIMBÓLICO ---
def calculate_fourier_analysis(func_input, a_v, b_v, n_terms):
    t = sp.Symbol("t")
    clean_func = str(func_input).replace("π", "pi")
    
    try:
        f_expr = sp.sympify(clean_func)
        f_num = sp.lambdify(t, f_expr, modules=["numpy", {"sign": np.sign}])
    except Exception as e:
        st.error(f"Error en la expresión matemática de f(t): {e}")
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


analysis_result = calculate_fourier_analysis(func_str, a_val, b_val, N_harmonics)

if analysis_result:
    f_expr, f_num, safe_f, a0_calc, an_calc, bn_calc, L, T, w0 = analysis_result

    # Expansor lateral para modificación manual de coeficientes
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

    # --- PESTAÑAS DE LA APLICACIÓN ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Comparación de Señales", 
        "Coeficientes y Espectro", 
        "Análisis de Error Explicado", 
        "Guía de Parámetros",
        "Desarrollo Matemático Paso a Paso"
    ])

    # Generación de vectores para graficar
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

    # --- FORMATO DE EJES EN MÚLTIPLOS DE PI ---
    def get_pi_ticks(min_v, max_v):
        step = np.pi / 2
        start_k = np.floor(min_v / step)
        end_k = np.ceil(max_v / step)
        tick_vals = []
        tick_text = []
        
        for k in range(int(start_k), int(end_k) + 1):
            val = k * step
            if val < min_v or val > max_v:
                continue
            tick_vals.append(val)
            
            # Formateo de texto en múltiplos de pi
            if k == 0:
                tick_text.append("0")
            elif k == 2:
                tick_text.append("π")
            elif k == -2:
                tick_text.append("-π")
            elif k == 1:
                tick_text.append("π/2")
            elif k == -1:
                tick_text.append("-π/2")
            elif k % 2 == 0:
                tick_text.append(f"{k//2}π")
            else:
                tick_text.append(f"{k}π/2")
                
        return tick_vals, tick_text

    x_ticks, x_labels = get_pi_ticks(a_val - T, b_val + T)

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
        
        st.markdown(
            """
### ¿Qué significa el error en una Serie de Fourier?
Una Serie de Fourier es una **aproximación** mediante una suma finita de funciones senos y cosenos. Al usar un número limitado de armónicos ($N$), existe una diferencia entre la función original $f(t)$ y la aproximación $S_N(t)$.

Para medir qué tan precisa es la aproximación, utilizamos dos métricas estándar:
            """
        )

        # Cálculo de errores en el intervalo fundamental [a, b]
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
            st.markdown(
                """
**Error Cuadrático Medio (MSE):**
Calcula el promedio de las diferencias al cuadrado entre la función real $f(t)$ y la aproximación $S_N(t)$ en cada instante de tiempo.
- **Fórmula:** $\\text{MSE} = \\frac{1}{T} \\int_{a}^{b} [f(t) - S_N(t)]^2 dt$
- **Interpretación:** Mientras más cercano sea a **0**, más idénticas son las dos curvas.
                """
            )

        with col_e2:
            st.metric("Porcentaje de Error Relativo (L²)", f"{rel_error:.4f} %")
            st.markdown(
                """
**Porcentaje de Error Relativo:**
Compara la magnitud del error respecto a la energía total de la señal original expressado en porcentaje.
- **Interpretación:** 
  - **0%:** Aproximación perfecta.
  - **Menor al 5%:** Muy buena aproximación en la práctica de ingeniería.
                """
            )

        st.subheader("Gráfica del Error Absoluto Punto a Punto en el Tiempo")
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=t_fund, y=np.abs(y_fund_orig - y_fund_fourier), mode="lines", name="|Error|", line=dict(color="orange")))
        
        x_ticks_f, x_labels_f = get_pi_ticks(a_val, b_val)
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
### 1. Variables de la Serie de Fourier

- **$t$ (Tiempo):** Es la variable independiente de la señal en lugar de $x$, ya que las ondas varían periódicamente a lo largo del tiempo.
- **$f(t)$ (Función / Señal):** Es la forma de onda que se desea analizar (onda cuadrada, triangular, diente de sierra, etc.).
- **$a$ y $b$ (Límites del Intervalo):** Definen el intervalo temporal en el cual se analiza la función. La distancia $b - a$ define el periodo de la onda.
- **$T$ (Periodo fundamental):** El tiempo que tarda la señal en repetir un ciclo completo ($T = b - a$).
- **$\omega_0$ (Frecuencia angular fundamental):** Velocidad de rotación de la onda base medida en radianes por segundo ($\omega_0 = \\frac{2\\pi}{T}$).
- **$N$ (Número de Armónicos):** Es la cantidad de ondas senoidales/cosenoidales simples que sumamos para aproximar la onda original. A mayor $N$, mayor precisión.

---

### 2. Significado de los Coeficientes

- **$a_0 / 2$ (Componente DC o Valor Medio):** Representa el nivel medio o el desplazamiento vertical de la señal respecto a cero.
- **$a_n$ (Coeficiente Cosenoidal):** Mide cuánta simetría par (coseno) tiene la componente armónica $n$.
- **$b_n$ (Coeficiente Senoidal):** Mide cuánta simetría impar (seno) tiene la componente armónica $n$.
- **$C_n = \sqrt{a_n^2 + b_n^2}$ (Magnitud del Armónico):** Representa la amplitud total del armónico $n$, sin importar si es seno o coseno.
- **$\theta_n = \text{arctan2}(-b_n, a_n)$ (Fase del Armónico):** Ángulo de desfase del armónico $n$ en radianes.
            """
        )

    # --- PESTAÑA 5: DESARROLLO MATEMÁTICO PASO A PASO ---
    with tab5:
        st.subheader("Desarrollo Matemático Analítico (Paso a Paso)")

        st.markdown("### 1. Definición Formal de la Serie de Fourier")
        st.latex(r"f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(n \omega_0 t) + b_n \sin(n \omega_0 t) \right]")

        st.markdown("### 2. Obtención de Parámetros de la Señal")
        st.write(f"- Límite inferior (a): `{a_str}` $\\rightarrow$ **{a_val:.4f}**")
        st.write(f"- Límite superior (b): `{b_str}` $\\rightarrow$ **{b_val:.4f}**")
        st.write(f"- Periodo $T = b - a = {b_val:.4f} - ({a_val:.4f}) =$ **{T:.4f}**")
        st.write(f"- Semiperiodo $L = T / 2 =$ **{L:.4f}**")
        st.write(f"- Frecuencia fundamental $\\omega_0 = \\frac{{2\\pi}}{{T}} =$ **{w0:.4f} rad/s**")

        st.markdown("### 3. Resolución Simbólica de Integrales")
        
        t_sym = sp.Symbol('t', real=True)
        n_sym = sp.Symbol('n', positive=True, integer=True)

        try:
            # Integración simbólica analítica de a0
            int_a0_expr = sp.integrate(f_expr, (t_sym, a_sym, b_sym))
            a0_sym_formula = (1 / (L)) * int_a0_expr
            
            st.write("**Cálculo analítico del valor promedio ($a_0$):**")
            st.latex(r"a_0 = \frac{1}{L} \int_{a}^{b} f(t) \, dt = \frac{1}{" + sp.latex(sp.sympify(L)) + r"} \int_{" + sp.latex(a_sym) + r"}^{" + sp.latex(b_sym) + r"} (" + sp.latex(f_expr) + r") \, dt")
            st.write("Resultado de la integración para $a_0$:")
            st.latex(r"a_0 = " + sp.latex(sp.simplify(a0_sym_formula)))
            
            # Integración simbólica para a_n y b_n
            arg_cos = n_sym * (2 * sp.pi / (b_sym - a_sym)) * (t_sym - a_sym - (b_sym - a_sym)/2)
            arg_sin = n_sym * (2 * sp.pi / (b_sym - a_sym)) * (t_sym - a_sym - (b_sym - a_sym)/2)

            st.write("**Fórmula general analítica para $a_n$:**")
            st.latex(r"a_n = \frac{1}{L} \int_{a}^{b} f(t) \cos(n \omega_0 t) \, dt")
            
            st.write("**Fórmula general analítica para $b_n$:**")
            st.latex(r"b_n = \frac{1}{L} \int_{a}^{b} f(t) \sin(n \omega_0 t) \, dt")

        except Exception as err:
            st.info(f"No se pudo resolver la integral simbólica cerrada directamente debido a la complejidad de la función. Se usó integración numérica de alta precisión: {err}")

        st.markdown("### 4. Serie de Fourier Sustituida (Primeros Armónicos)")
        
        # Construcción LaTeX de la serie final
        series_terms = [f"{a0/2.0:.4f}"]
        for n_idx in range(1, min(6, N_harmonics + 1)):
            an_v = an_list[n_idx - 1]
            bn_v = bn_list[n_idx - 1]
            
            if abs(an_v) > 1e-4:
                series_terms.append(f"{an_v:+.4f}\\cos({n_idx}\\omega_0 t)")
            if abs(bn_v) > 1e-4:
                series_terms.append(f"{bn_v:+.4f}\\sin({n_idx}\\omega_0 t)")

        latex_series_str = " ".join(series_terms)
        st.latex(r"S_N(t) \approx " + latex_series_str)
        if N_harmonics > 5:
            st.caption("Nota: Por claridad visual, la expresión LaTeX muestra únicamente hasta los primeros 5 armónicos.")
