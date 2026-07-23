"""
Calculadora de Serie de Fourier — Desarrollo simbólico paso a paso.

Ejecutar localmente:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

# ----------------------------------------------------------------------------
# CONFIGURACION DE LA PAGINA
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Calculadora de Serie de Fourier", layout="wide")

st.title("Calculadora de Serie de Fourier")
st.caption(
    "Desarrollo simbolico paso a paso con SymPy, grafica de la extension periodica "
    "y reconstruccion con N armonicos."
)


# ----------------------------------------------------------------------------
# UTILIDADES SIMBOLICAS
# ----------------------------------------------------------------------------
def clean(text: str) -> str:
    """Normaliza la entrada del usuario para sympify."""
    return (
        text.replace("π", "pi")
        .replace("^", "**")
        .strip()
    )


def simplify_integer_n(expr, n_sym):
    """
    Reescribe una expresion aprovechando que n es un entero:
    cos(n*pi) -> (-1)**n, sin(n*pi) -> 0, etc.
    Esto es lo que convierte resultados como (2/(n*pi))(1-cos(n*pi))
    en la forma limpia de libro (2/(n*pi))(1-(-1)**n).
    """
    if expr is None:
        return None
    e = sp.simplify(expr)
    # Reemplazos tipicos de las funciones evaluadas en multiplos enteros de pi.
    e = e.rewrite(sp.Piecewise)
    try:
        e = e.replace(sp.cos(sp.pi * n_sym), (-1) ** n_sym)
        e = e.replace(sp.sin(sp.pi * n_sym), sp.Integer(0))
        e = e.replace(sp.cos(2 * sp.pi * n_sym), sp.Integer(1))
        e = e.replace(sp.sin(2 * sp.pi * n_sym), sp.Integer(0))
    except Exception:
        pass
    return sp.simplify(e)


def integral_pretty(integrand, var, lo, hi):
    """Devuelve el objeto Integral sin evaluar (para mostrarlo bonito)."""
    return sp.Integral(integrand, (var, lo, hi))


# ----------------------------------------------------------------------------
# PANEL LATERAL
# ----------------------------------------------------------------------------
st.sidebar.header("1. Definicion de la funcion")

 #--- Ejemplos precargados (se llenan con un clic) ---
EXAMPLES = {
    "Personalizado": None,
    "f(x) = x  en [-pi, pi]": {"var": "x", "f": "x", "a": "-pi", "b": "pi"},
    "f(x) = x^2  en [-pi, pi]": {"var": "x", "f": "x**2", "a": "-pi", "b": "pi"},
    "Onda cuadrada impar  en [-pi, pi]": {
        "var": "x",
        "f": "Piecewise((-1, x < 0), (1, True))",
        "a": "-pi",
        "b": "pi",
    },
    "Rectificador media onda (10 V, 60 Hz)": {
        "var": "t",
        "f": "Piecewise((10*sin(120*pi*t), t < 1/120), (0, True))",
        "a": "0",
        "b": "1/60",
    },
    "Rectificador onda completa (10 V, 60 Hz)": {
        "var": "t",
        "f": "10*Abs(sin(120*pi*t))",
        "a": "0",
        "b": "1/120",
    },
}

# Valores iniciales en session_state (antes de crear los widgets).
st.session_state.setdefault("var_choice", "x")
st.session_state.setdefault("func_str", "x")
st.session_state.setdefault("a_str", "-pi")
st.session_state.setdefault("b_str", "pi")


def apply_example():
    ex = EXAMPLES.get(st.session_state["ejemplo"])
    if ex:
        st.session_state["var_choice"] = ex["var"]
        st.session_state["func_str"] = ex["f"]
        st.session_state["a_str"] = ex["a"]
        st.session_state["b_str"] = ex["b"]


st.sidebar.selectbox(
    "Ejemplos:", list(EXAMPLES.keys()), key="ejemplo", on_change=apply_example
)

var_choice = st.sidebar.radio(
    "Variable independiente:", ["x", "t"], horizontal=True, key="var_choice"
)

FUNC_HELP = "Sintaxis SymPy. Ej: x, x**2, sin(x), pi - x, Abs(x). Para tramos usa Piecewise((expr1, cond1), (expr2, cond2))."

func_str = st.sidebar.text_input(
    f"Funcion f({var_choice}):",
    key="func_str",
    help=FUNC_HELP,
)

c1, c2 = st.sidebar.columns(2)
with c1:
     a_str = st.sidebar.text_input("Limite inferior a:", key="a_str")
with c2:
    b_str = st.sidebar.text_input("Limite superior b:", key="b_str")

axis_format = st.sidebar.radio(
    "Formato del eje horizontal:",
    ["Automatico", "Multiplos de pi", "Decimales"],
    index=0,
    help="Automatico usa multiplos de pi solo si el periodo es multiplo de pi; "
    "en senales temporales (p. ej. T = 1/60) usa decimales.",
)

st.sidebar.header("2. Parametros de la serie")
N_harmonics = st.sidebar.slider(
    "Numero de armonicos (N):", min_value=1, max_value=400, value=20, step=1
)
n_periods = st.sidebar.slider(
    "Periodos a graficar:", min_value=1, max_value=5, value=3, step=1
)


# ----------------------------------------------------------------------------
# PARSEO
# ----------------------------------------------------------------------------
v = sp.Symbol(var_choice, real=True)
n = sp.Symbol("n", integer=True, positive=True)

try:
    a_sym = sp.nsimplify(sp.sympify(clean(a_str)), [sp.pi])
    b_sym = sp.nsimplify(sp.sympify(clean(b_str)), [sp.pi])
    f_expr = sp.sympify(clean(func_str), locals={var_choice: v})
    a_val = float(a_sym.evalf())
    b_val = float(b_sym.evalf())
except Exception as e:
    st.error(f"No se pudo interpretar la funcion o los limites: {e}")
    st.stop()

if a_val >= b_val:
    st.error("El limite inferior a debe ser menor que el limite superior b.")
    st.stop()

# Periodo, semiperiodo y frecuencia angular fundamental (simbolicos y numericos).
T_sym = sp.simplify(b_sym - a_sym)
L_sym = sp.simplify(T_sym / 2)
w0_sym = sp.simplify(2 * sp.pi / T_sym)
T_val = float(T_sym.evalf())
w0_val = float(w0_sym.evalf())

# Un intervalo es simetrico si a = -b (=> [-L, L]).
symmetric_interval = sp.simplify(a_sym + b_sym) == 0


# ----------------------------------------------------------------------------
# MOTOR SIMBOLICO: coeficientes y pasos intermedios
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Calculando la serie de Fourier...")
def fourier_engine(func_clean, var_name, a_txt, b_txt, N):
    vv = sp.Symbol(var_name, real=True)
    nn = sp.Symbol("n", integer=True, positive=True)
    f = sp.sympify(func_clean, locals={var_name: vv})
    a_s = sp.nsimplify(sp.sympify(a_txt), [sp.pi])
    b_s = sp.nsimplify(sp.sympify(b_txt), [sp.pi])
    T_s = sp.simplify(b_s - a_s)
    w0_s = sp.simplify(2 * sp.pi / T_s)

    # --- Deteccion de simetria (solo valida en intervalo simetrico) ---
    sym_ok = sp.simplify(a_s + b_s) == 0
    is_even = is_odd = False
    if sym_ok:
        # Intento simbolico.
        is_even = sp.simplify(f.subs(vv, -vv) - f) == 0
        is_odd = sp.simplify(f.subs(vv, -vv) + f) == 0
        # Respaldo numerico (util para funciones a tramos tipo onda cuadrada).
        if not is_even and not is_odd:
            try:
                f_chk = sp.lambdify(
                    vv, f, modules=["numpy", {"sign": np.sign, "Abs": np.abs}]
                )
                L = float(b_s.evalf())
                xs = np.linspace(-L * 0.999, L * 0.999, 401)
                xs = xs[np.abs(xs) > 1e-6]
                fp, fm = f_chk(xs), f_chk(-xs)
                fp = np.full_like(xs, fp) if np.isscalar(fp) else fp
                fm = np.full_like(xs, fm) if np.isscalar(fm) else fm
                if np.allclose(fp, fm, atol=1e-6):
                    is_even = True
                elif np.allclose(fp, -fm, atol=1e-6):
                    is_odd = True
            except Exception:
                pass

    steps = {}

    # --- a0 ---
    a0_integrand = f
    a0_anti = sp.integrate(a0_integrand, vv)
    if is_odd:
        a0_sym = sp.Integer(0)
    else:
        a0_def = sp.integrate(a0_integrand, (vv, a_s, b_s))
        a0_sym = sp.simplify((2 / T_s) * a0_def)
    steps["a0"] = {
        "integrand": a0_integrand,
        "anti": a0_anti,
        "result": a0_sym,
    }

    # --- an ---
    an_integrand = sp.simplify(f * sp.cos(nn * w0_s * vv))
    if is_odd:
        an_sym = sp.Integer(0)
        an_anti = sp.Integer(0)
    else:
        an_anti = sp.integrate(an_integrand, vv)
        an_def = sp.integrate(an_integrand, (vv, a_s, b_s))
        an_sym = sp.simplify((2 / T_s) * an_def)
    steps["an"] = {
        "integrand": an_integrand,
        "anti": an_anti,
        "result": an_sym,
    }

    # --- bn ---
    bn_integrand = sp.simplify(f * sp.sin(nn * w0_s * vv))
    if is_even:
        bn_sym = sp.Integer(0)
        bn_anti = sp.Integer(0)
    else:
        bn_anti = sp.integrate(bn_integrand, vv)
        bn_def = sp.integrate(bn_integrand, (vv, a_s, b_s))
        bn_sym = sp.simplify((2 / T_s) * bn_def)
    steps["bn"] = {
        "integrand": bn_integrand,
        "anti": bn_anti,
        "result": bn_sym,
    }

    # --- Evaluacion numerica de coeficientes 1..N ---
    a0_num = float(a0_sym.evalf())

    # Muestreo numerico de f para el fallback por integracion directa.
    f_lamb = sp.lambdify(vv, f, modules=["numpy", {"sign": np.sign, "Abs": np.abs}])
    a_n = float(a_s.evalf())
    b_n = float(b_s.evalf())
    T_n = b_n - a_n
    w0_n = 2 * np.pi / T_n
    x_s = np.linspace(a_n, b_n, 6000)
    y_s = f_lamb(x_s)
    if np.isscalar(y_s):
        y_s = np.full_like(x_s, float(y_s))

    def coeff_values(sym_expr, kind):
        """kind = 'cos' o 'sin'. Usa la formula simbolica y, si falla, integra numericamente."""
        vals = np.zeros(N)
        f_of_n = None
        if sym_expr != 0:
            f_of_n = sp.lambdify(nn, sym_expr, modules=["numpy"])
        for k in range(1, N + 1):
            val = None
            if f_of_n is not None:
                try:
                    cand = complex(f_of_n(k))
                    if np.isfinite(cand.real):
                        val = cand.real
                except Exception:
                    val = None
            if val is None:
                # Integracion numerica directa como respaldo.
                basis = np.cos(k * w0_n * x_s) if kind == "cos" else np.sin(k * w0_n * x_s)
                trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
                if trapz is None:
                    # Regla del trapecio manual como ultimo respaldo.
                    integrand = y_s * basis
                    val = (2.0 / T_n) * float(
                        np.sum((integrand[:-1] + integrand[1:]) / 2.0 * np.diff(x_s))
                    )
                else:
                    val = (2.0 / T_n) * trapz(y_s * basis, x_s)
            vals[k - 1] = val
        return vals

    an_vals = coeff_values(steps["an"]["result"], "cos")
    bn_vals = coeff_values(steps["bn"]["result"], "sin")

    return {
        "is_even": is_even,
        "is_odd": is_odd,
        "sym_ok": sym_ok,
        "steps": steps,
        "a0_num": a0_num,
        "an_vals": an_vals,
        "bn_vals": bn_vals,
    }


data = fourier_engine(
    clean(func_str), var_choice, clean(a_str), clean(b_str), N_harmonics
)

is_even = data["is_even"]
is_odd = data["is_odd"]
steps = data["steps"]
a0_num = data["a0_num"]
an_vals = data["an_vals"]
bn_vals = data["bn_vals"]

# Limpieza de los coeficientes simbolicos para presentacion tipo libro.
a0_pretty = simplify_integer_n(steps["a0"]["result"], n)
an_pretty = simplify_integer_n(steps["an"]["result"], n)
bn_pretty = simplify_integer_n(steps["bn"]["result"], n)


# ----------------------------------------------------------------------------
# EXTENSION PERIODICA Y RECONSTRUCCION NUMERICA
# ----------------------------------------------------------------------------
f_num = sp.lambdify(v, f_expr, modules=["numpy", {"sign": np.sign, "Abs": np.abs}])


def periodic_eval(x):
    """Evalua f sobre su extension periodica llevando x al intervalo base [a,b)."""
    x_base = a_val + np.mod(x - a_val, T_val)
    y = f_num(x_base)
    if np.isscalar(y):
        y = np.full_like(x_base, float(y))
    return np.asarray(y, dtype=float)


# Dominio: n_periods periodos centrados alrededor del intervalo base.
x_min = a_val - (n_periods - 1) / 2 * T_val
x_max = b_val + (n_periods - 1) / 2 * T_val
x_plot = np.linspace(x_min, x_max, 4000)

y_orig = periodic_eval(x_plot)

# Reconstruccion vectorizada: (a0/2) + sum an cos(n w0 x) + bn sin(n w0 x)
n_mat = np.arange(1, N_harmonics + 1).reshape(-1, 1)
cos_terms = np.cos(n_mat * w0_val * x_plot)
sin_terms = np.sin(n_mat * w0_val * x_plot)
fourier_series = (a0_num / 2.0) + np.sum(
    an_vals.reshape(-1, 1) * cos_terms + bn_vals.reshape(-1, 1) * sin_terms, axis=0
)


# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Grafica",
        "Desarrollo paso a paso",
        "Tabla de coeficientes",
        "Espectro",
    ]
)

# --- TAB 1: GRAFICA ---
with tab1:
     st.markdown("#### Funcion definida")
     latex_def = f"f({var_choice}) = {sp.latex(f_expr)} \\qquad {var_choice} \\in \\left[{sp.latex(a_sym)},\\; {sp.latex(b_sym)}\\right]"
    st.latex(latex_def)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_orig,
            mode="lines",
            name=f"f({var_choice}) (extension periodica)",
            line=dict(color="#2563eb", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=fourier_series,
            mode="lines",
            name=f"Aproximacion de Fourier (N={N_harmonics})",
            line=dict(color="#dc2626", width=2, dash="dash"),
        )
    )

    # Decidir si el eje usa multiplos de pi.
    if axis_format == "Multiplos de pi":
         use_pi_axis = True
    elif axis_format == "Decimales":
        use_pi_axis = False
    else:  # Automatico: usar pi solo si el periodo es multiplo racional de pi.
        try:
            ratio = sp.nsimplify(T_sym / sp.pi)
            use_pi_axis = ratio.is_rational is True
        except Exception:
            use_pi_axis = False

    xaxis = dict(title=f"{var_choice}", zeroline=True)
    if use_pi_axis:
        kmin = int(np.floor(x_min / np.pi))
        kmax = int(np.ceil(x_max / np.pi))
        tickvals = [k * np.pi for k in range(kmin, kmax + 1)]
        ticktext = []
        for k in range(kmin, kmax + 1):
            if k == 0:
                ticktext.append("0")
            elif k == 1:
                ticktext.append("π")
            elif k == -1:
                ticktext.append("-π")
            else:
                ticktext.append(f"{k}π")
        xaxis.update(dict(tickmode="array", tickvals=tickvals, ticktext=ticktext))

    fig.update_layout(
        xaxis=xaxis,
        yaxis_title="Amplitud",
        template="plotly_white",
        hovermode="x unified",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "La linea azul es la extension periodica de f y la roja punteada es la suma "
        "parcial de Fourier con N armonicos. En las discontinuidades apareceran las "
        "oscilaciones del fenomeno de Gibbs."
    )
 st.caption(f"Definicion ingresada (sintaxis SymPy):  f({var_choice}) = {func_str}")
# --- TAB 2: PASO A PASO ---
with tab2:
    st.header("Solucion de la Serie de Fourier")

    sim_txt = (
        "PAR (f(-x) = f(x))"
        if is_even
        else ("IMPAR (f(-x) = -f(x))" if is_odd else "SIN SIMETRIA par/impar")
    )

    # PASO 1
    st.markdown("### Paso 1. Identificar el periodo")
    st.write("El intervalo de definicion es:")
    st.latex(f"[a, b] = [{sp.latex(a_sym)},\\; {sp.latex(b_sym)}]")
    st.write("El periodo de la extension periodica es la longitud del intervalo:")
    st.latex(f"T = b - a = {sp.latex(b_sym)} - ({sp.latex(a_sym)}) = {sp.latex(T_sym)}")
    st.latex(f"L = \\frac{{T}}{{2}} = {sp.latex(L_sym)}")

    # PASO 2
    st.markdown("### Paso 2. Frecuencia angular fundamental")
    st.latex(
        f"\\omega_0 = \\frac{{2\\pi}}{{T}} = \\frac{{2\\pi}}{{{sp.latex(T_sym)}}} = {sp.latex(w0_sym)}"
    )

    # PASO 3
    st.markdown("### Paso 3. Forma general de la serie")
    st.latex(
        f"f({var_choice}) = \\frac{{a_0}}{{2}} + "
        f"\\sum_{{n=1}}^{{\\infty}} \\Big[a_n\\cos(n\\omega_0 {var_choice}) "
        f"+ b_n\\sin(n\\omega_0 {var_choice})\\Big]"
    )
    st.markdown("con los coeficientes de Euler-Fourier:")
    st.latex(
        f"a_0 = \\frac{{2}}{{T}}\\int_{{{sp.latex(a_sym)}}}^{{{sp.latex(b_sym)}}} f({var_choice})\\,d{var_choice}"
    )
    st.latex(
        f"a_n = \\frac{{2}}{{T}}\\int_{{{sp.latex(a_sym)}}}^{{{sp.latex(b_sym)}}} f({var_choice})\\cos(n\\omega_0 {var_choice})\\,d{var_choice}"
    )
    st.latex(
        f"b_n = \\frac{{2}}{{T}}\\int_{{{sp.latex(a_sym)}}}^{{{sp.latex(b_sym)}}} f({var_choice})\\sin(n\\omega_0 {var_choice})\\,d{var_choice}"
    )

    # PASO 4
    st.markdown("### Paso 4. Analisis de simetria")
    if not data["sym_ok"]:
        st.warning(
            "El intervalo no es simetrico respecto al origen, por lo que no se aplica "
            "la simplificacion por paridad: se calculan a0, an y bn completos."
        )
    st.write(f"Simetria detectada: **{sim_txt}**")
    if is_odd:
        st.write("Al ser impar, a0 = 0 y an = 0. Solo se calcula bn.")
    elif is_even:
        st.write("Al ser par, bn = 0. Solo se calculan a0 y an.")
    else:
        st.write("Se deben calcular a0, an y bn.")

    # PASO 5: a0 detallado
    st.markdown("### Paso 5. Calculo de $a_0$")
    if is_odd:
        st.write("Por simetria impar, la integral sobre el intervalo simetrico es cero:")
        st.latex("a_0 = 0")
    else:
        st.write("Planteamos la integral:")
        st.latex(
            f"a_0 = \\frac{{2}}{{{sp.latex(T_sym)}}}"
            f"{sp.latex(integral_pretty(steps['a0']['integrand'], v, a_sym, b_sym))}"
        )
        st.write("Antiderivada (integral indefinida):")
        st.latex(
            f"\\int {sp.latex(steps['a0']['integrand'])}\\,d{var_choice} = "
            f"{sp.latex(steps['a0']['anti'])} + C"
        )
        st.write("Evaluando en los limites y multiplicando por 2/T:")
        st.latex(f"a_0 = {sp.latex(a0_pretty)}")

    # PASO 6: an detallado
    st.markdown("### Paso 6. Calculo de $a_n$")
    if is_odd:
        st.write("Por simetria impar:")
        st.latex("a_n = 0")
    else:
        st.write("Planteamos la integral del producto con el coseno:")
        st.latex(
            f"a_n = \\frac{{2}}{{{sp.latex(T_sym)}}}"
            f"{sp.latex(integral_pretty(steps['an']['integrand'], v, a_sym, b_sym))}"
        )
        st.write("Antiderivada respecto a la variable (con n entero):")
        st.latex(
            f"\\int {sp.latex(steps['an']['integrand'])}\\,d{var_choice} = "
            f"{sp.latex(steps['an']['anti'])} + C"
        )
        st.write("Evaluando limites, simplificando y usando que n es entero:")
        st.latex(f"a_n = {sp.latex(an_pretty)}")

    # PASO 7: bn detallado
    st.markdown("### Paso 7. Calculo de $b_n$")
    if is_even:
        st.write("Por simetria par:")
        st.latex("b_n = 0")
    else:
        st.write("Planteamos la integral del producto con el seno:")
        st.latex(
            f"b_n = \\frac{{2}}{{{sp.latex(T_sym)}}}"
            f"{sp.latex(integral_pretty(steps['bn']['integrand'], v, a_sym, b_sym))}"
        )
        st.write("Antiderivada respecto a la variable (con n entero):")
        st.latex(
            f"\\int {sp.latex(steps['bn']['integrand'])}\\,d{var_choice} = "
            f"{sp.latex(steps['bn']['anti'])} + C"
        )
        st.write("Evaluando limites, simplificando y usando que n es entero:")
        st.latex(f"b_n = {sp.latex(bn_pretty)}")

    # PASO 8: serie final
    st.markdown("### Paso 8. Serie de Fourier resultante")
    st.write("Sustituyendo los coeficientes en la forma general:")
    final_parts = []
    if a0_pretty != 0:
        final_parts.append(f"\\frac{{1}}{{2}}\\left({sp.latex(a0_pretty)}\\right)")
    if an_pretty != 0:
        final_parts.append(
            f"\\sum_{{n=1}}^{{\\infty}}\\left({sp.latex(an_pretty)}\\right)\\cos(n\\omega_0 {var_choice})"
        )
    if bn_pretty != 0:
        final_parts.append(
            f"\\sum_{{n=1}}^{{\\infty}}\\left({sp.latex(bn_pretty)}\\right)\\sin(n\\omega_0 {var_choice})"
        )
    series_rhs = " + ".join(final_parts) if final_parts else "0"
    st.latex(f"f({var_choice}) = {series_rhs}")
    st.caption(f"donde w0 = {sp.latex(w0_sym)}")

# --- TAB 3: TABLA ---
with tab3:
    st.subheader("Coeficientes numericos (n = 1 .. N)")
    df = pd.DataFrame(
        {
            "n": np.arange(1, N_harmonics + 1),
            "a_n": an_vals,
            "b_n": bn_vals,
            "|C_n| = sqrt(a_n^2 + b_n^2)": np.sqrt(an_vals**2 + bn_vals**2),
        }
    )
    st.metric("a_0", f"{a0_num:.6f}")
    st.dataframe(df, height=420, use_container_width=True)

# --- TAB 4: ESPECTRO ---
with tab4:
    st.subheader("Espectro de magnitud |C_n|")
    magnitudes = np.sqrt(an_vals**2 + bn_vals**2)
    fig_spec = go.Figure(
        data=[
            go.Bar(
                x=list(range(1, N_harmonics + 1)),
                y=magnitudes,
                marker_color="#2563eb",
            )
        ]
    )
    fig_spec.update_layout(
        xaxis_title="Armonico n",
        yaxis_title="Magnitud",
        template="plotly_white",
        height=460,
    )
    st.plotly_chart(fig_spec, use_container_width=True)
