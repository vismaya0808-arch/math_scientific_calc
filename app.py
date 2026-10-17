"""
MathPilot — Scientific Learning Studio (redesigned UI/UX + feature pack)

Run:
    pip install streamlit sympy numpy pandas plotly
    streamlit run app.py

Files (keep them in the same folder):
    app.py                  this file
    mathpilot_features.py   visual explainer, step solver, adaptive practice, offline mentor,
                            equation-to-graph, snap-to-solve, exam mode (SQLite progress)

Put the provided config.toml in  .streamlit/config.toml  for the matching dark theme.
Optional LLM tutor: pip install openai, then set OPENAI_API_KEY in .streamlit/secrets.toml.
Optional Snap-to-Solve OCR: pip install pix2tex   (or pytesseract + the Tesseract binary).
"""
from __future__ import annotations

import ast
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

import mathpilot_features as F
from mathpilot_features import local_tutor  # structured, fully offline tutor

st.set_page_config(page_title="MathPilot", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

# ============================ Design system ============================
# Base #0e1522 · Panel #151e30 · Raised #1b2740 · Line #2b3a58 · Accent #4d8dff · Success #34c38f · Warning #f0b429
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{--bg:#0e1522;--panel:#151e30;--raised:#1b2740;--line:#2b3a58;--text:#e8edf7;--muted:#98a6c3;
--accent:#4d8dff;--accent-d:#3a6fd8;--good:#34c38f;--warn:#f0b429;--bad:#f06a6a;}
html,body,[class*="css"]{font-family:'IBM Plex Sans',system-ui,sans-serif;}
.stApp{background:var(--bg);color:var(--text);}
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden;}
.block-container{max-width:1320px;padding:2rem 2.2rem 3rem;}
h1,h2,h3,h4{letter-spacing:-.02em;color:var(--text);font-weight:600;}
p,li,label,span{color:inherit;}
a{color:var(--accent);}

/* Sidebar */
[data-testid="stSidebar"]{background:#0a111c;border-right:1px solid var(--line);}
[data-testid="stSidebar"] .block-container{padding:1.4rem 1rem;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:2px;}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.5rem .75rem;border-radius:8px;border-left:3px solid transparent;cursor:pointer;width:100%;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:var(--raised);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:var(--raised);border-left-color:var(--accent);}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label p{font-size:.93rem;font-weight:500;}
.mp-brand{display:flex;align-items:center;gap:.65rem;margin-bottom:.2rem;}
.mp-logo{width:36px;height:36px;border-radius:9px;background:var(--accent);display:grid;place-items:center;font-size:1.15rem;}
.mp-brand b{font-size:1.15rem;letter-spacing:-.02em;}
.mp-side-label{color:var(--muted);font-size:.8rem;margin:1.1rem 0 .3rem .2rem;}

/* Page header */
.mp-head{display:flex;align-items:center;gap:1rem;padding-bottom:1rem;margin-bottom:1.4rem;border-bottom:1px solid var(--line);}
.mp-head .ic{width:46px;height:46px;border-radius:12px;background:var(--raised);border:1px solid var(--line);display:grid;place-items:center;font-size:1.4rem;}
.mp-head h2{margin:0;font-size:1.55rem;}
.mp-head p{margin:.1rem 0 0;color:var(--muted);font-size:.93rem;}

/* Hero */
.mp-hero{padding:1.8rem 2rem;border:1px solid var(--line);border-radius:16px;background:var(--panel);margin-bottom:1.4rem;position:relative;overflow:hidden;}
.mp-hero::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--accent);}
.mp-hero h1{font-size:2rem;margin:0 0 .4rem;font-weight:700;}
.mp-hero p{color:var(--muted);max-width:62ch;margin:0;line-height:1.6;}

/* Cards */
.mp-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.1rem 1.2rem;height:100%;}
.mp-card h3{margin:.5rem 0 .3rem;font-size:1.05rem;}
.mp-card p{margin:0;color:var(--muted);font-size:.9rem;line-height:1.5;}
.mp-card .ic{font-size:1.4rem;}
.mp-chip{display:inline-block;color:#b9d2ff;background:#17305a;border:1px solid #2a4d86;border-radius:6px;padding:.15rem .6rem;font-size:.78rem;font-weight:500;}

/* Calculator display */
.mp-display{background:#0a111c;border:1px solid var(--line);border-radius:12px;padding:1rem 1.3rem;min-height:104px;
display:flex;flex-direction:column;justify-content:center;align-items:flex-end;margin-bottom:.8rem;}
.mp-display .expr{font-family:'IBM Plex Mono',monospace;color:var(--muted);font-size:.95rem;word-break:break-all;}
.mp-display .res{font-family:'IBM Plex Mono',monospace;color:var(--text);font-size:2.1rem;font-weight:500;word-break:break-all;}
.mp-display .res.err{color:var(--bad);font-size:1rem;font-weight:400;}
.mp-display .res.ok{color:var(--warn);}
.mp-equation{font-family:'IBM Plex Mono',monospace;background:#0a111c;border:1px solid var(--line);border-left:3px solid var(--accent);padding:.8rem 1rem;border-radius:8px;overflow-x:auto;}

/* Inputs */
.stTextInput label,.stSelectbox label,.stNumberInput label,.stSlider label,.stTextArea label,.stRadio > label{color:#c7d3ea !important;font-weight:500;}
div[data-baseweb="select"] > div,div[data-baseweb="input"] > div,div[data-baseweb="textarea"] > div{background:#0f1a2e !important;border-color:var(--line) !important;border-radius:8px !important;}
div[data-baseweb="input"] input,textarea{font-family:'IBM Plex Mono',monospace !important;color:var(--text) !important;}
div[data-baseweb="input"]:focus-within > div,div[data-baseweb="textarea"]:focus-within > div{border-color:var(--accent) !important;box-shadow:0 0 0 1px var(--accent);}

/* Buttons: primary = filled accent, secondary = quiet raised surface */
.stButton > button,.stFormSubmitButton > button,.stDownloadButton > button{border-radius:8px;border:1px solid var(--line);background:var(--raised);color:var(--text);font-weight:500;min-height:2.5rem;transition:background .12s,border-color .12s;}
.stButton > button:hover,.stDownloadButton > button:hover{background:#24345a;border-color:var(--accent);color:#fff;}
.stButton > button:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
.stButton > button[kind="primary"],.stFormSubmitButton > button[kind="primary"]{background:var(--accent);border-color:var(--accent);color:#06101f;font-weight:600;}
.stButton > button[kind="primary"]:hover{background:#6ba1ff;border-color:#6ba1ff;color:#06101f;}

/* Metrics, tabs, expanders, tables, alerts */
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);padding:.9rem 1.1rem;border-radius:12px;}
[data-testid="stMetricLabel"] p{color:var(--muted) !important;font-size:.85rem;}
[data-testid="stMetricValue"]{color:var(--text) !important;font-weight:600;}
.stTabs [data-baseweb="tab-list"]{gap:.25rem;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:2.7rem;padding:0 1rem;color:var(--muted);font-weight:500;}
.stTabs [aria-selected="true"]{color:var(--text);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent);}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:10px;background:var(--panel);}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden;}
.stAlert{border-radius:10px;}
hr{border-color:var(--line) !important;margin:1.4rem 0 !important;}
code,pre{font-family:'IBM Plex Mono',monospace !important;}
.mp-footer{color:#6f7f9f;font-size:.8rem;text-align:center;padding:1.2rem 0 0;}
@media (max-width:760px){.block-container{padding:1rem;}.mp-hero h1{font-size:1.5rem;}}
@media (prefers-reduced-motion:reduce){*{transition:none !important;}}
</style>
""", unsafe_allow_html=True)

# ============================ State & navigation ============================
PAGES = {
    "Overview": ("🏠", "Overview", "Your learning workspace at a glance."),
    "AI Math Copilot": ("💬", "AI Math Copilot", "Ask in plain language and get a structured explanation."),
    "Offline Math Mentor": ("🎓", "Offline Math Mentor", "Concepts, worked examples and hints with no internet."),
    "Scientific Calculator": ("🧮", "Scientific Calculator", "Keypad or keyboard, with real degree and radian modes."),
    "Visual Explainer": ("🔭", "Visual Math Explainer", "Watch a function, its tangent and its derivative move together."),
    "Step-by-Step Solver": ("🪜", "Step-by-Step Solver", "Worked solutions with factoring, zero-product rule and verification."),
    "Symbolic Solver": ("∫", "Symbolic Solver", "Exact calculus and algebra powered by SymPy."),
    "Equation-to-Graph Lab": ("📐", "Equation-to-Graph Lab", "Graph, roots, turning points and numerical root finding."),
    "Graphing Lab": ("📈", "Graphing Lab", "Plot up to three functions and find roots."),
    "Snap-to-Solve": ("📷", "Snap-to-Solve", "Photograph an equation, correct it, solve it."),
    "Matrix Lab": ("▦", "Matrix Lab", "Arithmetic, determinants, inverses and eigenvalues."),
    "Statistics Lab": ("📊", "Statistics Lab", "Summarise a dataset and see its distribution."),
    "Practice Arena": ("🎯", "Adaptive Practice", "Questions that adapt to your weak topics, with a mastery map."),
    "Exam Mode": ("📝", "Exam Mode", "Timed papers, negative marking, reports and saved progress."),
    "Calculator Handbook": ("📚", "Calculator Handbook", "Quick reference for common calculator keys."),
    "Session Insights": ("🧠", "Session Insights", "Review your activity and get next steps."),
}

def init_state():
    defaults = {
        "page": "Overview", "history": [], "quiz_score": 0, "quiz_attempts": 0, "quiz_streak": 0,
        "best_streak": 0, "chat": [],
        "session_started": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "calc_expr": "", "calc_out": None, "ans": 0, "angle_mode": "Degrees",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()

def goto(page):
    st.session_state.page = page

def page_header(page):
    icon, title, sub = PAGES[page]
    st.markdown(f'<div class="mp-head"><div class="ic">{icon}</div><div><h2>{title}</h2><p>{sub}</p></div></div>',
                unsafe_allow_html=True)

# ============================ Safe expression engine ============================
ALLOWED_AST = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.FloorDiv)

def build_env(deg: bool, ans):
    to_rad = math.radians if deg else (lambda v: v)
    to_ang = math.degrees if deg else (lambda v: v)
    funcs = {
        "sin": lambda v: math.sin(to_rad(v)), "cos": lambda v: math.cos(to_rad(v)), "tan": lambda v: math.tan(to_rad(v)),
        "asin": lambda v: to_ang(math.asin(v)), "acos": lambda v: to_ang(math.acos(v)), "atan": lambda v: to_ang(math.atan(v)),
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "sqrt": math.sqrt,
        "log": math.log10, "log10": math.log10, "log2": math.log2, "ln": math.log, "exp": math.exp,
        "abs": abs, "floor": math.floor, "ceil": math.ceil, "round": round, "factorial": math.factorial,
        "degrees": math.degrees, "radians": math.radians, "pow": pow, "min": min, "max": max,
        "nPr": math.perm, "nCr": math.comb,
    }
    consts = {"pi": math.pi, "e": math.e, "tau": math.tau, "ans": ans}
    return funcs, consts

def safe_calculate(expression: str, deg: bool = True, ans=0):
    expression = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    if not expression:
        raise ValueError("Enter an expression, for example sin(30) + sqrt(144).")
    if len(expression) > 180:
        raise ValueError("Expression is too long (180 characters max).")
    funcs, consts = build_env(deg, ans)
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST):
            raise ValueError("Unsupported syntax. Use numbers, operators and the listed functions.")
        if isinstance(node, ast.Name) and node.id not in funcs and node.id not in consts:
            raise ValueError(f"Unknown name '{node.id}'. Try sin, sqrt, log, pi, e or ans.")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in funcs or node.keywords):
            raise ValueError("Only the supported functions can be called, with positional arguments.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Only numbers are allowed.")
    # guard against huge powers
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and abs(node.right.value) > 10000:
                raise ValueError("Exponent is too large.")
    result = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, {**funcs, **consts})
    if isinstance(result, float):
        if not math.isfinite(result):
            raise ValueError("The result is not finite.")
        if abs(result) < 1e-12:
            result = 0.0
    return result

def fmt(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10g}"
    return str(value)

def record_calculation(expression, result):
    st.session_state.history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"),
                                        "expression": expression, "result": fmt(result)})
    st.session_state.history = st.session_state.history[:50]

# ---- calculator callbacks (run before the page re-renders, so they can edit widget state) ----
def run_calc(expr=None):
    expr = st.session_state.calc_expr if expr is None else expr
    try:
        res = safe_calculate(expr, st.session_state.angle_mode == "Degrees", st.session_state.ans)
        st.session_state.calc_out = (expr, fmt(res), None)
        st.session_state.ans = res
        record_calculation(expr, res)
    except ZeroDivisionError:
        st.session_state.calc_out = (expr, "", "Division by zero is undefined.")
    except Exception as exc:
        st.session_state.calc_out = (expr, "", f"{exc}")

KEY_TEXT = {"÷": "/", "×": "*", "−": "-", "Ans": "ans"}

def keypad(token):
    e = st.session_state.calc_expr
    if token == "C":
        st.session_state.calc_expr, st.session_state.calc_out = "", None
    elif token == "⌫":
        st.session_state.calc_expr = re.sub(r"[A-Za-z0-9]+\($|.$", "", e) if e else ""
    elif token == "=":
        run_calc()
    else:
        st.session_state.calc_expr = e + KEY_TEXT.get(token, token)

def load_expr(expr):
    st.session_state.calc_expr = expr

# ============================ Symbolic helpers ============================
x, y, z = sp.symbols("x y z", real=True)
SYMPY_LOCALS = {"x": x, "y": y, "z": z, "pi": sp.pi, "E": sp.E, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh,
                "tanh": sp.tanh, "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log, "exp": sp.exp, "Abs": sp.Abs, "abs": sp.Abs}

def parse_symbolic(text):
    text = text.strip().replace("^", "**")
    if len(text) > 180:
        raise ValueError("Expression is too long.")
    if not re.fullmatch(r"[0-9A-Za-z_+\-*/().,\s\[\]]+", text):
        raise ValueError("Use standard mathematical notation, e.g. x**2 + 3*x.")
    return sp.sympify(text, locals=SYMPY_LOCALS, evaluate=True)

def symbolic_steps(expr, operation, point=None):
    if operation == "Differentiate":
        out = sp.diff(expr, x)
        steps = [f"Original function: f(x) = {sp.sstr(expr)}", f"Apply differentiation rules: f'(x) = {sp.sstr(out)}"]
        if point is not None:
            steps.append(f"Substitute x = {point:g}: f'({point:g}) = {sp.sstr(out.subs(x, sp.nsimplify(point)))}")
    elif operation == "Integrate (indefinite)":
        out = sp.integrate(expr, x)
        steps = [f"Original integrand: {sp.sstr(expr)}", f"Integrate term by term: ∫f(x)dx = {sp.sstr(out)} + C"]
    elif operation == "Simplify":
        out = sp.simplify(expr); steps = [f"Input: {sp.sstr(expr)}", f"Simplified: {sp.sstr(out)}"]
    elif operation == "Factor":
        out = sp.factor(expr); steps = [f"Input: {sp.sstr(expr)}", f"Factorised: {sp.sstr(out)}"]
    elif operation == "Expand":
        out = sp.expand(expr); steps = [f"Input: {sp.sstr(expr)}", f"Expanded: {sp.sstr(out)}"]
    else:
        raise ValueError("Unknown operation.")
    return out, steps

# ============================ Tutor ============================
# local_tutor() is imported from mathpilot_features (offline, rule-based, topic-specific).
def tutor_response(question):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            out = client.chat.completions.create(model="gpt-4o-mini", temperature=0.25, messages=[
                {"role": "system", "content": "You are MathPilot, a patient scientific-calculator and mathematics tutor for engineering students. Explain clearly with numbered steps, state assumptions and units, and never claim a numerical answer was verified unless you computed it. Casio key sequences vary by model. Be concise."},
                {"role": "user", "content": question}])
            return out.choices[0].message.content or local_tutor(question)
    except Exception:
        pass
    return local_tutor(question)

def ask(prompt):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.session_state.chat.append({"role": "assistant", "content": tutor_response(prompt)})

# ============================ Plot helper ============================
PLOT_COLORS = ["#4d8dff", "#f0b429", "#34c38f"]

def style_fig(fig, height=480):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f1a2e", height=height,
                      font=dict(family="IBM Plex Sans", color="#c7d3ea"), margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1), colorway=PLOT_COLORS)
    fig.update_xaxes(showgrid=True, gridcolor="#223250", zeroline=True, zerolinecolor="#5a6e94")
    fig.update_yaxes(showgrid=True, gridcolor="#223250", zeroline=True, zerolinecolor="#5a6e94")
    return fig

# ============================ Sidebar ============================
with st.sidebar:
    st.markdown('<div class="mp-brand"><div class="mp-logo">🧭</div><b>MathPilot</b></div>', unsafe_allow_html=True)
    st.caption("Scientific learning studio")
    st.markdown('<div class="mp-side-label">Workspace</div>', unsafe_allow_html=True)
    st.radio("Navigate", list(PAGES), key="page", label_visibility="collapsed",
             format_func=lambda p: f"{PAGES[p][0]}  {p}")
    st.markdown('<div class="mp-side-label">Session</div>', unsafe_allow_html=True)
    st.caption(f"Started {st.session_state.session_started}")
    if st.button("Clear calculation history", use_container_width=True):
        st.session_state.history = []
        st.toast("Calculation history cleared")
        st.rerun()
    st.caption("A learning aid. Always double-check answers you rely on for exams.")

page = st.session_state.page

# ============================ Overview ============================
if page == "Overview":
    st.markdown("""<div class="mp-hero"><h1>From button memorising to real understanding</h1>
    <p>Calculate, visualise, practise and review in one place. Start with the calculator, watch a derivative come to life,
    follow a worked solution, or sit a timed exam. Your practice and exam progress is saved on this machine.</p></div>""", unsafe_allow_html=True)
    attempts = st.session_state.quiz_attempts
    acc = 100 * st.session_state.quiz_score / attempts if attempts else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calculations", len(st.session_state.history))
    c2.metric("Practice accuracy", f"{acc:.0f}%" if attempts else "–")
    c3.metric("Questions attempted", attempts)
    c4.metric("Best streak", st.session_state.best_streak)
    st.markdown("#### Jump into a workspace")
    cards = [("💬", "AI Math Copilot", "Ask questions and get structured explanations."),
             ("🧮", "Scientific Calculator", "Keypad with degree and radian modes."),
             ("🔭", "Visual Explainer", "Move a point and watch the tangent and slope change."),
             ("🪜", "Step-by-Step Solver", "Factor, apply the zero-product rule and verify."),
             ("📐", "Equation-to-Graph Lab", "Roots, turning points and numerical solving."),
             ("📷", "Snap-to-Solve", "Photograph an equation and solve it."),
             ("🎓", "Offline Math Mentor", "Concepts, examples and hints, no internet needed."),
             ("🎯", "Practice Arena", "Adaptive questions and a topic mastery map."),
             ("📝", "Exam Mode", "Timed papers, negative marking and reports.")]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f'<div class="mp-card"><div class="ic">{icon}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
            st.button(f"Open {title}", key=f"open_{i}", on_click=goto, args=(title,), use_container_width=True)
            st.write("")
    st.markdown("#### How it helps you learn")
    l1, l2, l3 = st.columns(3)
    l1.markdown("**Learn loop**  \nExplain, solve, visualise, practise, review.")
    l2.markdown("**Safe evaluation**  \nExpressions are parsed and restricted, never run as arbitrary Python.")
    l3.markdown("**Verify results**  \nCheck symbolic answers numerically and inspect them on a graph.")

# ============================ Copilot ============================
elif page == "AI Math Copilot":
    page_header(page)
    if (prompt := st.chat_input("Ask about a function, calculator key or concept…")):
        ask(prompt)
    if not st.session_state.chat:
        st.info("Start with a suggested question, or type your own below.")
        sugg = ["Explain why sin(30°) = 0.5", "How do I find a matrix determinant?", "Explain derivatives with an example"]
        cols = st.columns(3)
        for i, p in enumerate(sugg):
            cols[i].button(p, key=f"copilot_{i}", on_click=ask, args=(p,), use_container_width=True)
    for msg in st.session_state.chat[-20:]:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🧭"):
            st.markdown(msg["content"])
    if st.session_state.chat:
        st.button("Clear conversation", on_click=lambda: st.session_state.update(chat=[]))

# ============================ Offline mentor ============================
elif page == "Offline Math Mentor":
    page_header(page)
    F.render_offline_mentor()

# ============================ Calculator ============================
elif page == "Scientific Calculator":
    page_header(page)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.radio("Angle mode", ["Degrees", "Radians"], key="angle_mode", horizontal=True)
        out = st.session_state.calc_out
        if out is None:
            body = '<div class="expr">&nbsp;</div><div class="res">0</div>'
        elif out[2]:
            body = f'<div class="expr">{out[0]}</div><div class="res err">{out[2]}</div>'
        else:
            body = f'<div class="expr">{out[0]} =</div><div class="res ok">{out[1]}</div>'
        st.markdown(f'<div class="mp-display">{body}</div>', unsafe_allow_html=True)
        st.text_input("Expression", key="calc_expr", on_change=run_calc, label_visibility="collapsed",
                      placeholder="Type or tap keys, then press Enter. Example: sin(30) + sqrt(144)/2")
        rows = [["sin(", "cos(", "tan(", "(", ")"], ["asin(", "acos(", "atan(", "^", "sqrt("],
                ["log(", "ln(", "exp(", "pi", "e"], ["7", "8", "9", "÷", "⌫"],
                ["4", "5", "6", "×", "C"], ["1", "2", "3", "−", "Ans"], ["0", ".", "+", "factorial(", "="]]
        for r, row in enumerate(rows):
            for c, tok in zip(st.columns(5), row):
                label = tok[:-1] if tok.endswith("(") and len(tok) > 1 else tok
                label = "n!" if tok == "factorial(" else label
                c.button(label, key=f"k_{r}_{tok}", on_click=keypad, args=(tok,), use_container_width=True,
                         type="primary" if tok == "=" else "secondary")
        st.caption("Use `ans` for the previous result. Also available: pow, nPr, nCr, log2, abs, round.")
    with right:
        st.markdown("#### Quick examples")
        examples = [("sin(30)", "sin 30°"), ("sqrt(144)", "√144"), ("log(1000)", "log₁₀ 1000"),
                    ("ln(e)", "ln e"), ("factorial(7)", "7!"), ("pi*5^2", "πr², r = 5"),
                    ("nCr(5,2)", "5C2"), ("pow(2,10)", "2¹⁰")]
        ec = st.columns(2)
        for i, (ex, label) in enumerate(examples):
            ec[i % 2].button(label, key=f"ex_{i}", on_click=load_expr, args=(ex,), use_container_width=True)
        st.markdown("#### History")
        if st.session_state.history:
            for i, h in enumerate(st.session_state.history[:6]):
                st.button(f"{h['expression']}  =  {h['result']}", key=f"hist_{i}", on_click=load_expr,
                          args=(h["expression"],), use_container_width=True, help="Click to reuse this expression")
            with st.expander("Full history"):
                st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        else:
            st.info("No calculations yet. Press a key or pick an example to begin.")

# ============================ Visual explainer ============================
elif page == "Visual Explainer":
    page_header(page)
    F.render_visual_explainer()

# ============================ Step-by-step solver ============================
elif page == "Step-by-Step Solver":
    page_header(page)
    F.render_step_solver()

# ============================ Symbolic ============================
elif page == "Symbolic Solver":
    page_header(page)
    tab1, tab2, tab3 = st.tabs(["Calculus and algebra", "Equation solver", "Numerical check"])
    with tab1:
        a, b = st.columns([1, 2])
        op = a.selectbox("Operation", ["Differentiate", "Integrate (indefinite)", "Simplify", "Factor", "Expand"])
        source = b.text_input("Expression in x", "x**3 + 2*x**2 - 5*x + 1", help="Use ** for powers, e.g. x**2")
        point_text = ""
        if op == "Differentiate":
            point_text = st.text_input("Evaluate the derivative at x = (optional)", "2")
        if st.button("Run operation", key="symbolic_run", type="primary"):
            try:
                point = float(point_text) if point_text.strip() else None
                out, steps = symbolic_steps(parse_symbolic(source), op, point)
                st.markdown("##### Result")
                st.latex(sp.latex(out) + (r" + C" if op.startswith("Integrate") else ""))
                st.markdown("##### Working")
                for i, s in enumerate(steps, 1):
                    st.markdown(f"{i}. {s}")
            except Exception as exc:
                st.error(f"Could not process the expression: {exc}. Check brackets and use * for multiplication.")
    with tab2:
        equation = st.text_input("Equation (use =)", "x**2 - 5*x + 6 = 0")
        if st.button("Solve equation", key="eqsolve", type="primary"):
            try:
                if equation.count("=") != 1:
                    raise ValueError("Enter exactly one equals sign, e.g. x**2 - 5*x + 6 = 0")
                lhs, rhs = equation.split("=")
                sols = sp.solve(sp.Eq(parse_symbolic(lhs), parse_symbolic(rhs)), x)
                if sols:
                    for i, s in enumerate(sols, 1):
                        st.latex(f"x_{{{i}}} = {sp.latex(s)}")
                    st.caption(f"{len(sols)} solution(s). Check domain restrictions for logs, denominators and radicals. "
                               "For a full worked solution, use the Step-by-Step Solver.")
                else:
                    st.warning("No solution found in the supported domain.")
            except Exception as exc:
                st.error(f"Could not solve: {exc}")
    with tab3:
        a, b = st.columns([2, 1])
        verify_expr = a.text_input("Function f(x)", "x**2 - 5*x + 6")
        test_x = b.number_input("Test x", value=2.0, step=0.5)
        if st.button("Evaluate", key="verify", type="primary"):
            try:
                exact = sp.simplify(parse_symbolic(verify_expr).subs(x, sp.Rational(str(test_x))))
                m1, m2 = st.columns(2)
                m1.metric(f"f({test_x:g})", fmt(float(sp.N(exact, 12))))
                m2.metric("Exact form", sp.sstr(exact))
            except Exception as exc:
                st.error(f"Verification failed: {exc}")

# ============================ Equation-to-graph ============================
elif page == "Equation-to-Graph Lab":
    page_header(page)
    F.render_equation_graph()

# ============================ Graphing ============================
elif page == "Graphing Lab":
    page_header(page)
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        f1 = st.text_input("f(x)", "sin(x)")
        f2 = st.text_input("g(x)  (optional)", "x/3")
        f3 = st.text_input("h(x)  (optional)", "")
    with c2:
        xmin = st.number_input("x minimum", value=-10.0, step=1.0)
        xmax = st.number_input("x maximum", value=10.0, step=1.0)
        points = st.slider("Resolution", 100, 1200, 500, 100)
    if xmin >= xmax:
        st.error("x minimum must be smaller than x maximum.")
    else:
        xs = np.linspace(xmin, xmax, points)
        fig = go.Figure()
        notes = []
        for label, src in (("f(x)", f1), ("g(x)", f2), ("h(x)", f3)):
            if not src.strip():
                continue
            try:
                sym = parse_symbolic(src)
                fn = sp.lambdify(x, sym, modules=["numpy"])
                with np.errstate(all="ignore"):
                    ys = np.asarray(fn(xs), dtype=float)
                if ys.ndim == 0:
                    ys = np.full_like(xs, float(ys))
                ys = np.where(np.isfinite(ys) & (np.abs(ys) < 1e6), ys, np.nan)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{label} = {src}", line=dict(width=2.5)))
                roots = []
                try:
                    for r in sp.solve(sp.Eq(sym, 0), x):
                        v = complex(sp.N(r))
                        if abs(v.imag) < 1e-8 and xmin <= v.real <= xmax:
                            roots.append(round(v.real, 6))
                except Exception:
                    pass
                if roots:
                    notes.append(f"**{label}** crosses zero at x = {', '.join(map(str, sorted(set(roots))))}")
                    fig.add_trace(go.Scatter(x=sorted(set(roots)), y=[0] * len(set(roots)), mode="markers",
                                             marker=dict(size=9, color="#f0b429"), name=f"{label} roots", showlegend=False))
            except Exception as exc:
                st.warning(f"Could not plot {label}: {exc}")
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)
        for n in notes:
            st.markdown(n)
        st.caption("Roots appear only when SymPy can solve them exactly. For numerical root finding, "
                   "turning points and intercepts, open the Equation-to-Graph Lab.")

# ============================ Snap-to-solve ============================
elif page == "Snap-to-Solve":
    page_header(page)
    F.render_snap_to_solve()

# ============================ Matrix ============================
elif page == "Matrix Lab":
    page_header(page)
    st.caption("Enter each row on its own line, with values separated by commas.")
    c1, c2 = st.columns(2)
    a_text = c1.text_area("Matrix A", "1, 2\n3, 4", height=130)
    b_text = c2.text_area("Matrix B", "5, 6\n7, 8", height=130)
    operation = st.selectbox("Operation", ["A + B", "A − B", "A × B", "det(A)", "inverse(A)", "transpose(A)", "eigenvalues(A)", "rank(A)"])

    def read_matrix(txt):
        rows = [[float(v) for v in line.split(",")] for line in txt.strip().splitlines() if line.strip()]
        if not rows or any(len(r) != len(rows[0]) for r in rows):
            raise ValueError("Each row must have the same number of values, and the matrix cannot be empty.")
        arr = np.array(rows)
        if arr.size > 100 or not np.all(np.isfinite(arr)):
            raise ValueError("Matrix is too large or has non-finite values.")
        return arr

    if st.button("Compute", type="primary"):
        try:
            A, B = read_matrix(a_text), read_matrix(b_text)
            sq = A.shape[0] == A.shape[1]
            if operation in ("A + B", "A − B"):
                if A.shape != B.shape: raise ValueError("Both matrices must have the same dimensions.")
                result = A + B if operation == "A + B" else A - B
            elif operation == "A × B":
                if A.shape[1] != B.shape[0]: raise ValueError("Columns of A must equal rows of B.")
                result = A @ B
            elif operation == "transpose(A)":
                result = A.T
            elif operation == "rank(A)":
                result = np.linalg.matrix_rank(A)
            else:
                if not sq: raise ValueError("This operation needs a square matrix A.")
                if operation == "det(A)": result = np.linalg.det(A)
                elif operation == "eigenvalues(A)": result = np.linalg.eigvals(A)
                else:
                    if abs(np.linalg.det(A)) < 1e-12: raise ValueError("Matrix is singular, so no inverse exists.")
                    result = np.linalg.inv(A)
            st.markdown("##### Result")
            if np.isscalar(result) or np.ndim(result) == 0:
                st.metric(operation, fmt(result))
            else:
                st.dataframe(pd.DataFrame(np.atleast_2d(result)), use_container_width=True, hide_index=True)
            st.caption(f"A is {A.shape[0]}×{A.shape[1]}, B is {B.shape[0]}×{B.shape[1]}")
        except Exception as exc:
            st.error(f"Matrix operation failed: {exc}")

# ============================ Statistics ============================
elif page == "Statistics Lab":
    page_header(page)
    raw = st.text_area("Dataset", "12, 15, 15, 18, 21, 24, 30", height=100, help="Separate values with commas, spaces or new lines.")
    if st.button("Analyse dataset", type="primary"):
        try:
            vals = [float(v) for v in re.split(r"[,;\s]+", raw.strip()) if v]
            if not vals: raise ValueError("Enter at least one number.")
            if len(vals) > 10000 or not all(math.isfinite(v) for v in vals): raise ValueError("Dataset is too large or has non-finite values.")
            arr = np.array(vals)
            c = st.columns(5)
            c[0].metric("Count", len(arr)); c[1].metric("Mean", fmt(np.mean(arr))); c[2].metric("Median", fmt(np.median(arr)))
            c[3].metric("Std dev (pop.)", fmt(np.std(arr)))
            c[4].metric("Std dev (sample)", fmt(np.std(arr, ddof=1)) if len(arr) > 1 else "–")
            q1, q3 = np.percentile(arr, [25, 75])
            st.markdown(f"**Min** {fmt(arr.min())} · **Q1** {fmt(q1)} · **Q3** {fmt(q3)} · **Max** {fmt(arr.max())} · **Range** {fmt(np.ptp(arr))}")
            fig = go.Figure(go.Histogram(x=arr, nbinsx=min(20, max(5, int(math.sqrt(len(arr))))), marker_color="#4d8dff"))
            fig.update_layout(xaxis_title="Value", yaxis_title="Frequency")
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not analyse the dataset: {exc}")

# ============================ Practice (adaptive) ============================
elif page == "Practice Arena":
    page_header(page)
    F.render_adaptive_practice()

# ============================ Exam mode ============================
elif page == "Exam Mode":
    page_header(page)
    F.render_exam_mode()

# ============================ Handbook ============================
elif page == "Calculator Handbook":
    page_header(page)
    st.caption("Key sequences vary between fx-991ES Plus, fx-991EX and ClassWiz models. Check your manual for exact keys.")
    handbook = {
        "Angle mode (DEG/RAD)": ("Use DEG for degree-based problems and RAD when the problem or calculus convention says so.", "A wrong mode is the most common cause of a wrong trig answer."),
        "sin / cos / tan": ("sin = opposite/hypotenuse, cos = adjacent/hypotenuse, tan = opposite/adjacent.", "Check the angle unit before calculating."),
        "Inverse trigonometry": ("sin⁻¹, cos⁻¹ and tan⁻¹ return angles from ratios, in the current angle unit.", "asin and acos accept inputs between -1 and 1."),
        "log and ln": ("log is base 10 on most scientific calculators; ln is base e.", "log(1000) = 3 and ln(e) = 1."),
        "SHIFT and ALPHA": ("SHIFT reaches the secondary function; ALPHA reaches labels or variables.", "Match the colour of the printed label to the key."),
        "Factorial, nPr, nCr": ("Factorial counts arrangements of all n objects. nPr is ordered selection, nCr is unordered.", "5P3 = 60 and 5C2 = 10."),
        "Matrices": ("Matrix mode works within your model's size limits.", "For A × B, columns of A must equal rows of B."),
        "Calculus": ("Many models give numerical derivatives and integrals but not symbolic results.", "Use the Symbolic Solver for exact answers."),
        "Scientific notation": ("EXP or ×10ˣ enters a power of ten.", "6.02 EXP 23 means 6.02 × 10²³."),
    }
    term = st.text_input("Search the handbook", placeholder="e.g. log, matrix, SHIFT")
    shown = {k: v for k, v in handbook.items() if term.lower() in (k + v[0]).lower()}
    if not shown:
        st.info("No topics match your search. Try a shorter keyword.")
    for title, (desc, tip) in shown.items():
        with st.expander(title):
            st.write(desc)
            st.caption("Tip: " + tip)

# ============================ Insights ============================
elif page == "Session Insights":
    page_header(page)
    attempts = st.session_state.quiz_attempts
    acc = 100 * st.session_state.quiz_score / attempts if attempts else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Session start", st.session_state.session_started)
    c2.metric("Calculations", len(st.session_state.history))
    c3.metric("Quiz accuracy", f"{acc:.1f}%" if attempts else "–")
    st.markdown("#### Recent activity")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download history (CSV)", df.to_csv(index=False).encode("utf-8"), "mathpilot_history.csv", "text/csv")
    else:
        st.info("No calculations yet. Open the Scientific Calculator to get started.")
    st.markdown("#### Saved topic progress")
    topic_df = F.topic_stats()
    if topic_df["Attempts"].sum() == 0:
        st.info("No saved practice or exam answers yet. Try the Practice Arena or Exam Mode.")
    else:
        st.dataframe(topic_df, use_container_width=True, hide_index=True, column_config={
            "Mastery": st.column_config.ProgressColumn("Mastery", min_value=0, max_value=100, format="%d%%")})
        weak = topic_df[topic_df.Status == "Revise"]
        if not weak.empty:
            st.warning("Revision recommended: " + ", ".join(weak.Topic))
    st.markdown("#### Suggested next step")
    if attempts == 0: st.write("Take a short practice quiz to set your baseline.")
    elif acc < 60: st.write("Review the explanations for questions you missed, then retry the same topic.")
    elif acc < 85: st.write("You have a solid base. Try mixed topics and explain each solution in your own words.")
    else: st.write("Strong accuracy. Move to harder engineering problems and verify results two ways.")
    st.caption("Calculation history is kept for this browser session. Practice and exam progress is saved in mathpilot.db.")

st.markdown('<div class="mp-footer">MathPilot · Built with Streamlit, SymPy, NumPy, Pandas and Plotly · Prototype for academic demonstration</div>', unsafe_allow_html=True)