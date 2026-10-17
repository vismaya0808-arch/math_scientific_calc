"""
MathPilot feature pack
======================
1. Visual Math Explainer      -> render_visual_explainer()
2. Step-by-Step Solver        -> render_step_solver()
3. Adaptive Practice Engine   -> render_adaptive_practice()
4. Offline Math Mentor        -> render_offline_mentor() + local_tutor()
5. Equation-to-Graph Lab      -> render_equation_graph()
6. Snap-to-Solve              -> render_snap_to_solve()
7. Exam Mode + Progress       -> render_exam_mode()

Progress (attempts, exams) is stored in SQLite (mathpilot.db next to this file,
or the path in the MATHPILOT_DB environment variable).
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import re
import sqlite3
import time
import warnings
from contextlib import closing
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from plotly.subplots import make_subplots
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

x = sp.Symbol("x")
PLOT_COLORS = ["#4d8dff", "#f0b429", "#34c38f", "#f06a6a"]

# =====================================================================
# Shared helpers: safe parsing, numeric functions, plot styling
# =====================================================================
_NAMES = {"x": x, "pi": sp.pi, "e": sp.E, "E": sp.E, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
          "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh,
          "tanh": sp.tanh, "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log, "exp": sp.exp,
          "abs": sp.Abs, "Abs": sp.Abs}
_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)
_UNICODE = {"−": "-", "–": "-", "×": "*", "·": "*", "÷": "/", "²": "^2", "³": "^3", "√": "sqrt", "π": "pi"}


def normalise(text: str) -> str:
    for k, v in _UNICODE.items():
        text = text.replace(k, v)
    return text.strip()


def parse_expr_safe(text: str) -> sp.Expr:
    """Parse a maths expression in x. Whitelisted characters and names only."""
    t = normalise(text)
    if not t:
        raise ValueError("Enter an expression, for example x^2 - 5x + 6.")
    if len(t) > 200:
        raise ValueError("Expression is too long (200 characters max).")
    if not re.fullmatch(r"[0-9A-Za-z+\-*/().,\s^]+", t):
        raise ValueError("Use standard notation, e.g. x^2 + 3x. Only letters, digits and + - * / ^ ( ) are allowed.")
    for name in re.findall(r"[A-Za-z_]+", t):
        if name not in _NAMES:
            raise ValueError(f"Unknown name '{name}'. Use x, pi, e, sin, cos, tan, sqrt, ln, log, exp or abs. "
                             "Put a space or * between letters, e.g. x*sin(x).")
    return parse_expr(t, local_dict=dict(_NAMES), transformations=_TRANS)


def _diff(expr):
    """d/dx treating x as real, so abs(x) differentiates to sign(x) instead of an unevaluated Derivative."""
    xr = sp.Symbol("xr", real=True)
    return sp.diff(expr.subs(x, xr), xr).subs(xr, x)


def make_fn(expr):
    """Vectorised numeric function: real results only, non-finite -> NaN."""
    f = sp.lambdify(x, expr, modules=["numpy"])

    def g(v):
        v = np.asarray(v, dtype=float)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.asarray(f(v))
            if np.iscomplexobj(out):
                out = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
            out = out.astype(float)
        if out.ndim == 0:
            out = np.full(v.shape, float(out))
        return np.where(np.isfinite(out), out, np.nan)
    return g


def style_fig(fig, height=480):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f1a2e", height=height,
                      font=dict(family="IBM Plex Sans", color="#c7d3ea"), margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08), colorway=PLOT_COLORS)
    fig.update_xaxes(showgrid=True, gridcolor="#223250", zeroline=True, zerolinecolor="#5a6e94")
    fig.update_yaxes(showgrid=True, gridcolor="#223250", zeroline=True, zerolinecolor="#5a6e94")
    return fig


def _fmt(v, digits=6):
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{float(v):.{digits}g}"


def _y_range(*arrays):
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a is not None] or [np.array([0.0])])
    if vals.size == 0:
        return None
    lo, hi = np.percentile(vals, [1, 99])
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = 0.15 * (hi - lo)
    return [float(lo - pad), float(hi + pad)]


# ---------- numerical root finding ----------
def _bisect(g, a, b, iters=100):
    fa = g(np.array([a]))[0]
    for _ in range(iters):
        m = (a + b) / 2
        fm = g(np.array([m]))[0]
        if np.isnan(fm) or np.isnan(fa):
            return None
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return (a + b) / 2


def numeric_roots(g, lo, hi, n=4001):
    """Sign-change scan + bisection. Returns (roots, poles). Poles are sign changes that are not roots."""
    xs = np.linspace(lo, hi, n)
    ys = g(xs)
    roots, poles = [], []
    for i in range(n - 1):
        a, b = ys[i], ys[i + 1]
        if np.isnan(b) and not np.isnan(a) and i + 2 < n and np.isfinite(ys[i + 2]) and a * ys[i + 2] < 0:
            poles.append(float(xs[i + 1]))  # undefined point between opposite signs, e.g. 1/x at 0
            continue
        if np.isnan(a) or np.isnan(b):
            continue
        if a == 0:
            roots.append(float(xs[i]))
        elif a * b < 0:
            r = _bisect(g, xs[i], xs[i + 1])
            if r is None:
                continue
            fr = g(np.array([r]))[0]
            if np.isfinite(fr) and abs(fr) < 1e-6 * (1 + max(abs(a), abs(b))):
                roots.append(float(r))
            else:
                poles.append(float(r))
    if ys[-1] == 0:
        roots.append(float(xs[-1]))
    return _dedupe(roots), _dedupe(poles)


def _dedupe(vals, tol=1e-6):
    out = []
    for v in sorted(vals):
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def _nice_form(r):
    """Try to recognise a numerical root as a simple exact value (rational or multiple of pi)."""
    try:
        cand = sp.nsimplify(r, [sp.pi], tolerance=1e-9, rational=False)
        if abs(float(cand) - r) < 1e-8 and (cand.is_Rational or cand.has(sp.pi)):
            return sp.latex(cand)
    except Exception:
        pass
    return None


def _real_in_range(values, lo, hi):
    out = []
    for r in values:
        try:
            v = complex(sp.N(r))
        except Exception:
            continue
        if abs(v.imag) < 1e-9 and lo <= v.real <= hi:
            out.append((float(v.real), sp.latex(sp.simplify(r))))
    return out


def find_roots(expr, lo, hi):
    """Exact (polynomials up to degree 4) plus numerical roots. Returns (rows, poles)."""
    g = make_fn(expr)
    rows = []
    exact = []
    if expr.is_polynomial(x) and 1 <= sp.Poly(expr, x).degree() <= 4:
        try:
            exact = _real_in_range(sp.solve(expr, x), lo, hi)
        except Exception:
            exact = []
    for v, ltx in exact:
        rows.append({"x": v, "exact": ltx, "method": "SymPy (exact)"})
    num, poles = numeric_roots(g, lo, hi)
    for r in num:
        if not any(abs(r - row["x"]) < 1e-5 for row in rows):
            nf = _nice_form(r)
            rows.append({"x": r, "exact": nf or "", "method": "Numerical (bisection)"})
    # repeated roots touch the axis without a sign change: look at turning points
    try:
        for tp in numeric_roots(make_fn(_diff(expr)), lo, hi)[0]:
            v = g(np.array([tp]))[0]
            if np.isfinite(v) and abs(v) < 1e-8 and not any(abs(tp - row["x"]) < 1e-5 for row in rows):
                rows.append({"x": tp, "exact": _nice_form(tp) or "", "method": "Numerical (repeated root)"})
    except Exception:
        pass
    rows.sort(key=lambda r: r["x"])
    return rows, poles


def find_turning_points(expr, lo, hi):
    g = make_fn(expr)
    d = _diff(expr)
    gd = make_fn(d)
    cands = []
    if d.is_polynomial(x) and d != 0 and sp.Poly(d, x).degree() <= 4:
        try:
            cands = [v for v, _ in _real_in_range(sp.solve(d, x), lo, hi)]
        except Exception:
            cands = []
    cands = _dedupe(cands + numeric_roots(gd, lo, hi)[0])
    delta = (hi - lo) * 1e-4
    out = []
    for c in cands:
        left, right = gd(np.array([c - delta]))[0], gd(np.array([c + delta]))[0]
        if left > 0 > right:
            kind = "local maximum"
        elif left < 0 < right:
            kind = "local minimum"
        else:
            kind = "stationary inflection"
        y = g(np.array([c]))[0]
        if np.isfinite(y):
            out.append({"x": float(c), "y": float(y), "type": kind})
    return out


# =====================================================================
# 1. Visual Math Explainer
# =====================================================================
def render_visual_explainer():
    c1, c2 = st.columns([2, 1], gap="large")
    src = c1.text_input("Function f(x)", "x^2", key="ve_src", help="Examples: x^2, sin(x), x^3 - 3x, exp(x)")
    lo, hi = c2.slider("Plot range", -20.0, 20.0, (-5.0, 5.0), 0.5, key="ve_range")
    try:
        f = parse_expr_safe(src)
        fp = _diff(f)
    except Exception as exc:
        st.error(f"Could not read the function: {exc}")
        return
    g, gp = make_fn(f), make_fn(fp)

    s1, s2 = st.columns(2)
    x0 = s1.slider("Move the point x", float(lo), float(hi), float(min(max(1.0, lo), hi)), 0.05,
                   key=f"ve_x0_{lo}_{hi}")
    h = s2.slider("Secant gap h (shrink it to 0 and watch the secant become the tangent)", 0.0, 3.0, 1.0, 0.05,
                  key="ve_h")
    y0, m = g(np.array([x0]))[0], gp(np.array([x0]))[0]
    if not (np.isfinite(y0) and np.isfinite(m)):
        st.warning("The function or its derivative is not defined at this x. Move the point.")
        return

    xs = np.linspace(lo, hi, 700)
    ys, dys = g(xs), gp(xs)
    tangent = y0 + m * (xs - x0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("f(x) with tangent at the chosen point", "f'(x): the slope at every x"))
    fig.add_trace(go.Scatter(x=xs, y=ys, name="f(x)", line=dict(width=3, color=PLOT_COLORS[0])), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=tangent, name="tangent", line=dict(width=2, color=PLOT_COLORS[1])), 1, 1)
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", name="point", marker=dict(size=12, color="#ffffff",
                  line=dict(width=2, color=PLOT_COLORS[1]))), 1, 1)
    secant_slope = None
    if h > 0:
        y1 = g(np.array([x0 + h]))[0]
        if np.isfinite(y1):
            secant_slope = (y1 - y0) / h
            fig.add_trace(go.Scatter(x=xs, y=y0 + secant_slope * (xs - x0), name=f"secant (h={h:g})",
                                     line=dict(width=1.8, dash="dash", color=PLOT_COLORS[2])), 1, 1)
            fig.add_trace(go.Scatter(x=[x0 + h], y=[y1], mode="markers", showlegend=False,
                                     marker=dict(size=9, color=PLOT_COLORS[2])), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=dys, name="f'(x)", line=dict(width=3, color=PLOT_COLORS[3])), 2, 1)
    fig.add_trace(go.Scatter(x=[x0], y=[m], mode="markers", showlegend=False,
                             marker=dict(size=12, color="#ffffff", line=dict(width=2, color=PLOT_COLORS[3]))), 2, 1)
    fig.add_vline(x=x0, line=dict(dash="dot", color="#5a6e94"))
    yr = _y_range(ys)
    if yr:
        fig.update_yaxes(range=yr, row=1, col=1)
    yr2 = _y_range(dys)
    if yr2:
        fig.update_yaxes(range=yr2, row=2, col=1)
    st.plotly_chart(style_fig(fig, 640), use_container_width=True)

    k1, k2, k3 = st.columns(3)
    k1.metric(f"f({x0:g})", _fmt(y0))
    k2.metric(f"Slope f'({x0:g})", _fmt(m))
    k3.metric("Secant slope" if secant_slope is not None else "Secant slope", _fmt(secant_slope) if secant_slope is not None else "–",
              delta=f"{secant_slope - m:+.4g} vs tangent" if secant_slope is not None else None, delta_color="off")
    st.latex(rf"f(x)={sp.latex(f)}\qquad f'(x)={sp.latex(fp)}")
    st.latex(rf"\text{{Tangent: }}\; y = {_fmt(y0)} + ({_fmt(m)})(x - {_fmt(x0)})")

    if abs(m) < 1e-9:
        meaning = ("The tangent is **horizontal**. The slope is zero, so this is a stationary point: a possible "
                   "local maximum, local minimum or flat inflection. Check the sign of f' on either side.")
    elif m > 0:
        meaning = (f"The tangent slopes **upwards** with slope {m:.4g}. Near x = {x0:g} the function is "
                   f"**increasing**: moving 1 unit right raises the curve by about {m:.4g}.")
    else:
        meaning = (f"The tangent slopes **downwards** with slope {m:.4g}. Near x = {x0:g} the function is "
                   f"**decreasing**: moving 1 unit right lowers the curve by about {abs(m):.4g}.")
    st.info(meaning)
    with st.expander("What does the derivative mean geometrically?", expanded=True):
        st.markdown(
            "- The **secant** line joins two points on the curve: its slope is the *average* rate of change over the gap h.\n"
            "- As h shrinks towards 0 the secant rotates into the **tangent**: "
            "$f'(x)=\\lim_{h\\to0}\\frac{f(x+h)-f(x)}{h}$.\n"
            "- So $f'(x_0)$ is the **slope of the tangent** at $x_0$: the *instantaneous* rate of change.\n"
            "- The lower graph records that slope for every x. Where it is positive the top curve rises; "
            "where it crosses zero the top curve has a peak or valley.")


# =====================================================================
# 2. Step-by-step solution engine
# =====================================================================
def _sub_latex(expr, value):
    """LaTeX of expr with x replaced by value, kept unevaluated so the substitution is visible."""
    try:
        return sp.latex(expr.subs(x, sp.UnevaluatedExpr(value)))
    except Exception:
        return sp.latex(expr)


def solve_steps(text: str) -> dict:
    t = normalise(text)
    if t.count("=") != 1:
        raise ValueError("Enter exactly one equals sign, e.g. x^2 - 5x + 6 = 0.")
    left, right = t.split("=")
    lhs, rhs = parse_expr_safe(left), parse_expr_safe(right)
    expr = sp.expand(lhs - rhs)
    steps: list[dict] = []

    def add(title, body="", latex=None, checks=None):
        steps.append({"title": title, "body": body, "latex": latex, "checks": checks or []})

    already = rhs == 0 and sp.simplify(lhs - expr) == 0
    add("Write in standard form" if already else "Move everything to one side",
        "The equation is already in the form $f(x) = 0$." if already else
        "Subtract the right-hand side so the equation reads $f(x) = 0$.",
        latex=rf"{sp.latex(expr)} = 0" if already else
        rf"{sp.latex(lhs)} = {sp.latex(rhs)} \quad\Rightarrow\quad {sp.latex(expr)} = 0")
    if not expr.has(x):
        add("No unknown left", "Every x cancelled, so the statement is " +
            ("always true (an identity): every x works." if expr == 0 else "never true: there is no solution."))
        return {"steps": steps, "solutions": [], "rejected": [], "approx": False, "kind": "degenerate"}

    poly = sp.Poly(expr, x) if expr.is_polynomial(x) else None
    candidates, approx, kind = [], False, "general"
    if poly is not None and poly.degree() == 2:
        kind = "quadratic"
        a, b, c = poly.all_coeffs()
        add("Identify the coefficients", "Compare with $ax^2 + bx + c = 0$.",
            latex=rf"a={sp.latex(a)},\quad b={sp.latex(b)},\quad c={sp.latex(c)}")
        disc = sp.simplify(b**2 - 4 * a * c)
        verdict = ("positive: two distinct real roots." if disc > 0 else
                   "zero: one repeated real root." if disc == 0 else "negative: two complex roots, no real ones.")
        add("Compute the discriminant", f"The discriminant is {verdict}",
            latex=rf"\Delta=b^2-4ac=({sp.latex(b)})^2-4({sp.latex(a)})({sp.latex(c)})={sp.latex(disc)}")
        coeff, factors = sp.factor_list(expr)
        if all(sp.Poly(fc, x).degree() == 1 for fc, _ in factors) and disc >= 0:
            add("Factor the polynomial", "Find two numbers whose product is $ac$ and whose sum is $b$.",
                latex=rf"{sp.latex(expr)} = {sp.latex(sp.factor(expr))}")
            lines = []
            for fc, _ in factors:
                root = sp.solve(fc, x)[0]
                candidates.append(root)
                lines.append(rf"{sp.latex(fc)}=0 \;\Rightarrow\; x={sp.latex(root)}")
            add("Apply the zero-product rule", "If a product is zero, at least one factor must be zero.",
                latex=r"\quad\text{or}\quad ".join(lines))
        else:
            why = "the discriminant is negative" if disc < 0 else "it does not factor over the rationals"
            add("Use the quadratic formula", f"Factoring is not practical because {why}.",
                latex=rf"x=\frac{{-b\pm\sqrt{{\Delta}}}}{{2a}}=\frac{{-({sp.latex(b)})\pm\sqrt{{{sp.latex(disc)}}}}}{{2({sp.latex(a)})}}")
            candidates = list(sp.solve(expr, x))
            add("Simplify", "", latex=r"\quad\text{or}\quad ".join(rf"x={sp.latex(sp.simplify(r))}" for r in candidates))
    elif poly is not None and poly.degree() == 1:
        kind = "linear"
        a, b = poly.all_coeffs()
        add("Isolate x", f"Write the equation as $ax + b = 0$, subtract $b$ from both sides, then divide by $a$.",
            latex=rf"{sp.latex(a)}x = {sp.latex(-b)} \;\Rightarrow\; x = {sp.latex(sp.nsimplify(-b / a))}")
        candidates = [sp.nsimplify(-b / a)]
    elif poly is not None:
        kind = "polynomial"
        fac = sp.factor(expr)
        add("Factor the polynomial", "Factor as far as possible over the rationals.",
            latex=rf"{sp.latex(expr)} = {sp.latex(fac)}")
        coeff, factors = sp.factor_list(expr)
        parts = []
        for fc, _ in factors:
            for r in sp.solve(fc, x):
                candidates.append(r)
                parts.append(rf"{sp.latex(fc)}=0 \;\Rightarrow\; x={sp.latex(sp.simplify(r))}")
        add("Apply the zero-product rule", "Solve each factor separately.", latex=r"\quad\text{or}\quad ".join(parts))
    else:
        try:
            candidates = list(sp.solve(expr, x))
        except Exception:
            candidates = []
        if candidates:
            add("Solve symbolically", "This is not a polynomial, so SymPy's general solver is used. "
                "Every candidate is verified in the last step because operations like squaring or taking logs can "
                "introduce extraneous roots.")
        else:
            roots, _ = numeric_roots(make_fn(expr), -50, 50)
            candidates, approx = [sp.Float(r, 10) for r in roots], True
            add("Solve numerically", "No exact solution was found, so the equation was scanned on [-50, 50] for sign "
                "changes and each crossing was refined by bisection. Treat the values as **approximations**.")

    unique = []
    for r in candidates:
        if not any(sp.simplify(r - u) == 0 for u in unique):
            unique.append(r)
    sols, rej, checks = [], [], []
    for r in unique:
        vl, vr = lhs.subs(x, r), rhs.subs(x, r)
        try:
            diff = sp.simplify(vl - vr)
            ok = bool(diff == 0 or abs(complex(sp.N(diff))) < (1e-6 if approx else 1e-9))
        except Exception:
            ok = False
        txt = (rf"x={sp.latex(sp.simplify(r) if not approx else sp.N(r, 8))}:\quad "
               rf"\text{{LHS}}={_sub_latex(lhs, r)}={sp.latex(sp.simplify(vl) if not approx else sp.N(vl, 6))},\quad "
               rf"\text{{RHS}}={_sub_latex(rhs, r)}={sp.latex(sp.simplify(vr) if not approx else sp.N(vr, 6))}")
        checks.append({"latex": txt, "ok": ok})
        (sols if ok else rej).append(r)
    add("Verify by substitution", "Put each candidate back into the **original** equation.", checks=checks)
    if sols:
        add("Answer", "", latex=r",\quad ".join(rf"x_{{{i}}}={sp.latex(sp.simplify(s) if not approx else sp.N(s, 8))}"
                                              for i, s in enumerate(sols, 1)))
    else:
        add("Answer", "No real or complex solution passed verification." if not unique else
            "Every candidate failed verification, so the equation has no solution.")
    return {"steps": steps, "solutions": sols, "rejected": rej, "approx": approx, "kind": kind}


def render_solution(res: dict):
    for i, s in enumerate(res["steps"], 1):
        st.markdown(f"**Step {i}: {s['title']}**")
        if s["body"]:
            st.markdown(s["body"])
        if s["latex"]:
            st.latex(s["latex"])
        for c in s["checks"]:
            st.latex(c["latex"])
            (st.success if c["ok"] else st.error)("Verified: both sides are equal." if c["ok"]
                                                  else "Rejected: the two sides differ (extraneous root).")
    if res["approx"]:
        st.caption("Numerical answers are approximate; they were checked to a tolerance of 1e-6.")


def render_step_solver():
    st.caption("Enter one equation in x. Use ^ for powers; implicit multiplication such as 2x works.")
    ex = st.columns(4)
    examples = ["x^2 - 5x + 6 = 0", "2x + 3 = 11", "x^2 + 2x + 5 = 0", "x^3 - 6x^2 + 11x - 6 = 0"]
    for i, e in enumerate(examples):
        ex[i].button(e, key=f"ss_ex_{i}", use_container_width=True, on_click=lambda v=e: st.session_state.update(ss_eq=v))
    eq = st.text_input("Equation", "x^2 - 5x + 6 = 0", key="ss_eq")
    if st.button("Solve step by step", type="primary", key="ss_go"):
        try:
            render_solution(solve_steps(eq))
        except Exception as exc:
            st.error(f"Could not solve: {exc}")


# =====================================================================
# Persistence (SQLite)
# =====================================================================
DB_PATH = Path(os.environ.get("MATHPILOT_DB", Path(__file__).with_name("mathpilot.db")))
_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, mode TEXT, qid TEXT, topic TEXT,
  difficulty INTEGER, chosen TEXT, correct INTEGER, seconds REAL);
CREATE TABLE IF NOT EXISTS exams(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, n INTEGER, marks REAL, max_marks REAL,
  correct INTEGER, wrong INTEGER, skipped INTEGER, per_correct REAL, neg REAL,
  limit_s INTEGER, used_s INTEGER, detail TEXT);
"""
_db_ready = False


def _connect():
    global _db_ready
    con = sqlite3.connect(DB_PATH, timeout=10)
    if not _db_ready:
        con.executescript(_SCHEMA)
        con.commit()
        _db_ready = True
    return con


def db_run(sql, params=()):
    with closing(_connect()) as con:
        cur = con.execute(sql, params)
        con.commit()
        return cur.lastrowid


def db_df(sql, params=()) -> pd.DataFrame:
    with closing(_connect()) as con:
        return pd.read_sql_query(sql, con, params=params)


def record_attempt(mode, q, chosen, correct, seconds=0.0):
    db_run("INSERT INTO attempts(ts,mode,qid,topic,difficulty,chosen,correct,seconds) VALUES(?,?,?,?,?,?,?,?)",
           (datetime.now().isoformat(timespec="seconds"), mode, q["id"], q["topic"], q["d"], chosen,
            int(correct), float(seconds)))


# =====================================================================
# Question bank (3 difficulty levels for each of 8 topics)
# =====================================================================
_RAW = [
    ("Trigonometry", 1, "In degree mode, what is sin(30°)?", ["0", "0.5", "1", "√3/2"], 1, "sin 30° = 1/2."),
    ("Trigonometry", 2, "What is cos(60°) + sin(90°)?", ["1", "1.5", "0.5", "2"], 1, "cos 60° = 0.5 and sin 90° = 1, so the sum is 1.5."),
    ("Trigonometry", 3, "If sin θ = 3/5 and θ is acute, what is cos θ?", ["3/5", "4/5", "5/4", "1/5"], 1, "cos²θ = 1 − sin²θ = 1 − 9/25 = 16/25, so cos θ = 4/5."),
    ("Logarithms", 1, "What is log₁₀(1000)?", ["2", "3", "10", "100"], 1, "10³ = 1000, so the logarithm is 3."),
    ("Logarithms", 2, "What is log₂(8) + log₂(4)?", ["3", "5", "6", "12"], 1, "log₂8 = 3 and log₂4 = 2. Equivalently log₂(32) = 5."),
    ("Logarithms", 3, "Solve 2ˣ = 20. What is x, to 2 decimal places?", ["3.32", "4.32", "5.32", "10"], 1, "x = log₂20 = ln 20 / ln 2 ≈ 4.32."),
    ("Calculus", 1, "If f(x) = x³, what is f'(x)?", ["x²", "2x", "3x²", "3x"], 2, "Power rule: d(xⁿ)/dx = n·xⁿ⁻¹."),
    ("Calculus", 2, "If f(x) = x² + 3x, what is f'(2)?", ["7", "10", "4", "8"], 0, "f'(x) = 2x + 3, so f'(2) = 7."),
    ("Calculus", 3, "What is the value of ∫₀² 3x² dx?", ["4", "6", "8", "12"], 2, "The antiderivative is x³, so 2³ − 0³ = 8."),
    ("Matrices", 1, "For A = [[1,2],[3,4]], what is det(A)?", ["2", "-2", "10", "-10"], 1, "det = ad − bc = 4 − 6 = −2."),
    ("Matrices", 2, "For A = [[2,0],[0,3]], what is the top-left entry of A⁻¹?", ["2", "1/2", "1/3", "3"], 1, "The inverse of a diagonal matrix inverts each diagonal entry: 1/2 and 1/3."),
    ("Matrices", 3, "The matrix [[1,2],[2,4]] is…", ["invertible", "singular (det = 0)", "orthogonal", "the identity"], 1, "det = 1·4 − 2·2 = 0, so it has no inverse."),
    ("Probability", 1, "How many ways can 2 students be chosen from 5 if order does not matter?", ["10", "20", "25", "60"], 0, "5C2 = 5!/(2!·3!) = 10."),
    ("Probability", 2, "Two fair dice are rolled. What is P(sum = 7)?", ["1/12", "1/6", "1/36", "7/36"], 1, "6 of the 36 equally likely outcomes sum to 7, so 6/36 = 1/6."),
    ("Probability", 3, "Two cards are drawn without replacement from 52. What is P(both are aces)?", ["1/169", "1/221", "1/13", "4/221"], 1, "(4/52)·(3/51) = 12/2652 = 1/221."),
    ("Equations", 1, "What are the roots of x² − 5x + 6 = 0?", ["1 and 6", "2 and 3", "-2 and -3", "0 and 5"], 1, "(x−2)(x−3) = 0."),
    ("Equations", 2, "x² + 2x + 5 = 0 has…", ["two distinct real roots", "one repeated real root", "two complex roots", "no roots at all"], 2, "Δ = 4 − 20 = −16 < 0, so the roots are complex: −1 ± 2i."),
    ("Equations", 3, "Solve |x − 3| = 5.", ["x = 8 only", "x = -2 only", "x = 8 or x = -2", "x = 2 or x = -8"], 2, "x − 3 = 5 gives 8; x − 3 = −5 gives −2."),
    ("Statistics", 1, "What is the mean of 2, 4, 6, 8?", ["4", "5", "6", "20"], 1, "(2+4+6+8)/4 = 5."),
    ("Statistics", 2, "What is the median of 14, 1, 5, 3, 7?", ["3", "5", "6", "7"], 1, "Sorted: 1, 3, 5, 7, 14. The middle value is 5 (the mean is 6)."),
    ("Statistics", 3, "What is the population standard deviation of 2, 4, 4, 4, 5, 5, 7, 9?", ["1", "2", "4", "5"], 1, "Mean = 5, squared deviations sum to 32, variance = 32/8 = 4, so σ = 2."),
    ("Complex Numbers", 1, "What is i²?", ["1", "i", "-1", "-i"], 2, "i is defined by i² = −1."),
    ("Complex Numbers", 2, "What is |3 + 4i|?", ["7", "5", "25", "√7"], 1, "√(3² + 4²) = 5."),
    ("Complex Numbers", 3, "What is (2 + 3i)(1 − i)?", ["5 + i", "-1 + i", "5 - i", "1 + 5i"], 0, "2 − 2i + 3i − 3i² = 2 + i + 3 = 5 + i."),
]
BANK = [{"id": f"q{i}", "topic": t, "d": d, "q": q, "opts": o, "answer": a, "why": w}
        for i, (t, d, q, o, a, w) in enumerate(_RAW, 1)]
BANK_BY_ID = {q["id"]: q for q in BANK}
TOPICS = list(dict.fromkeys(q["topic"] for q in BANK))
LEVEL_NAMES = {1: "Easy", 2: "Medium", 3: "Hard"}


# =====================================================================
# 3. Adaptive practice engine
# =====================================================================
def _level(results: list[int]) -> int:
    """Replay history: 3 correct in a row = level up, 2 wrong in a row = level down."""
    level, run_c, run_w = 1, 0, 0
    for ok in results:
        if ok:
            run_c, run_w = run_c + 1, 0
            if run_c >= 3 and level < 3:
                level, run_c = level + 1, 0
        else:
            run_w, run_c = run_w + 1, 0
            if run_w >= 2 and level > 1:
                level, run_w = level - 1, 0
    return level


def _mastery(results: list[int]):
    if not results:
        return None
    recent = results[-8:]
    w = np.arange(1, len(recent) + 1, dtype=float)  # newer answers count more
    return float(np.dot(w, recent) / w.sum())


def topic_stats() -> pd.DataFrame:
    df = db_df("SELECT topic, correct FROM attempts ORDER BY id")
    rows = []
    for topic in TOPICS:
        res = df.loc[df.topic == topic, "correct"].astype(int).tolist()
        n, c = len(res), sum(res)
        mastery = _mastery(res)
        repeated = n >= 2 and res[-4:].count(0) >= 2
        if n == 0:
            status = "Not started"
        elif repeated or (n >= 3 and mastery < 0.5):
            status = "Revise"
        elif n >= 3 and mastery >= 0.8:
            status = "Strong"
        else:
            status = "Developing"
        rows.append({"Topic": topic, "Attempts": n, "Correct": c, "Mastery": round(100 * mastery) if mastery is not None else 0,
                     "Level": _level(res), "Status": status})
    return pd.DataFrame(rows)


def pick_question(focus: str | None, avoid_id: str | None):
    stats = topic_stats().set_index("Topic")
    if focus and focus in TOPICS:
        topic = focus
    else:
        weights = [2.0 if stats.loc[t, "Attempts"] == 0 else 1 + 3 * (1 - stats.loc[t, "Mastery"] / 100) for t in TOPICS]
        topic = random.choices(TOPICS, weights=weights)[0]
    level = int(stats.loc[topic, "Level"])
    hist = db_df("SELECT qid, correct FROM attempts WHERE topic=? ORDER BY id", (topic,))
    last = hist.groupby("qid")["correct"].last().to_dict() if not hist.empty else {}
    pool = [q for q in BANK if q["topic"] == topic and q["id"] != avoid_id]
    best = sorted(pool, key=lambda q: (abs(q["d"] - level), last.get(q["id"], -1) == 1, random.random()))
    return best[0]


def _ss_defaults():
    for k, v in {"ap_q": None, "ap_feedback": None, "ap_qid": 0, "ap_start": time.time(), "ap_focus": "Adaptive"}.items():
        st.session_state.setdefault(k, v)
    for k in ("quiz_score", "quiz_attempts", "quiz_streak", "best_streak"):
        st.session_state.setdefault(k, 0)


def _next_question():
    focus = st.session_state.get("ap_focus")
    cur = st.session_state.ap_q
    st.session_state.ap_q = pick_question(None if focus == "Adaptive" else focus, cur["id"] if cur else None)
    st.session_state.ap_feedback = None
    st.session_state.ap_qid += 1
    st.session_state.ap_start = time.time()


def _focus_topic(topic):
    st.session_state.ap_focus = topic
    _next_question()


def _practice_question(qid):
    st.session_state.ap_q = BANK_BY_ID[qid]
    st.session_state.ap_feedback = None
    st.session_state.ap_qid += 1
    st.session_state.ap_start = time.time()


def _submit_answer():
    q = st.session_state.ap_q
    choice = st.session_state.get(f"ap_choice_{st.session_state.ap_qid}")
    if st.session_state.ap_feedback is not None or choice is None:
        return
    before = _level(db_df("SELECT correct FROM attempts WHERE topic=? ORDER BY id", (q["topic"],))["correct"].astype(int).tolist())
    idx = q["opts"].index(choice)
    correct = idx == q["answer"]
    record_attempt("practice", q, choice, correct, time.time() - st.session_state.ap_start)
    after = _level(db_df("SELECT correct FROM attempts WHERE topic=? ORDER BY id", (q["topic"],))["correct"].astype(int).tolist())
    ss = st.session_state
    ss.quiz_attempts += 1
    if correct:
        ss.quiz_score += 1
        ss.quiz_streak += 1
        ss.best_streak = max(ss.best_streak, ss.quiz_streak)
    else:
        ss.quiz_streak = 0
    note = (f"Level up: {q['topic']} questions are now {LEVEL_NAMES[after]}." if after > before else
            f"Stepping {q['topic']} back to {LEVEL_NAMES[after]} to rebuild confidence." if after < before else "")
    ss.ap_feedback = ("correct" if correct else "incorrect", q["why"], note)


def revision_items() -> pd.DataFrame:
    df = db_df("SELECT qid, correct FROM attempts ORDER BY id")
    if df.empty:
        return pd.DataFrame(columns=["qid", "topic", "question", "answer", "why", "times_wrong", "repeated"])
    rows = []
    for qid, grp in df.groupby("qid"):
        if int(grp["correct"].iloc[-1]) == 0 and qid in BANK_BY_ID:
            q = BANK_BY_ID[qid]
            wrong = int((grp["correct"] == 0).sum())
            rows.append({"qid": qid, "topic": q["topic"], "question": q["q"], "answer": q["opts"][q["answer"]],
                         "why": q["why"], "times_wrong": wrong, "repeated": wrong >= 2})
    return pd.DataFrame(rows, columns=["qid", "topic", "question", "answer", "why", "times_wrong", "repeated"])


def render_revision_list(key="rev"):
    rev = revision_items()
    if rev.empty:
        st.success("Nothing to revise. Questions you get wrong appear here until you answer them correctly.")
        return
    rev = rev.sort_values(["repeated", "times_wrong"], ascending=False)
    st.caption(f"{len(rev)} question(s) whose latest attempt was wrong. Repeated mistakes are listed first.")
    for _, r in rev.iterrows():
        tag = "  ·  repeated mistake" if r["repeated"] else ""
        with st.expander(f"{r['topic']}: {r['question']}{tag}"):
            st.markdown(f"**Correct answer:** {r['answer']}")
            st.markdown(r["why"])
            st.caption(f"Missed {r['times_wrong']} time(s).")
            if key == "rev":
                st.button("Practise this question", key=f"{key}_{r['qid']}", on_click=_practice_question, args=(r["qid"],))


def _mastery_map():
    df = db_df("SELECT topic, difficulty, correct FROM attempts")
    z, text = [], []
    for t in TOPICS:
        zr, tr = [], []
        for d in (1, 2, 3):
            sub = df[(df.topic == t) & (df.difficulty == d)] if not df.empty else df
            n = len(sub)
            zr.append(float(sub.correct.mean()) if n else None)
            tr.append(f"{int(sub.correct.sum())}/{n}" if n else "")
        z.append(zr)
        text.append(tr)
    fig = go.Figure(go.Heatmap(z=z, x=[LEVEL_NAMES[d] for d in (1, 2, 3)], y=TOPICS, text=text, texttemplate="%{text}",
                               zmin=0, zmax=1, xgap=3, ygap=3, hoverongaps=False, colorbar=dict(title="Accuracy", tickformat=".0%"),
                               colorscale=[[0, "#f06a6a"], [0.5, "#f0b429"], [1, "#34c38f"]]))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 430)


def render_adaptive_practice():
    _ss_defaults()
    tab_p, tab_m, tab_r = st.tabs(["Practice", "Mastery map", "Revision list"])
    stats = topic_stats()
    with tab_p:
        flagged = stats[stats.Status == "Revise"]
        if not flagged.empty:
            topic = flagged.iloc[0]["Topic"]
            st.warning(f"Revision recommended: **{', '.join(flagged.Topic)}**. Repeated mistakes or low mastery were detected.")
            st.button(f"Focus on {topic}", on_click=_focus_topic, args=(topic,), key="ap_focus_btn")
        st.selectbox("Question selection", ["Adaptive"] + TOPICS, key="ap_focus", on_change=_next_question,
                     help="Adaptive favours topics where your recent accuracy is lowest and adjusts difficulty per topic.")
        if st.session_state.ap_q is None:
            _next_question()
        q = st.session_state.ap_q
        row = stats.set_index("Topic").loc[q["topic"]]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Topic", q["topic"])
        m2.metric("Your level here", LEVEL_NAMES[int(row.Level)])
        m3.metric("Topic mastery", f"{int(row.Mastery)}%" if row.Attempts else "–")
        m4.metric("Current streak", st.session_state.quiz_streak)
        st.markdown(f'<div class="mp-card"><span class="mp-chip">{q["topic"]} · {LEVEL_NAMES[q["d"]]}</span><h3>{q["q"]}</h3></div>',
                    unsafe_allow_html=True)
        answered = st.session_state.ap_feedback is not None
        st.radio("Your answer", q["opts"], index=None, key=f"ap_choice_{st.session_state.ap_qid}", disabled=answered)
        if not answered:
            st.button("Check answer", on_click=_submit_answer, type="primary", use_container_width=True)
        else:
            status, why, note = st.session_state.ap_feedback
            (st.success if status == "correct" else st.error)(
                "Correct. Nice work." if status == "correct" else f"Not quite. The correct answer is {q['opts'][q['answer']]}.")
            st.info(why)
            if note:
                st.toast(note)
                st.caption(note)
            st.button("Next question", on_click=_next_question, type="primary", use_container_width=True)
        st.caption("Difficulty rises after 3 correct answers in a row in a topic and eases after 2 misses in a row.")
    with tab_m:
        st.plotly_chart(_mastery_map(), use_container_width=True)
        st.dataframe(stats, hide_index=True, use_container_width=True, column_config={
            "Mastery": st.column_config.ProgressColumn("Mastery", min_value=0, max_value=100, format="%d%%"),
            "Level": st.column_config.NumberColumn("Level (1-3)")})
        st.caption("Mastery weights recent answers more heavily. Status: Revise = repeated mistakes or mastery below 50%.")
    with tab_r:
        render_revision_list("rev")


# =====================================================================
# 4. Offline math mentor (rule-based tutor)
# =====================================================================
def _n(v):
    v = float(v)
    return str(int(v)) if v == int(v) else f"{v:.4g}"


def _lin(r):
    return f"(x {'-' if r > 0 else '+'} {abs(r)})"


def _signed(a, b, var1="x", var2="y"):
    return f"{a}{var1} {'+' if b >= 0 else '-'} {abs(b)}{var2}"


def _g_quadratic(r):
    while True:
        r1, r2 = r.randint(-6, 6), r.randint(-6, 6)
        if r1 != r2 and 0 not in (r1, r2):
            break
    b, c = -(r1 + r2), r1 * r2
    poly = sp.expand((x - r1) * (x - r2))
    big, small = max(r1, r2), min(r1, r2)
    return dict(problem=f"Solve $ {sp.latex(poly)} = 0 $ and enter the **larger** root.",
                hints=[f"Look for two numbers that multiply to ${c}$ and add to ${b}$.",
                       "Write the polynomial as $(x - p)(x - q)$ using those two numbers.",
                       "Use the zero-product rule: set each bracket equal to zero."],
                steps=[f"Coefficients: $a=1,\\ b={b},\\ c={c}$.",
                       f"Two numbers with product ${c}$ and sum ${b}$ are ${-r1}$ and ${-r2}$.",
                       f"Factor: ${_lin(r1)}{_lin(r2)} = 0$.",
                       f"Zero-product rule: $x = {r1}$ or $x = {r2}$.",
                       f"The larger root is ${big}$."], answer=big)


def _g_derivative(r):
    k, n, m, a = r.randint(1, 5), r.randint(2, 5), r.randint(-5, 5), r.randint(1, 3)
    f = k * x**n + m * x
    ans = k * n * a ** (n - 1) + m
    return dict(problem=f"Let $f(x) = {sp.latex(f)}$. Find $f'({a})$.",
                hints=["Power rule: $\\frac{d}{dx}x^n = n x^{n-1}$.",
                       "Differentiate each term separately. The derivative of $mx$ is $m$.",
                       f"Differentiate first, then substitute $x={a}$."],
                steps=[f"$f'(x) = {sp.latex(sp.diff(f, x))}$.", f"Substitute $x={a}$: $f'({a}) = {ans}$."], answer=ans)


def _g_integral(r):
    k, n, m, b = r.randint(1, 6), r.randint(1, 3), r.randint(0, 4), r.randint(1, 3)
    f = k * x**n + m
    F = sp.integrate(f, x)
    ans = F.subs(x, b) - F.subs(x, 0)
    return dict(problem=f"Evaluate $\\int_0^{{{b}}} \\left({sp.latex(f)}\\right) dx$.",
                hints=["Reverse the power rule: $\\int x^n dx = \\frac{x^{n+1}}{n+1}$.",
                       "Integrate term by term, and a constant $c$ integrates to $cx$.",
                       "Evaluate the antiderivative at the top limit and subtract its value at 0."],
                steps=[f"Antiderivative: $F(x) = {sp.latex(F)}$.", f"$F({b}) - F(0) = {sp.latex(ans)}$."], answer=float(ans))


def _g_log(r):
    b, i, j = r.choice([2, 3, 5]), r.randint(1, 3), r.randint(1, 3)
    A, B = b**i, b**j
    return dict(problem=f"Evaluate $\\log_{{{b}}} {A} + \\log_{{{b}}} {B}$.",
                hints=["Product rule: $\\log_b M + \\log_b N = \\log_b (MN)$.",
                       f"Multiply {A} and {B}. The result is a power of {b}.",
                       "$\\log_b (b^k) = k$."],
                steps=[f"$\\log_{{{b}}}{A} + \\log_{{{b}}}{B} = \\log_{{{b}}}({A}\\cdot{B}) = \\log_{{{b}}}{A * B}$.",
                       f"${A * B} = {b}^{{{i + j}}}$, so the value is ${i + j}$."], answer=i + j)


def _g_det(r):
    a, b, c, d = (r.randint(-5, 5) for _ in range(4))
    return dict(problem=f"Find $\\det\\begin{{bmatrix}}{a}&{b}\\\\{c}&{d}\\end{{bmatrix}}$.",
                hints=["For $\\begin{bmatrix}a&b\\\\c&d\\end{bmatrix}$ the determinant is $ad - bc$.",
                       "Multiply the main diagonal, then the other diagonal.", "Subtract the second product from the first."],
                steps=[f"$ad - bc = ({a})({d}) - ({b})({c})$.", f"$= {a * d} - ({b * c}) = {a * d - b * c}$."], answer=a * d - b * c)


def _g_comb(r):
    n = r.randint(5, 9)
    k = r.randint(2, min(4, n - 1))
    return dict(problem=f"In how many ways can {k} students be chosen from {n}, if order does not matter?",
                hints=["Order does not matter, so use combinations: $\\binom{n}{r} = \\frac{n!}{r!(n-r)!}$.",
                       f"Here $n={n}$ and $r={k}$.", f"Cancel $(n-r)! = {n - k}!$ before multiplying."],
                steps=[f"$\\binom{{{n}}}{{{k}}} = \\frac{{{n}!}}{{{k}!\\,{n - k}!}}$.", f"$= {math.comb(n, k)}$."], answer=math.comb(n, k))


def _g_stats(r):
    while True:
        vals = [r.randint(2, 12) for _ in range(5)]
        if sum(vals) % 5 == 0:
            break
    mean = sum(vals) / 5
    var = sum((v - mean) ** 2 for v in vals) / 5
    return dict(problem=f"Find the **population variance** of {', '.join(map(str, vals))}.",
                hints=["First find the mean.", "Square each deviation from the mean.",
                       "Population variance divides the sum of squared deviations by $n$."],
                steps=[f"Mean $= {_n(mean)}$.", "Squared deviations: " + ", ".join(_n((v - mean) ** 2) for v in vals) + ".",
                       f"Variance $= {_n(sum((v - mean) ** 2 for v in vals))}/5 = {_n(var)}$."], answer=var)


def _g_simul(r):
    while True:
        x0, y0 = r.randint(-4, 4), r.randint(-4, 4)
        a1, b1, a2, b2 = (r.choice([-3, -2, -1, 1, 2, 3, 4]) for _ in range(4))
        if a1 * b2 - a2 * b1 != 0:
            break
    c1, c2 = a1 * x0 + b1 * y0, a2 * x0 + b2 * y0
    D = a1 * b2 - a2 * b1
    return dict(problem=f"Solve $\\;{_signed(a1, b1)} = {c1}\\;$ and $\\;{_signed(a2, b2)} = {c2}$. Enter the value of **x**.",
                hints=["Multiply the equations so that y has the same coefficient size in both.",
                       "Add or subtract the equations to eliminate y.", "Solve for x, then substitute back to check y."],
                steps=[f"Determinant of the coefficients: ${a1}\\cdot{b2} - {a2}\\cdot{b1} = {D}$.",
                       f"$x = \\frac{{{c1}\\cdot{b2} - {c2}\\cdot{b1}}}{{{D}}} = {x0}$.",
                       f"Then $y = {y0}$. Check in the first equation: ${a1 * x0 + b1 * y0} = {c1}$."], answer=x0)


_TRIG = [("sin", 30, sp.Rational(1, 2)), ("cos", 60, sp.Rational(1, 2)), ("tan", 45, 1), ("sin", 90, 1),
         ("cos", 0, 1), ("cos", 90, 0), ("sin", 0, 0), ("cos", 180, -1), ("sin", 270, -1)]


def _g_trig(r):
    (f1, a1, v1), (f2, a2, v2) = r.sample(_TRIG, 2)
    ans = sp.nsimplify(v1) + sp.nsimplify(v2)
    return dict(problem=f"Evaluate $\\{f1} {a1}^\\circ + \\{f2} {a2}^\\circ$ (angles in degrees).",
                hints=["Recall the exact values at 0°, 30°, 45°, 60°, 90° from the unit circle.",
                       "Evaluate each term separately.", "Add the two values."],
                steps=[f"$\\{f1} {a1}^\\circ = {sp.latex(sp.nsimplify(v1))}$.", f"$\\{f2} {a2}^\\circ = {sp.latex(sp.nsimplify(v2))}$.",
                       f"Sum $= {sp.latex(ans)}$."], answer=float(ans))


TUTOR = {
    "quadratic": dict(title="Quadratic equations", gen=_g_quadratic,
                      patterns=[r"quadratic", r"\broots?\b", r"discriminant", r"\bfactor"],
                      concept="A quadratic $ax^2+bx+c=0$ can be solved by **factoring**, by completing the square, or by the formula "
                              "$x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}$. The discriminant $\\Delta=b^2-4ac$ tells you the roots: "
                              "$\\Delta>0$ two real, $\\Delta=0$ one repeated, $\\Delta<0$ complex.\n\n"
                              "**Common mistake:** forgetting to move every term to one side before factoring."),
    "derivative": dict(title="Derivatives", gen=_g_derivative,
                       patterns=[r"deriv", r"differentiat", r"\bslope\b", r"tangent", r"power rule"],
                       concept="The derivative $f'(x)$ is the instantaneous rate of change, or the slope of the tangent. "
                               "Power rule: $\\frac{d}{dx}x^n=nx^{n-1}$. Constants differentiate to 0 and the rule works term by term.\n\n"
                               "**Common mistake:** forgetting to lower the exponent by one."),
    "integral": dict(title="Integration", gen=_g_integral,
                     patterns=[r"integra", r"antideriv", r"area under"],
                     concept="Integration reverses differentiation: $\\int x^n dx=\\frac{x^{n+1}}{n+1}+C$. "
                             "A definite integral $\\int_a^b f\\,dx=F(b)-F(a)$ gives the net signed area.\n\n"
                             "**Common mistake:** forgetting $+C$ on an indefinite integral."),
    "logarithm": dict(title="Logarithms", gen=_g_log, patterns=[r"\blog", r"\bln\b", r"logarithm"],
                      concept="$\\log_b N$ is the power you raise $b$ to in order to get $N$. Rules: "
                              "$\\log(MN)=\\log M+\\log N$, $\\log(M/N)=\\log M-\\log N$, $\\log(M^k)=k\\log M$. The input must be positive.\n\n"
                              "**Common mistake:** writing $\\log(M+N)=\\log M+\\log N$ (it is not)."),
    "determinant": dict(title="Matrices and determinants", gen=_g_det,
                        patterns=[r"determinant", r"\bdet\b", r"matri", r"inverse"],
                        concept="For $A=\\begin{bmatrix}a&b\\\\c&d\\end{bmatrix}$, $\\det A=ad-bc$. A square matrix has an inverse only "
                                "if $\\det A\\ne0$, and then $A^{-1}=\\frac{1}{\\det A}\\begin{bmatrix}d&-b\\\\-c&a\\end{bmatrix}$.\n\n"
                                "**Common mistake:** mixing up the order, it is $ad-bc$, not $bc-ad$."),
    "combinatorics": dict(title="Permutations and combinations", gen=_g_comb,
                          patterns=[r"combin", r"permut", r"\bncr\b", r"\bnpr\b", r"factorial", r"choose"],
                          concept="Use **permutations** $nPr=\\frac{n!}{(n-r)!}$ when order matters and **combinations** "
                                  "$nCr=\\frac{n!}{r!(n-r)!}$ when it does not. $nCr=nC(n-r)$.\n\n"
                                  "**Common mistake:** using nPr for a committee (order does not matter)."),
    "statistics": dict(title="Mean, variance and spread", gen=_g_stats,
                       patterns=[r"\bmean\b", r"\bmedian\b", r"varianc", r"standard deviation", r"statistic", r"average"],
                       concept="Mean $=\\frac{\\sum x}{n}$. Population variance $=\\frac{\\sum(x-\\bar x)^2}{n}$; for a **sample** divide "
                               "by $n-1$. Standard deviation is the square root of the variance.\n\n"
                               "**Common mistake:** using $n$ when the data is a sample (use $n-1$)."),
    "simultaneous": dict(title="Simultaneous linear equations", gen=_g_simul,
                         patterns=[r"simultaneous", r"system of", r"elimination", r"linear equations?"],
                         concept="Two equations in two unknowns can be solved by **elimination** (make one variable's coefficients match, "
                                 "then add or subtract) or by **substitution**. Cramer's rule gives "
                                 "$x=\\frac{c_1b_2-c_2b_1}{a_1b_2-a_2b_1}$.\n\n**Common mistake:** multiplying only one side of an equation."),
    "trigonometry": dict(title="Trigonometry", gen=_g_trig, patterns=[r"\bsin", r"\bcos", r"\btan", r"trig", r"\bangle"],
                         concept="In a right triangle: $\\sin\\theta=\\frac{opp}{hyp}$, $\\cos\\theta=\\frac{adj}{hyp}$, $\\tan\\theta=\\frac{opp}{adj}$. "
                                 "Learn the exact values at 30°, 45°, 60°. Always check the calculator is in **DEG** or **RAD** mode as required.\n\n"
                                 "**Common mistake:** a wrong angle mode on the calculator."),
}

CALC_NOTES = [
    (r"\balpha\b", "ALPHA key", "On many Casio scientific calculators, ALPHA selects the secondary labels printed near keys. "
     "Exact mappings differ by model, so check your manual."),
    (r"\bshift\b", "SHIFT key", "SHIFT reaches the second function printed above a key, usually in yellow. Match the colour of the label to the key."),
    (r"\b(deg|rad|radians?|degrees?)\b", "DEG and RAD", "DEG measures angles in degrees and RAD in radians ($\\pi$ rad = 180°). "
     "Check the mode before using trig functions: it is the most common cause of a wrong trig answer."),
]


def match_topic(question: str):
    q = question.lower()
    for key, t in TUTOR.items():
        if any(re.search(p, q) for p in t["patterns"]):
            return key
    return None


def worked_example(key, seed=None):
    return TUTOR[key]["gen"](random.Random(seed))


def _example_md(ex):
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(ex["steps"], 1))
    return f"{ex['problem']}\n\n{steps}\n\n**Answer:** ${_n(ex['answer'])}$"


def local_tutor(question: str) -> str:
    """Drop-in replacement for the original local_tutor(): structured, rule-based and fully offline."""
    q = question.lower().strip()
    key = match_topic(q)
    if key:
        t = TUTOR[key]
        return (f"### {t['title']}\n\n{t['concept']}\n\n**Worked example**\n\n{_example_md(worked_example(key))}\n\n"
                "*Want to try one yourself, with hints? Open **Offline Math Mentor**.*")
    for pat, title, text in CALC_NOTES:
        if re.search(pat, q):
            return f"### {title}\n\n{text}\n\n**Next step:** Tell me your calculator model for exact key sequences."
    if re.match(r"^(hello|hi|hey)\b", q):
        return "### Welcome\n\nAsk about a calculator key, a concept or a worked problem. I work fully offline."
    return ("### I can help with common topics\n\nThe offline tutor covers: **" + ", ".join(t["title"] for t in TUTOR.values()) +
            "**, plus the ALPHA, SHIFT and DEG/RAD keys. For exact answers use the **Step-by-Step Solver** or **Symbolic Solver**.")


def _mentor_new_problem():
    st.session_state.mentor_seed = random.randint(0, 10**9)
    st.session_state.mentor_hints = 0
    st.session_state.mentor_reveal = False
    st.session_state.mentor_result = None
    st.session_state.mentor_answer = ""


def _mentor_check(key):
    ex = worked_example(key, st.session_state.mentor_seed)
    try:
        val = float(sp.N(parse_expr_safe(st.session_state.mentor_answer)))
        ok = abs(val - float(ex["answer"])) <= max(1e-6, 0.005 * abs(ex["answer"]))
        st.session_state.mentor_result = "correct" if ok else "wrong"
    except Exception:
        st.session_state.mentor_result = "invalid"


def render_offline_mentor():
    for k, v in {"mentor_seed": 1234, "mentor_hints": 0, "mentor_reveal": False, "mentor_result": None, "mentor_answer": ""}.items():
        st.session_state.setdefault(k, v)
    st.caption("Runs entirely on your machine. No API key, no internet, and your questions are never sent anywhere.")
    keys = list(TUTOR)
    ask = st.text_input("Ask in your own words (optional)", placeholder="e.g. how do I find the determinant?", key="mentor_ask")
    guess = match_topic(ask) if ask else None
    if ask and not guess:
        st.info("No matching topic yet. Pick one below.")
    if guess:
        st.session_state["mentor_topic"] = guess
    key = st.selectbox("Topic", keys, format_func=lambda k: TUTOR[k]["title"], key="mentor_topic", on_change=_mentor_new_problem)
    t = TUTOR[key]
    tab_c, tab_w, tab_p = st.tabs(["Concept", "Worked example", "Practise with hints"])
    with tab_c:
        st.markdown(f"#### {t['title']}")
        st.markdown(t["concept"])
    with tab_w:
        st.button("Show a different example", key="mentor_other", on_click=lambda: st.session_state.update(mentor_wseed=random.randint(0, 10**9)))
        st.markdown(_example_md(worked_example(key, st.session_state.get("mentor_wseed", 7))))
    with tab_p:
        ex = worked_example(key, st.session_state.mentor_seed)
        st.markdown(ex["problem"])
        shown = st.session_state.mentor_hints
        for i in range(shown):
            st.info(f"**Hint {i + 1}:** {ex['hints'][i]}")
        c1, c2, c3 = st.columns(3)
        c1.button("Give me a hint", disabled=shown >= len(ex["hints"]) or st.session_state.mentor_reveal, use_container_width=True,
                  key="mentor_hint", on_click=lambda: st.session_state.update(mentor_hints=shown + 1))
        c2.button("Reveal solution", use_container_width=True, key="mentor_rev", on_click=lambda: st.session_state.update(mentor_reveal=True))
        c3.button("New problem", use_container_width=True, key="mentor_new", on_click=_mentor_new_problem)
        st.text_input("Your answer (a number or simple fraction)", key="mentor_answer", on_change=_mentor_check, args=(key,))
        res = st.session_state.mentor_result
        if res == "correct":
            st.success("Correct. Try explaining each step to yourself, then start a new problem.")
        elif res == "wrong":
            st.error("Not quite. Use a hint or check each step again.")
        elif res == "invalid":
            st.warning("Enter a number such as 7, -2.5 or 3/4.")
        if st.session_state.mentor_reveal:
            st.markdown("**Solution**\n\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(ex["steps"], 1)) +
                        f"\n\n**Answer:** ${_n(ex['answer'])}$")


# =====================================================================
# 5. Equation-to-Graph Lab
# =====================================================================
def render_equation_graph():
    c1, c2 = st.columns([2, 1], gap="large")
    src = c1.text_input("Equation or function", "x^2 - 4 = 0", key="eg_src",
                        help="Examples: x^2 - 4 = 0, cos(x) = x, x^3 - 3x, x^2 = 2x + 3. Without = it is treated as y = f(x).")
    lo = c2.number_input("x minimum", value=-6.0, step=1.0, key="eg_lo")
    hi = c2.number_input("x maximum", value=6.0, step=1.0, key="eg_hi")
    if lo >= hi:
        st.error("x minimum must be smaller than x maximum.")
        return
    try:
        t = normalise(src)
        if t.count("=") > 1:
            raise ValueError("Use at most one equals sign.")
        if "=" in t:
            l, r = t.split("=")
            lhs, rhs = parse_expr_safe(l), parse_expr_safe(r)
        else:
            lhs, rhs = parse_expr_safe(t), sp.Integer(0)
        f = sp.expand(lhs - rhs)
        if not f.has(x):
            raise ValueError("Include x in the equation.")
    except Exception as exc:
        st.error(f"Could not read the input: {exc}")
        return

    g, gl, gr = make_fn(f), make_fn(lhs), make_fn(rhs)
    roots, poles = find_roots(f, lo, hi)
    turning = find_turning_points(f, lo, hi)
    xs = np.linspace(lo, hi, 1500)
    yl = gl(xs)
    # break the line at near-vertical jumps (asymptotes) so they are not joined
    jump = np.abs(np.diff(yl, prepend=yl[0])) > 50 * (np.nanstd(yl) + 1e-9)
    yl_plot = np.where(jump, np.nan, yl)
    fig = go.Figure()
    show_rhs = rhs != 0
    fig.add_trace(go.Scatter(x=xs, y=yl_plot, mode="lines", name=f"y = {sp.sstr(lhs).replace('**', '^')}", line=dict(width=3)))
    if show_rhs:
        fig.add_trace(go.Scatter(x=xs, y=gr(xs), mode="lines", name=f"y = {sp.sstr(rhs).replace('**', '^')}", line=dict(width=2.5, dash="dash")))
    if roots:
        rx = np.array([r_["x"] for r_ in roots])
        fig.add_trace(go.Scatter(x=rx, y=gl(rx), mode="markers+text", name="solutions" if show_rhs else "roots",
                                 text=[f"x={r_['x']:.4g}" for r_ in roots], textposition="top center",
                                 marker=dict(size=11, color="#f0b429", line=dict(width=1, color="#fff"))))
    if turning:
        fig.add_trace(go.Scatter(x=[p["x"] for p in turning], y=[p["y"] for p in turning], mode="markers", name="turning points",
                                 marker=dict(size=11, symbol="diamond", color="#34c38f"),
                                 hovertext=[p["type"] for p in turning]))
    y_int = g(np.array([0.0]))[0] if lo <= 0 <= hi else np.nan
    if np.isfinite(y_int):
        yv = gl(np.array([0.0]))[0]
        fig.add_trace(go.Scatter(x=[0], y=[yv], mode="markers", name="y-intercept", marker=dict(size=10, symbol="square", color="#f06a6a")))
    for p in poles[:10]:
        fig.add_vline(x=p, line=dict(dash="dot", color="#f06a6a"))
    yr = _y_range(yl, gr(xs) if show_rhs else None)
    if yr:
        fig.update_yaxes(range=yr)
    st.plotly_chart(style_fig(fig, 540), use_container_width=True)

    st.markdown("#### Key features")
    m1, m2, m3 = st.columns(3)
    m1.metric("Solutions / roots in range", len(roots))
    m2.metric("Turning points", len(turning))
    m3.metric("y-intercept", _fmt(gl(np.array([0.0]))[0]) if np.isfinite(y_int) else "–")
    if show_rhs:
        st.caption("Solutions are the x-values where the two curves meet, i.e. the roots of lhs − rhs.")
    if roots:
        df = pd.DataFrame(roots)
        df["x (approx.)"] = df["x"].map(lambda v: round(v, 8))
        df["exact form"] = df["exact"].map(lambda s: f"${s}$" if s else "–")
        st.dataframe(df[["x (approx.)", "exact form", "method"]].head(50), hide_index=True, use_container_width=True)
        if any("Numerical" in r_["method"] for r_ in roots):
            st.caption("Rows marked Numerical are approximations found by sign-change scanning and bisection, "
                       "used when SymPy cannot give an exact answer.")
    else:
        st.info("No real roots in this x-range. Widen the range, or the equation may have only complex solutions.")
    if turning:
        st.dataframe(pd.DataFrame(turning).rename(columns={"x": "x", "y": "f(x)", "type": "type"}).round(6),
                     hide_index=True, use_container_width=True)
    if poles:
        st.warning("Possible vertical asymptote(s) near x = " + ", ".join(f"{p:.4g}" for p in poles[:6]))
    with st.expander("Check a root with the Step-by-Step Solver"):
        st.code(f"{sp.sstr(lhs).replace('**', '^')} = {sp.sstr(rhs).replace('**', '^')}")
        st.caption("Copy this into the Step-by-Step Solver page for a full worked solution with verification.")


# =====================================================================
# 6. Snap-to-Solve
# =====================================================================
@st.cache_resource(show_spinner="Loading the equation recognition model…")
def _load_latex_ocr():
    from pix2tex.cli import LatexOCR
    return LatexOCR()


def latex_to_plain(latex: str) -> str:
    s = latex.strip().strip("$")
    s = re.sub(r"\\(left|right|,|;|!|quad|qquad)", "", s)
    for _ in range(5):
        s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"^(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/").replace("\\pi", "pi")
    s = re.sub(r"\\(sin|cos|tan|ln|log|exp)", r"\1", s)
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_ocr_text(text: str) -> str:
    s = normalise(text.replace("\n", " "))
    s = re.sub(r"[|]", "1", s)
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", s)  # 3 x 4 -> 3*4
    return re.sub(r"\s+", " ", s).strip()


def ocr_image(data: bytes):
    """Return (plain_text, engine). Tries pix2tex (maths-aware), then Tesseract."""
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    problems = []
    try:
        return latex_to_plain(_load_latex_ocr()(img)), "pix2tex (LaTeX-OCR)"
    except Exception as exc:
        problems.append(f"pix2tex unavailable ({type(exc).__name__})")
    try:
        import pytesseract
        return clean_ocr_text(pytesseract.image_to_string(img, config="--psm 7")), "Tesseract OCR"
    except Exception as exc:
        problems.append(f"Tesseract unavailable ({type(exc).__name__})")
    raise RuntimeError("No OCR engine is installed. Install one of: `pip install pix2tex` (best for maths) or "
                       "`pip install pytesseract` plus the Tesseract binary. " + "; ".join(problems))


def _run_ocr():
    f = st.session_state.get("snap_cam") or st.session_state.get("snap_up")
    if f is None:
        return
    try:
        text, engine = ocr_image(f.getvalue())
        st.session_state.snap_text = text
        st.session_state.snap_msg = ("ok", f"Recognised with {engine}. Please check the text below and fix any mistakes.")
    except Exception as exc:
        st.session_state.snap_msg = ("err", str(exc))


def render_snap_to_solve():
    st.session_state.setdefault("snap_text", "")
    st.session_state.setdefault("snap_msg", None)
    src = st.radio("Image source", ["Upload a photo", "Use camera"], horizontal=True, key="snap_src")
    if src == "Use camera":
        st.camera_input("Take a photo of the equation", key="snap_cam")
        st.session_state.pop("snap_up", None)
    else:
        up = st.file_uploader("Upload an image of a printed or handwritten equation", type=["png", "jpg", "jpeg", "webp"], key="snap_up")
        if up:
            st.image(up, width=360)
    has_img = bool(st.session_state.get("snap_cam") if src == "Use camera" else st.session_state.get("snap_up"))
    st.button("Recognise equation", type="primary", disabled=not has_img, on_click=_run_ocr, key="snap_ocr")
    msg = st.session_state.snap_msg
    if msg:
        (st.success if msg[0] == "ok" else st.warning)(msg[1])
    st.text_input("Equation (edit to correct any recognition errors, or type one directly)", key="snap_text",
                  placeholder="e.g. x^2 - 5x + 6 = 0")
    text = st.session_state.snap_text
    parsed = None
    if text.strip():
        try:
            if "=" in text:
                l, r = normalise(text).split("=")
                parsed = ("eq", parse_expr_safe(l), parse_expr_safe(r))
                st.latex(sp.latex(sp.Eq(parsed[1], parsed[2], evaluate=False)))
            else:
                parsed = ("expr", parse_expr_safe(text))
                st.latex(sp.latex(parsed[1]))
        except Exception as exc:
            st.error(f"I can't read this as an equation yet: {exc}")
    if not parsed:
        return
    if parsed[0] == "eq":
        if st.button("Solve and verify", type="primary", key="snap_solve"):
            try:
                render_solution(solve_steps(text))
            except Exception as exc:
                st.error(f"Could not solve: {exc}")
    else:
        op = st.selectbox("Operation", ["Simplify", "Factor", "Expand", "Differentiate", "Integrate (indefinite)"], key="snap_op")
        if st.button("Run", type="primary", key="snap_run"):
            e = parsed[1]
            fn = {"Simplify": sp.simplify, "Factor": sp.factor, "Expand": sp.expand, "Differentiate": lambda v: sp.diff(v, x),
                  "Integrate (indefinite)": lambda v: sp.integrate(v, x)}[op]
            try:
                out = fn(e)
                st.latex(sp.latex(out) + (r" + C" if op.startswith("Integrate") else ""))
                if op == "Differentiate":
                    st.caption("Check: the slope of the curve matches this at any test point in the Visual Math Explainer.")
            except Exception as exc:
                st.error(f"Could not compute: {exc}")
    st.caption("Handwriting recognition is imperfect: always review the recognised text before solving.")


# =====================================================================
# 7. Exam mode and progress report
# =====================================================================
NEG_OPTIONS = {"None": 0.0, "1/4 of the marks": 0.25, "1/3 of the marks": 1 / 3, "Full marks": 1.0}


def _exam_start(topics, n, minutes, per_correct, neg_label):
    pool = [q for q in BANK if q["topic"] in topics]
    qs = random.sample(pool, min(n, len(pool)))
    st.session_state.exam = {"phase": "active", "id": int(time.time() * 1000), "qids": [q["id"] for q in qs], "start": time.time(),
                             "limit_s": int(minutes * 60), "per_correct": per_correct, "neg": per_correct * NEG_OPTIONS[neg_label],
                             "neg_label": neg_label}


def _exam_finish():
    ex = st.session_state.get("exam")
    if not ex or ex["phase"] != "active":
        return
    used = min(time.time() - ex["start"], ex["limit_s"])
    detail, correct_n, wrong_n, skipped_n, marks = [], 0, 0, 0, 0.0
    for i, qid in enumerate(ex["qids"]):
        q = BANK_BY_ID[qid]
        choice = st.session_state.get(f"exam_ans_{ex['id']}_{i}")
        if choice is None:
            status, m = "skipped", 0.0
            skipped_n += 1
        elif q["opts"].index(choice) == q["answer"]:
            status, m = "correct", ex["per_correct"]
            correct_n += 1
            record_attempt("exam", q, choice, True)
        else:
            status, m = "wrong", -ex["neg"]
            wrong_n += 1
            record_attempt("exam", q, choice, False)
        marks += m
        detail.append({"No": i + 1, "Topic": q["topic"], "Level": LEVEL_NAMES[q["d"]], "Question": q["q"], "Your answer": choice or "(skipped)",
                       "Correct answer": q["opts"][q["answer"]], "Result": status, "Marks": round(m, 3), "Explanation": q["why"]})
    max_marks = ex["per_correct"] * len(ex["qids"])
    ts = datetime.now().isoformat(timespec="seconds")
    db_run("INSERT INTO exams(ts,n,marks,max_marks,correct,wrong,skipped,per_correct,neg,limit_s,used_s,detail) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
           (ts, len(ex["qids"]), marks, max_marks, correct_n, wrong_n, skipped_n, ex["per_correct"], ex["neg"], ex["limit_s"], int(used),
            json.dumps(detail)))
    ex.update(phase="done", result={"ts": ts, "marks": marks, "max": max_marks, "correct": correct_n, "wrong": wrong_n,
                                    "skipped": skipped_n, "used": int(used), "detail": detail})


def _timer_body():
    ex = st.session_state.get("exam")
    if not ex or ex["phase"] != "active":
        return
    left = ex["limit_s"] - (time.time() - ex["start"])
    if left <= 0:
        _exam_finish()
        st.rerun()
    mm, ss = divmod(int(left), 60)
    st.markdown(f"### ⏱ {mm:02d}:{ss:02d} remaining")
    st.progress(max(0.0, min(1.0, left / ex["limit_s"])))


def _render_timer():
    if hasattr(st, "fragment"):
        st.fragment(run_every=1)(_timer_body)()
    else:  # older Streamlit: the timer refreshes whenever you interact
        _timer_body()


def _report_html(res, ex_info) -> str:
    df = pd.DataFrame(res["detail"])
    topic = (df.groupby("Topic")["Result"].value_counts().unstack(fill_value=0)
             .reindex(columns=["correct", "wrong", "skipped"], fill_value=0))
    topic["Marks"] = df.groupby("Topic")["Marks"].sum()
    rows = "".join(f"<tr><td>{escape(str(r['No']))}</td><td>{escape(r['Topic'])}</td><td>{escape(r['Question'])}</td>"
                   f"<td>{escape(r['Your answer'])}</td><td>{escape(r['Correct answer'])}</td><td>{escape(r['Result'])}</td>"
                   f"<td>{r['Marks']}</td></tr>" for r in res["detail"])
    return f"""<!doctype html><meta charset="utf-8"><title>MathPilot exam report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}
td,th{{border:1px solid #ccc;padding:.4rem .6rem;text-align:left;font-size:.9rem}}th{{background:#f0f3f9}}</style>
<h1>MathPilot exam report</h1><p>{escape(res['ts'])}</p>
<h2>Score: {res['marks']:.2f} / {res['max']:.2f}</h2>
<p>Correct {res['correct']} · Wrong {res['wrong']} · Skipped {res['skipped']} · Time used {res['used'] // 60}m {res['used'] % 60}s · Negative marking: {escape(ex_info)}</p>
<h2>Topic-wise results</h2>{topic.reset_index().to_html(index=False)}
<h2>Question review</h2><table><tr><th>#</th><th>Topic</th><th>Question</th><th>Your answer</th><th>Correct</th><th>Result</th><th>Marks</th></tr>{rows}</table>"""


def _render_results():
    ex = st.session_state.exam
    res = ex["result"]
    k = st.columns(5)
    k[0].metric("Marks", f"{res['marks']:.2f} / {res['max']:.0f}")
    attempted = res["correct"] + res["wrong"]
    k[1].metric("Accuracy (attempted)", f"{100 * res['correct'] / attempted:.0f}%" if attempted else "–")
    k[2].metric("Correct", res["correct"])
    k[3].metric("Wrong / skipped", f"{res['wrong']} / {res['skipped']}")
    k[4].metric("Time used", f"{res['used'] // 60}m {res['used'] % 60}s")
    df = pd.DataFrame(res["detail"])
    st.markdown("#### Topic-wise results")
    topic = (df.groupby("Topic")["Result"].value_counts().unstack(fill_value=0)
             .reindex(columns=["correct", "wrong", "skipped"], fill_value=0))
    topic["marks"] = df.groupby("Topic")["Marks"].sum().round(2)
    topic["accuracy %"] = (100 * topic["correct"] / (topic["correct"] + topic["wrong"]).replace(0, np.nan)).round(0)
    st.dataframe(topic.reset_index(), hide_index=True, use_container_width=True)
    fig = go.Figure(go.Bar(x=topic.index, y=topic["marks"], marker_color=PLOT_COLORS[0]))
    fig.update_layout(yaxis_title="Marks")
    st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    st.markdown("#### Revision list from this exam")
    miss = df[df.Result != "correct"]
    if miss.empty:
        st.success("A perfect paper. Nothing to revise.")
    for _, r in miss.iterrows():
        with st.expander(f"{r['Topic']} ({r['Result']}): {r['Question']}"):
            st.markdown(f"**Your answer:** {r['Your answer']}  \n**Correct answer:** {r['Correct answer']}")
            st.info(r["Explanation"])
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download detail (CSV)", df.to_csv(index=False).encode("utf-8"), "mathpilot_exam_detail.csv", "text/csv", use_container_width=True)
    c2.download_button("Download report (HTML)", _report_html(res, ex["neg_label"]).encode("utf-8"), "mathpilot_exam_report.html",
                       "text/html", use_container_width=True)
    c3.button("Start a new exam", type="primary", use_container_width=True, on_click=lambda: st.session_state.update(exam=None))


def _render_active():
    ex = st.session_state.exam
    _render_timer()
    st.caption(f"{ex['per_correct']:g} mark(s) per correct answer · negative marking: {ex['neg_label']} · unanswered questions score 0")
    for i, qid in enumerate(ex["qids"]):
        q = BANK_BY_ID[qid]
        st.markdown(f"**Q{i + 1}.** <span class='mp-chip'>{q['topic']}</span>  {q['q']}", unsafe_allow_html=True)
        st.radio(f"Answer {i + 1}", q["opts"], index=None, key=f"exam_ans_{ex['id']}_{i}", label_visibility="collapsed")
        st.write("")
    answered = sum(st.session_state.get(f"exam_ans_{ex['id']}_{i}") is not None for i in range(len(ex["qids"])))
    st.button(f"Submit exam ({answered}/{len(ex['qids'])} answered)", type="primary", on_click=_exam_finish, use_container_width=True)


def _render_config():
    c1, c2 = st.columns(2)
    topics = c1.multiselect("Topics", TOPICS, default=TOPICS)
    avail = sum(q["topic"] in topics for q in BANK)
    if avail < 3:
        st.warning("Select topics with at least 3 questions in total.")
        return
    n = c1.slider("Number of questions", 3, avail, min(10, avail))
    minutes = c2.number_input("Time limit (minutes)", 1, 180, max(3, int(n * 1.2)))
    per = c2.selectbox("Marks per correct answer", [1, 2, 4], index=2)
    neg = c2.selectbox("Negative marking", list(NEG_OPTIONS), index=1)
    st.caption(f"Maximum score: {per * n} · Penalty per wrong answer: {per * NEG_OPTIONS[neg]:.2f}")
    st.button("Start exam", type="primary", on_click=_exam_start, args=(topics, n, minutes, per, neg))


def _render_history():
    ex = db_df("SELECT id, ts, n, marks, max_marks, correct, wrong, skipped, used_s FROM exams ORDER BY id DESC")
    st.markdown("#### Past exams")
    if ex.empty:
        st.info("No saved exams yet. Finished exams are stored in the local SQLite database.")
    else:
        ex["score %"] = (100 * ex["marks"] / ex["max_marks"]).round(1)
        st.dataframe(ex.rename(columns={"ts": "When", "n": "Questions", "marks": "Marks", "max_marks": "Out of", "used_s": "Seconds used"}),
                     hide_index=True, use_container_width=True)
        st.download_button("Download all exam results (CSV)", ex.to_csv(index=False).encode("utf-8"), "mathpilot_exams.csv", "text/csv")
    att = db_df("SELECT correct FROM attempts ORDER BY id")
    st.markdown("#### Accuracy trend (all practice and exam answers)")
    if len(att) >= 3:
        roll = att["correct"].rolling(min(10, len(att)), min_periods=1).mean() * 100
        fig = go.Figure(go.Scatter(x=list(range(1, len(att) + 1)), y=roll, mode="lines", line=dict(width=3)))
        fig.update_layout(xaxis_title="Answer number", yaxis_title="Rolling accuracy % (last 10)", yaxis_range=[0, 105])
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    else:
        st.caption("Answer a few questions to see your trend.")
    st.caption(f"Data is saved in {DB_PATH.name} and survives browser refreshes and restarts.")


def render_exam_mode():
    st.session_state.setdefault("exam", None)
    ex = st.session_state.exam
    if ex and ex["phase"] == "active":
        _render_active()
        return
    if ex and ex["phase"] == "done":
        _render_results()
        return
    tab_e, tab_h, tab_r = st.tabs(["New exam", "Reports and progress", "Revision list"])
    with tab_e:
        _render_config()
    with tab_h:
        _render_history()
    with tab_r:
        render_revision_list("exam_rev")