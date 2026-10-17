import streamlit as st
import math
import numpy as np
import sympy as sp
import random

st.set_page_config(page_title="Scientific Calculator Learning Assistant", layout="wide", page_icon="🧮")

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}
h1, h2, h3 {
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
}
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}
.block-container {
    padding-top: 1.5rem;
}
.stSelectbox > div, .stTextInput > div > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #fff !important;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102,126,234,0.5);
}
.info-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
}
.formula-box {
    background: rgba(102,126,234,0.18);
    border-left: 4px solid #667eea;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    margin: 0.5rem 0;
}
.tip-box {
    background: rgba(118,75,162,0.25);
    border-left: 4px solid #a78bfa;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0;
}
.example-box {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
}
.chat-user {
    background: linear-gradient(135deg, #667eea22, #764ba222);
    border: 1px solid #667eea55;
    border-radius: 14px 14px 4px 14px;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 80%;
    text-align: right;
}
.chat-bot {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px 14px 14px 4px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    max-width: 90%;
}
.button-key {
    display: inline-block;
    background: linear-gradient(135deg, #374151, #1f2937);
    border: 1px solid #4b5563;
    border-radius: 6px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e5e7eb;
    margin: 2px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
}
.shift-key {
    background: linear-gradient(135deg, #92400e, #78350f);
    border-color: #b45309;
    color: #fcd34d;
}
.alpha-key {
    background: linear-gradient(135deg, #065f46, #064e3b);
    border-color: #059669;
    color: #6ee7b7;
}
.mode-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 700;
    margin: 2px;
}
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    margin: 1rem 0 0.4rem 0;
    border-bottom: 1px solid rgba(167,139,250,0.3);
    padding-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  COMPLETE FUNCTION DATABASE
# ─────────────────────────────────────────────
calculator_guide = {

    # ── TRIGONOMETRY ─────────────────────────
    "sin": {
        "title": "Sine Function", "category": "Trigonometry",
        "purpose": "Finds the sine of an angle — ratio of opposite side to hypotenuse in a right triangle.",
        "formula": "sin(θ)  |  In a right triangle: sin = Opposite / Hypotenuse",
        "steps": ["Press [sin] key.", "Enter the angle value.", "Press [=]."],
        "examples": [{"q": "sin(30°)", "a": "0.5"}, {"q": "sin(90°)", "a": "1"}, {"q": "sin(0°)", "a": "0"}],
        "beginner_tip": "Make sure calculator is in DEG mode (not RAD) for degree inputs. Check display shows 'D'.",
        "try_expression": "math.sin(math.radians(30))"
    },
    "cos": {
        "title": "Cosine Function", "category": "Trigonometry",
        "purpose": "Finds the cosine of an angle — ratio of adjacent side to hypotenuse.",
        "formula": "cos(θ)  |  cos = Adjacent / Hypotenuse",
        "steps": ["Press [cos] key.", "Enter the angle.", "Press [=]."],
        "examples": [{"q": "cos(60°)", "a": "0.5"}, {"q": "cos(0°)", "a": "1"}],
        "beginner_tip": "Cosine of 90° = 0. Cosine is always between -1 and 1.",
        "try_expression": "math.cos(math.radians(60))"
    },
    "tan": {
        "title": "Tangent Function", "category": "Trigonometry",
        "purpose": "Finds the tangent — ratio of opposite to adjacent side.",
        "formula": "tan(θ)  |  tan = Opposite / Adjacent = sin(θ)/cos(θ)",
        "steps": ["Press [tan].", "Enter angle.", "Press [=]."],
        "examples": [{"q": "tan(45°)", "a": "1"}, {"q": "tan(0°)", "a": "0"}],
        "beginner_tip": "tan(90°) is undefined — the calculator shows ERROR.",
        "try_expression": "math.tan(math.radians(45))"
    },
    "arcsin": {
        "title": "Inverse Sine (arcsin / sin⁻¹)", "category": "Inverse Trigonometry",
        "purpose": "Finds the angle whose sine equals a given value. Used to find angles from ratios.",
        "formula": "sin⁻¹(x)  →  returns angle θ such that sin(θ) = x",
        "steps": ["Press [SHIFT] then [sin] (shows sin⁻¹).", "Enter value between -1 and 1.", "Press [=]."],
        "examples": [{"q": "sin⁻¹(0.5)", "a": "30°"}, {"q": "sin⁻¹(1)", "a": "90°"}],
        "beginner_tip": "The input must be between -1 and 1. Result is in DEG or RAD depending on mode.",
        "try_expression": "math.degrees(math.asin(0.5))"
    },
    "arccos": {
        "title": "Inverse Cosine (arccos / cos⁻¹)", "category": "Inverse Trigonometry",
        "purpose": "Finds the angle whose cosine equals a given value.",
        "formula": "cos⁻¹(x)  →  returns angle θ such that cos(θ) = x",
        "steps": ["Press [SHIFT] then [cos].", "Enter value between -1 and 1.", "Press [=]."],
        "examples": [{"q": "cos⁻¹(0.5)", "a": "60°"}, {"q": "cos⁻¹(1)", "a": "0°"}],
        "beginner_tip": "Result range: 0° to 180°.",
        "try_expression": "math.degrees(math.acos(0.5))"
    },
    "arctan": {
        "title": "Inverse Tangent (arctan / tan⁻¹)", "category": "Inverse Trigonometry",
        "purpose": "Finds the angle whose tangent equals a given value.",
        "formula": "tan⁻¹(x)  →  returns angle θ such that tan(θ) = x",
        "steps": ["Press [SHIFT] then [tan].", "Enter any real number.", "Press [=]."],
        "examples": [{"q": "tan⁻¹(1)", "a": "45°"}, {"q": "tan⁻¹(0)", "a": "0°"}],
        "beginner_tip": "Result range: -90° to 90°. Useful in navigation and physics.",
        "try_expression": "math.degrees(math.atan(1))"
    },

    # ── HYPERBOLIC ────────────────────────────
    "sinh": {
        "title": "Hyperbolic Sine (sinh)", "category": "Hyperbolic Functions",
        "purpose": "Calculates the hyperbolic sine — used in engineering, physics, and advanced math.",
        "formula": "sinh(x) = (eˣ - e⁻ˣ) / 2",
        "steps": ["Press [hyp] first (activates hyperbolic mode).", "Then press [sin].", "Enter value.", "Press [=]."],
        "examples": [{"q": "sinh(0)", "a": "0"}, {"q": "sinh(1)", "a": "1.1752"}],
        "beginner_tip": "Press [hyp] to switch sin/cos/tan into their hyperbolic versions.",
        "try_expression": "math.sinh(1)"
    },
    "cosh": {
        "title": "Hyperbolic Cosine (cosh)", "category": "Hyperbolic Functions",
        "purpose": "Calculates the hyperbolic cosine — models hanging cables (catenary curves).",
        "formula": "cosh(x) = (eˣ + e⁻ˣ) / 2",
        "steps": ["Press [hyp].", "Press [cos].", "Enter value.", "Press [=]."],
        "examples": [{"q": "cosh(0)", "a": "1"}, {"q": "cosh(1)", "a": "1.5431"}],
        "beginner_tip": "cosh(0) = 1 always. Values are always ≥ 1.",
        "try_expression": "math.cosh(1)"
    },
    "tanh": {
        "title": "Hyperbolic Tangent (tanh)", "category": "Hyperbolic Functions",
        "purpose": "Calculates the hyperbolic tangent — used in neural networks and signal processing.",
        "formula": "tanh(x) = sinh(x) / cosh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)",
        "steps": ["Press [hyp].", "Press [tan].", "Enter value.", "Press [=]."],
        "examples": [{"q": "tanh(0)", "a": "0"}, {"q": "tanh(1)", "a": "0.7616"}],
        "beginner_tip": "tanh always returns values between -1 and 1.",
        "try_expression": "math.tanh(1)"
    },

    # ── LOGARITHMS ────────────────────────────
    "log": {
        "title": "Logarithm (Base 10)", "category": "Logarithms",
        "purpose": "Finds the power to which 10 must be raised to get x. log(100) = 2 because 10² = 100.",
        "formula": "log₁₀(x)  |  log(x) = y means 10ʸ = x",
        "steps": ["Press [log].", "Enter the number.", "Press [=]."],
        "examples": [{"q": "log(100)", "a": "2"}, {"q": "log(1000)", "a": "3"}, {"q": "log(1)", "a": "0"}],
        "beginner_tip": "log(10) = 1, log(1) = 0. Cannot take log of 0 or negative numbers.",
        "try_expression": "math.log10(100)"
    },
    "ln": {
        "title": "Natural Logarithm (ln)", "category": "Logarithms",
        "purpose": "Finds the power to which e (≈2.718) must be raised to get x.",
        "formula": "ln(x)  |  ln(x) = y means eʸ = x",
        "steps": ["Press [ln].", "Enter the number.", "Press [=]."],
        "examples": [{"q": "ln(e)", "a": "1"}, {"q": "ln(1)", "a": "0"}, {"q": "ln(7.389)", "a": "≈2"}],
        "beginner_tip": "ln and eˣ are inverses of each other.",
        "try_expression": "math.log(2.718281828)"
    },
    "log_base_n": {
        "title": "Logarithm Any Base", "category": "Logarithms",
        "purpose": "Calculate logarithm with any custom base using change of base formula.",
        "formula": "logₙ(x) = log(x) / log(n)  or  ln(x) / ln(n)",
        "steps": ["Enter log(x) ÷ log(base).", "Or use [log] button twice.", "Press [=]."],
        "examples": [{"q": "log₂(8)", "a": "3  (because 2³=8)"}, {"q": "log₃(27)", "a": "3"}],
        "beginner_tip": "No direct button — use change of base: log₂(8) = log(8)/log(2) = 3.",
        "try_expression": "math.log(8, 2)"
    },

    # ── ROOTS & POWERS ────────────────────────
    "square_root": {
        "title": "Square Root (√)", "category": "Roots & Powers",
        "purpose": "Finds the number that when multiplied by itself gives x.",
        "formula": "√x  |  √x = x^(1/2)",
        "steps": ["Press [√] key.", "Enter the value.", "Press [=]."],
        "examples": [{"q": "√25", "a": "5"}, {"q": "√144", "a": "12"}, {"q": "√2", "a": "1.4142"}],
        "beginner_tip": "Negative numbers have no real square root — calculator shows Math ERROR.",
        "try_expression": "math.sqrt(25)"
    },
    "cube_root": {
        "title": "Cube Root (∛)", "category": "Roots & Powers",
        "purpose": "Finds the number that when cubed gives x.",
        "formula": "∛x = x^(1/3)",
        "steps": ["Press [SHIFT] then [√] (or use x^(1/3)).", "Enter value.", "Press [=]."],
        "examples": [{"q": "∛27", "a": "3"}, {"q": "∛8", "a": "2"}, {"q": "∛125", "a": "5"}],
        "beginner_tip": "For any root, use x^(1/n). Cube root of negatives is valid: ∛(-8) = -2.",
        "try_expression": "27 ** (1/3)"
    },
    "nth_root": {
        "title": "Nth Root", "category": "Roots & Powers",
        "purpose": "Finds the nth root of a number.",
        "formula": "ⁿ√x = x^(1/n)",
        "steps": ["Enter n.", "Press [SHIFT][x√y] or [x^(1/n)].", "Enter x.", "Press [=]."],
        "examples": [{"q": "4th root of 81", "a": "3"}, {"q": "5th root of 32", "a": "2"}],
        "beginner_tip": "Use the x^y key: enter x, press ^, enter (1/n), press =.",
        "try_expression": "81 ** (1/4)"
    },
    "power": {
        "title": "Power / Exponent (xʸ)", "category": "Roots & Powers",
        "purpose": "Raises a base number to a given exponent.",
        "formula": "xʸ  |  e.g. 2³ = 2×2×2 = 8",
        "steps": ["Enter the base.", "Press [^] or [xʸ] key.", "Enter the exponent.", "Press [=]."],
        "examples": [{"q": "2^10", "a": "1024"}, {"q": "3^4", "a": "81"}, {"q": "10^6", "a": "1000000"}],
        "beginner_tip": "Any number to the power 0 = 1. Negative exponent means reciprocal: 2⁻¹ = 0.5.",
        "try_expression": "2**10"
    },
    "reciprocal": {
        "title": "Reciprocal (1/x or x⁻¹)", "category": "Roots & Powers",
        "purpose": "Finds the multiplicative inverse of a number.",
        "formula": "1/x  |  x⁻¹",
        "steps": ["Enter the number.", "Press [x⁻¹] or [SHIFT][x²]."],
        "examples": [{"q": "1/4", "a": "0.25"}, {"q": "1/8", "a": "0.125"}],
        "beginner_tip": "1/0 is undefined — shows Math ERROR.",
        "try_expression": "1/4"
    },
    "square": {
        "title": "Square (x²)", "category": "Roots & Powers",
        "purpose": "Multiplies a number by itself.",
        "formula": "x² = x × x",
        "steps": ["Enter the number.", "Press [x²] key.", "Press [=]."],
        "examples": [{"q": "7²", "a": "49"}, {"q": "12²", "a": "144"}],
        "beginner_tip": "x² button saves time instead of pressing x^2.",
        "try_expression": "7**2"
    },

    # ── STATISTICS ────────────────────────────
    "mean": {
        "title": "Mean (Average)", "category": "Statistics",
        "purpose": "Finds the average of a data set.",
        "formula": "x̄ = Σx / n  (sum of all values ÷ count)",
        "steps": ["Press [MODE] → [STAT] (Mode 3).", "Choose 1-VAR.", "Enter data using [=] after each.", "Press [AC].", "Press [SHIFT][1] → Var → x̄ → [=]."],
        "examples": [{"q": "Mean of {2, 4, 6, 8, 10}", "a": "6.0"}],
        "beginner_tip": "In STAT mode, enter each data value then press = to add it to the list.",
        "try_expression": "np.mean([2, 4, 6, 8, 10])"
    },
    "median": {
        "title": "Median", "category": "Statistics",
        "purpose": "Finds the middle value of a sorted dataset.",
        "formula": "Middle value (or average of two middle values for even n)",
        "steps": ["Sort your data manually.", "Find the middle position.", "For even n: average the two middle values."],
        "examples": [{"q": "Median of {1,3,5,7,9}", "a": "5"}, {"q": "Median of {2,4,6,8}", "a": "5"}],
        "beginner_tip": "Most basic calculators don't compute median directly — use the formula.",
        "try_expression": "np.median([1, 3, 5, 7, 9])"
    },
    "std_dev": {
        "title": "Standard Deviation (σ)", "category": "Statistics",
        "purpose": "Measures how spread out values are from the mean.",
        "formula": "σ = √(Σ(xᵢ - x̄)² / n)",
        "steps": ["Press [MODE] → [STAT].", "Enter data values.", "Press [AC].", "Press [SHIFT][1] → Var → σx → [=]."],
        "examples": [{"q": "σ of {2,4,6,8,10}", "a": "2.828"}],
        "beginner_tip": "Use σx for population std dev, Sx for sample std dev.",
        "try_expression": "np.std([2, 4, 6, 8, 10])"
    },
    "permutation": {
        "title": "Permutation (nPr)", "category": "Statistics & Probability",
        "purpose": "Counts arrangements of r items from n items where order matters.",
        "formula": "nPr = n! / (n-r)!",
        "steps": ["Enter n.", "Press [SHIFT][nPr] (or nCr key).", "Enter r.", "Press [=]."],
        "examples": [{"q": "5P3", "a": "60"}, {"q": "4P2", "a": "12"}],
        "beginner_tip": "Order matters in permutations. Password digits = permutation.",
        "try_expression": "math.perm(5, 3)"
    },
    "combination": {
        "title": "Combination (nCr)", "category": "Statistics & Probability",
        "purpose": "Counts selections of r items from n items where order doesn't matter.",
        "formula": "nCr = n! / (r! × (n-r)!)",
        "steps": ["Enter n.", "Press [nCr].", "Enter r.", "Press [=]."],
        "examples": [{"q": "5C2", "a": "10"}, {"q": "6C3", "a": "20"}],
        "beginner_tip": "Order doesn't matter. Lottery ticket = combination.",
        "try_expression": "math.comb(5, 2)"
    },

    # ── CALCULUS ─────────────────────────────
    "integration": {
        "title": "Definite Integration (∫)", "category": "Calculus",
        "purpose": "Calculates the area under a curve between two limits.",
        "formula": "∫ₐᵇ f(x) dx",
        "steps": ["Press [∫dx] key.", "Enter the function f(x).", "Enter lower limit a.", "Enter upper limit b.", "Press [=]."],
        "examples": [{"q": "∫x² from 0 to 3", "a": "9.0"}, {"q": "∫2x from 1 to 4", "a": "15"}],
        "beginner_tip": "This gives the NET area. Use numerical integration for complex functions.",
        "try_expression": "sp.integrate(sp.Symbol('x')**2, (sp.Symbol('x'), 0, 3))"
    },
    "derivative": {
        "title": "Derivative (d/dx)", "category": "Calculus",
        "purpose": "Finds the instantaneous rate of change (slope) of a function at a point.",
        "formula": "d/dx f(x)  or  f'(x)",
        "steps": ["Press [d/dx] key.", "Enter the function.", "Enter the x-value.", "Press [=]."],
        "examples": [{"q": "d/dx(x²) at x=3", "a": "6"}, {"q": "d/dx(x³) at x=2", "a": "12"}],
        "beginner_tip": "d/dx(xⁿ) = n·xⁿ⁻¹. The derivative gives the slope at any point.",
        "try_expression": "sp.diff(sp.Symbol('x')**2, sp.Symbol('x'))"
    },

    # ── FACTORIAL & PROBABILITY ───────────────
    "factorial": {
        "title": "Factorial (n!)", "category": "Factorial & Probability",
        "purpose": "Multiplies all positive integers from 1 up to n.",
        "formula": "n! = n × (n-1) × (n-2) × ... × 1",
        "steps": ["Enter the integer.", "Press [SHIFT][x!] key.", "Press [=]."],
        "examples": [{"q": "5!", "a": "120"}, {"q": "0!", "a": "1"}, {"q": "10!", "a": "3628800"}],
        "beginner_tip": "0! = 1 by definition. Factorial grows very fast — 13! already exceeds 6 billion.",
        "try_expression": "math.factorial(5)"
    },
    "probability": {
        "title": "Probability", "category": "Factorial & Probability",
        "purpose": "Measures likelihood of an event as a fraction between 0 and 1.",
        "formula": "P(E) = Favourable outcomes / Total outcomes",
        "steps": ["Count favourable outcomes.", "Count total outcomes.", "Divide: favourable ÷ total.", "Convert to decimal or percentage if needed."],
        "examples": [{"q": "Rolling a 3 on a die", "a": "1/6 ≈ 0.1667"}, {"q": "Flipping heads", "a": "1/2 = 0.5"}],
        "beginner_tip": "P = 0 means impossible. P = 1 means certain.",
        "try_expression": "1/6"
    },

    # ── MATRICES ─────────────────────────────
    "matrix": {
        "title": "Matrix Operations", "category": "Matrices & Linear Algebra",
        "purpose": "Perform addition, subtraction, multiplication on matrices.",
        "formula": "C = A × B  (matrix multiplication)",
        "steps": ["Press [MODE] → [MATRIX] (Mode 6 on Casio fx-991).", "Press [1] to define MatA.", "Select matrix dimension (e.g. 2×2).", "Enter each element pressing [=] after each.", "Repeat for MatB if needed.", "Press [MATRIX][1] for MatA, select operation (+, ×, etc).", "Press [=]."],
        "examples": [{"q": "[[1,2],[3,4]] × Identity", "a": "[[1,2],[3,4]]"}, {"q": "[[1,0],[0,1]] = Identity matrix", "a": "I"}],
        "beginner_tip": "For multiplication: columns of A must equal rows of B. 2×3 matrix × 3×2 matrix = 2×2 result.",
        "try_expression": "np.array([[1,2],[3,4]]) @ np.eye(2)"
    },
    "determinant": {
        "title": "Matrix Determinant (det)", "category": "Matrices & Linear Algebra",
        "purpose": "A scalar value that encodes properties of a square matrix. det ≠ 0 means matrix is invertible.",
        "formula": "For 2×2: det([[a,b],[c,d]]) = ad - bc",
        "steps": ["Enter MATRIX mode.", "Define your matrix.", "After entering matrix, press [MATRIX].", "Scroll to 'det(' option.", "Select your matrix.", "Press [=]."],
        "examples": [{"q": "det([[1,2],[3,4]])", "a": "-2  (1×4 - 2×3)"}, {"q": "det([[2,0],[0,3]])", "a": "6"}],
        "beginner_tip": "If det = 0, the matrix has no inverse (singular matrix).",
        "try_expression": "np.linalg.det([[1,2],[3,4]])"
    },
    "inverse_matrix": {
        "title": "Matrix Inverse (A⁻¹)", "category": "Matrices & Linear Algebra",
        "purpose": "Finds the inverse matrix such that A × A⁻¹ = Identity.",
        "formula": "A⁻¹  |  A × A⁻¹ = I",
        "steps": ["Enter MATRIX mode.", "Enter your matrix as MatA.", "Press [MATRIX][1] to recall MatA.", "Press [x⁻¹].", "Press [=]."],
        "examples": [{"q": "Inverse of [[2,1],[5,3]]", "a": "[[3,-1],[-5,2]]"}],
        "beginner_tip": "Only square matrices with non-zero determinant have an inverse.",
        "try_expression": "np.linalg.inv([[2,1],[5,3]])"
    },

    # ── EQUATIONS ────────────────────────────
    "linear_equation": {
        "title": "Linear Equation Solver", "category": "Equations",
        "purpose": "Solves equations of form ax + b = 0 or simultaneous linear equations.",
        "formula": "ax + b = 0  →  x = -b/a",
        "steps": ["Press [MODE] → [EQN] (Mode 5 on Casio fx-991).", "Choose: 1 = anX+b=0, 2 = anX²+bnX+c=0.", "For simultaneous: choose number of unknowns.", "Enter coefficients a, b (press [=] after each).", "Press [=] to see solution."],
        "examples": [{"q": "2x + 6 = 0", "a": "x = -3"}, {"q": "3x - 9 = 0", "a": "x = 3"}],
        "beginner_tip": "For ax + b = 0, enter a then b. The calculator solves for x automatically.",
        "try_expression": "sp.solve(2*sp.Symbol('x') + 6, sp.Symbol('x'))"
    },
    "quadratic_equation": {
        "title": "Quadratic Equation Solver", "category": "Equations",
        "purpose": "Solves equations of form ax² + bx + c = 0 giving up to 2 solutions.",
        "formula": "x = (-b ± √(b²-4ac)) / 2a  (Quadratic formula)",
        "steps": ["Press [MODE] → [EQN].", "Choose [2] for ax²+bx+c=0.", "Enter a, b, c (press [=] after each).", "Press [=] to get x₁.", "Press [=] again for x₂."],
        "examples": [{"q": "x²-5x+6=0 (a=1,b=-5,c=6)", "a": "x=3, x=2"}, {"q": "x²-4=0 (a=1,b=0,c=-4)", "a": "x=2, x=-2"}],
        "beginner_tip": "Discriminant b²-4ac: >0 = two real roots, =0 = one root, <0 = complex roots.",
        "try_expression": "sp.solve(sp.Symbol('x')**2 - 5*sp.Symbol('x') + 6, sp.Symbol('x'))"
    },
    "cubic_equation": {
        "title": "Cubic Equation Solver", "category": "Equations",
        "purpose": "Solves equations of form ax³ + bx² + cx + d = 0.",
        "formula": "ax³ + bx² + cx + d = 0",
        "steps": ["Press [MODE] → [EQN].", "Choose [3] for cubic (on supported models).", "Enter a, b, c, d.", "Press [=] to scroll through up to 3 roots."],
        "examples": [{"q": "x³-6x²+11x-6=0", "a": "x=1, x=2, x=3"}],
        "beginner_tip": "Not all models have cubic solver. Casio fx-991EX supports it.",
        "try_expression": "sp.solve(sp.Symbol('x')**3 - 6*sp.Symbol('x')**2 + 11*sp.Symbol('x') - 6, sp.Symbol('x'))"
    },

    # ── FRACTIONS ────────────────────────────
    "fraction": {
        "title": "Fraction Calculations", "category": "Fractions",
        "purpose": "Perform arithmetic with fractions and mixed numbers.",
        "formula": "a/b ± c/d  |  a/b × c/d",
        "steps": ["Enter numerator.", "Press [a b/c] key.", "Enter denominator.", "Perform operation.", "Press [=]."],
        "examples": [{"q": "1/2 + 1/3", "a": "5/6"}, {"q": "3/4 × 2/5", "a": "3/10"}],
        "beginner_tip": "Use [SHIFT][d/c] to switch between mixed number (2½) and improper fraction (5/2).",
        "try_expression": "sp.Rational(1,2) + sp.Rational(1,3)"
    },
    "mixed_number": {
        "title": "Mixed Numbers", "category": "Fractions",
        "purpose": "Work with numbers that have both a whole part and a fraction part.",
        "formula": "a b/c  (e.g. 2½ = 5/2)",
        "steps": ["Enter whole number.", "Press [a b/c].", "Enter numerator.", "Press [a b/c] again.", "Enter denominator.", "Press [=]."],
        "examples": [{"q": "2½ + 1¾", "a": "4¼"}, {"q": "3¼ × 2", "a": "6½"}],
        "beginner_tip": "Press [SHIFT][a b/c] to convert between mixed and improper fraction forms.",
        "try_expression": "sp.Rational(5,2) + sp.Rational(7,4)"
    },

    # ── COMPLEX NUMBERS ──────────────────────
    "complex": {
        "title": "Complex Numbers", "category": "Complex Mathematics",
        "purpose": "Perform calculations involving imaginary numbers (i = √-1).",
        "formula": "z = a + bi  |  i² = -1",
        "steps": ["Press [MODE] → [CMPLX] (Mode 2).", "Use [ENG] key for i.", "Enter complex expression.", "Press [=] to get result in a+bi form.", "Press [SHIFT][2] for polar/rectangular conversion."],
        "examples": [{"q": "(2+3i)+(1+2i)", "a": "3+5i"}, {"q": "(1+i)²", "a": "2i"}, {"q": "|3+4i|", "a": "5"}],
        "beginner_tip": "Press [SHIFT][2] for complex menu: Abs (magnitude), Arg (angle), Conjugate.",
        "try_expression": "complex(2,3) + complex(1,2)"
    },

    # ── NUMBER SYSTEMS ───────────────────────
    "binary": {
        "title": "Binary (Base 2)", "category": "Number Systems (BASE-N)",
        "purpose": "Work with binary numbers used in computers and digital electronics.",
        "formula": "Base 2: uses digits 0 and 1 only",
        "steps": ["Press [MODE] → [BASE-N] (Mode 4).", "Press [BIN] (F2 or equivalent).", "Enter binary number.", "Press [=]."],
        "examples": [{"q": "1010 + 0011 (BIN)", "a": "1101"}, {"q": "1111 in decimal", "a": "15"}, {"q": "8 in binary", "a": "1000"}],
        "beginner_tip": "In BASE-N mode: [BIN]=binary, [OCT]=octal, [DEC]=decimal, [HEX]=hexadecimal.",
        "try_expression": "bin(10)"
    },
    "hexadecimal": {
        "title": "Hexadecimal (Base 16)", "category": "Number Systems (BASE-N)",
        "purpose": "Work with hex numbers used in programming, color codes, memory addresses.",
        "formula": "Base 16: digits 0-9 and A-F (A=10, B=11, ..., F=15)",
        "steps": ["Press [MODE] → [BASE-N].", "Press [HEX].", "Use A-F keys (shown in yellow on ALPHA keys).", "Press [=]."],
        "examples": [{"q": "FF in decimal", "a": "255"}, {"q": "255 in hex", "a": "FF"}, {"q": "1A + 2B", "a": "45"}],
        "beginner_tip": "Colors like #FF5733 are hex values: FF=255 red, 57=87 green, 33=51 blue.",
        "try_expression": "hex(255)"
    },
    "octal": {
        "title": "Octal (Base 8)", "category": "Number Systems (BASE-N)",
        "purpose": "Work with base-8 numbers used in older computing systems.",
        "formula": "Base 8: digits 0 through 7 only",
        "steps": ["Press [MODE] → [BASE-N].", "Press [OCT].", "Enter octal number.", "Press [=]."],
        "examples": [{"q": "17 (octal) in decimal", "a": "15"}, {"q": "8 in octal", "a": "10"}],
        "beginner_tip": "Octal uses digits 0-7 only. Digit 8 or 9 gives SYNTAX ERROR in OCT mode.",
        "try_expression": "oct(15)"
    },

    # ── SCIENTIFIC NOTATION ──────────────────
    "scientific_notation": {
        "title": "Scientific Notation", "category": "Scientific",
        "purpose": "Express very large or very small numbers compactly using powers of 10.",
        "formula": "a × 10ⁿ  |  1 ≤ |a| < 10",
        "steps": ["Enter the coefficient (e.g. 6.02).", "Press [EXP] key.", "Enter the exponent (e.g. 23).", "Press [=]."],
        "examples": [{"q": "6.02 EXP 23", "a": "6.02×10²³ (Avogadro's number)"}, {"q": "1.6 EXP -19", "a": "1.6×10⁻¹⁹ (electron charge)"}],
        "beginner_tip": "[EXP] means ×10^. So 3 EXP 5 = 3×10⁵ = 300000. Never write 3×10 EXP 5.",
        "try_expression": "6.02e23"
    },

    # ── PERCENTAGE ───────────────────────────
    "percentage": {
        "title": "Percentage (%)", "category": "Arithmetic",
        "purpose": "Calculate percentages, percentage change, and percentage of a value.",
        "formula": "x% of y = (x/100) × y  |  Change% = (new-old)/old × 100",
        "steps": ["Enter the value.", "Press [SHIFT][%].", "For 'x% of y': enter y [×] x [SHIFT][%] [=]."],
        "examples": [{"q": "20% of 500", "a": "100"}, {"q": "15% of 80", "a": "12"}, {"q": "500 increased by 10%", "a": "550"}],
        "beginner_tip": "To find what % A is of B: (A ÷ B) × 100.",
        "try_expression": "20/100 * 500"
    },

    # ── MEMORY ───────────────────────────────
    "memory": {
        "title": "Memory Functions (M+, M-, MR, MC)", "category": "Memory",
        "purpose": "Store, recall, add to, and clear a single memory register for reuse.",
        "formula": "M+ adds to memory  |  MR recalls  |  MC clears",
        "steps": ["Calculate a result.", "Press [M+] to add it to memory.", "Perform another calculation.", "Press [MR] to recall stored value.", "Press [MC] or [SHIFT][MC] to clear memory."],
        "examples": [{"q": "Store 25 in memory then add 10", "a": "MR + 10 = 35"}, {"q": "Running total of 3+4+5", "a": "M+, M+, M+, MR = 12"}],
        "beginner_tip": "Memory is useful when the same intermediate result is used multiple times.",
        "try_expression": ""
    },

    # ── VECTOR ───────────────────────────────
    "vector": {
        "title": "Vector Operations", "category": "Vector (MODE 8)",
        "purpose": "Perform vector addition, subtraction, dot product, and cross product.",
        "formula": "A⃗ · B⃗ = |A||B|cos(θ)  |  A⃗ × B⃗ = |A||B|sin(θ)n̂",
        "steps": ["Press [MODE] → [VECTOR] (Mode 8 on Casio fx-991EX).", "Press [1] to define VctA.", "Choose dimension (2D or 3D).", "Enter components.", "Repeat for VctB.", "Press [VECTOR][1] then choose operation (Dot, Cross, etc)."],
        "examples": [{"q": "VctA=(1,2) + VctB=(3,4)", "a": "(4,6)"}, {"q": "Dot product (1,2)·(3,4)", "a": "11"}],
        "beginner_tip": "Dot product result is a scalar (number). Cross product result is a vector.",
        "try_expression": "np.dot([1,2,3],[4,5,6])"
    },

    # ── TABLE ────────────────────────────────
    "table": {
        "title": "Table Mode (f(x) Table)", "category": "Table (MODE 7)",
        "purpose": "Generate a table of values for a function over a range — great for graphing by hand.",
        "formula": "f(x) for x = start, start+step, ..., end",
        "steps": ["Press [MODE] → [TABLE] (Mode 7).", "Enter the function in terms of X (use [ALPHA][X]).", "Press [=].", "Enter Start value → [=].", "Enter End value → [=].", "Enter Step size → [=].", "Calculator displays the table."],
        "examples": [{"q": "f(x)=x² for x=1 to 5, step 1", "a": "1, 4, 9, 16, 25"}, {"q": "f(x)=2x+1 for x=0 to 4", "a": "1, 3, 5, 7, 9"}],
        "beginner_tip": "Use TABLE mode to verify your hand-drawn graphs in class.",
        "try_expression": "[x**2 for x in range(1,6)]"
    },
}

# ─────────────────────────────────────────────
#  CALCULATOR MODES DATABASE
# ─────────────────────────────────────────────
calc_modes = {
    "COMP (Mode 1)": {
        "full_name": "Computation Mode",
        "desc": "Standard calculation mode. Used for all normal arithmetic, trig, log, powers, etc.",
        "use_cases": ["Basic arithmetic", "Trigonometry", "Logarithms", "Powers and roots", "Fractions", "Complex expressions"],
        "how_to": "Press [MODE] → [1]",
        "note": "This is the default mode when you first turn on the calculator."
    },
    "CMPLX (Mode 2)": {
        "full_name": "Complex Number Mode",
        "desc": "Enables calculations with complex numbers (a + bi form).",
        "use_cases": ["Complex addition/subtraction", "Complex multiplication/division", "Finding magnitude |z|", "Finding argument (angle)", "Polar ↔ Rectangular conversion"],
        "how_to": "Press [MODE] → [2]",
        "note": "The [i] symbol appears when this mode is active. Use ENG key for i."
    },
    "STAT (Mode 3)": {
        "full_name": "Statistics Mode",
        "desc": "Data entry and statistical calculation mode.",
        "use_cases": ["Mean (x̄)", "Standard deviation (σx, Sx)", "Sum (Σx, Σx²)", "Regression (linear, quadratic, etc.)", "Single variable and paired variable stats"],
        "how_to": "Press [MODE] → [3], then choose sub-type (1-VAR, A+BX, etc.)",
        "note": "Enter data then press AC. Use [SHIFT][1] to access STAT menu."
    },
    "BASE-N (Mode 4)": {
        "full_name": "Base-N (Number Systems) Mode",
        "desc": "Perform calculations in binary, octal, decimal, or hexadecimal.",
        "use_cases": ["Binary arithmetic (BIN)", "Octal calculations (OCT)", "Hexadecimal (HEX)", "Decimal (DEC)", "Number base conversion"],
        "how_to": "Press [MODE] → [4]. Then select BIN/OCT/HEX/DEC.",
        "note": "Letters A-F are available in HEX mode via function keys."
    },
    "EQN (Mode 5)": {
        "full_name": "Equation Solver Mode",
        "desc": "Solve linear, quadratic, and cubic equations automatically.",
        "use_cases": ["Linear: ax + b = 0", "Simultaneous: 2 or 3 unknowns", "Quadratic: ax²+bx+c=0", "Cubic: ax³+bx²+cx+d=0"],
        "how_to": "Press [MODE] → [5], choose equation type, enter coefficients.",
        "note": "Enter coefficients one by one pressing = after each."
    },
    "MATRIX (Mode 6)": {
        "full_name": "Matrix Mode",
        "desc": "Define and perform operations on matrices up to 3×3.",
        "use_cases": ["Matrix addition/subtraction", "Matrix multiplication", "Determinant", "Inverse", "Transpose"],
        "how_to": "Press [MODE] → [6]. Define MatA, MatB. Then use [MATRIX] menu for operations.",
        "note": "Supports matrices up to 3×3 on most Casio fx-991 models."
    },
    "TABLE (Mode 7)": {
        "full_name": "Function Table Mode",
        "desc": "Generate a table of values for f(x) and g(x) over a range.",
        "use_cases": ["Plotting points by hand", "Studying function behavior", "Comparing two functions"],
        "how_to": "Press [MODE] → [7], enter function using X key, set range and step.",
        "note": "Use [ALPHA][X] to input variable X in the function."
    },
    "VECTOR (Mode 8)": {
        "full_name": "Vector Mode",
        "desc": "Define vectors and compute dot products, cross products, and sums.",
        "use_cases": ["2D and 3D vectors", "Dot product (scalar result)", "Cross product (vector result)", "Vector addition/subtraction", "Vector magnitude"],
        "how_to": "Press [MODE] → [8]. Define VctA, VctB. Use VECTOR menu for operations.",
        "note": "Cross product is only defined for 3D vectors."
    },
}

# ─────────────────────────────────────────────
#  SHIFT+MODE (SETUP) DATABASE
# ─────────────────────────────────────────────
shift_mode_settings = {
    "Input/Output Format": {
        "MthIO (1)": "Math Input/Output — displays fractions, roots, and powers in natural textbook format (e.g. shows √2 as √2 not 1.414).",
        "LineIO (2)": "Line Input/Output — displays everything in a single line like a traditional calculator.",
    },
    "Angle Unit": {
        "Deg (3)": "Degrees — most common. A full circle = 360°. Use for everyday trig problems.",
        "Rad (4)": "Radians — mathematical standard. A full circle = 2π ≈ 6.283. Use in calculus and physics.",
        "Gra (5)": "Gradians — a full circle = 400 grad. Rarely used (surveying and some European engineering).",
    },
    "Display Format": {
        "Fix (6)": "Fixed decimal — sets how many decimal places to show. Fix 2 shows exactly 2 decimal places.",
        "Sci (7)": "Scientific notation — always shows result in scientific notation.",
        "Norm (8)": "Normal display — calculator chooses the best format automatically. Norm 1 and Norm 2 differ in when they switch to scientific notation.",
    },
    "Other Settings": {
        "Stat (9) ON/OFF": "Show or hide the FREQ (frequency) column in STAT mode data entry.",
        "Disp (Shift+9)": "Controls whether fractions are shown as decimals or fractions in certain display modes.",
    }
}

# ─────────────────────────────────────────────
#  BUTTON REFERENCE DATABASE
# ─────────────────────────────────────────────
button_reference = {
    "Basic Arithmetic": {
        "[+] [-] [×] [÷]": "Addition, subtraction, multiplication, division.",
        "[=]": "Executes the calculation and shows result.",
        "[AC]": "All Clear — clears everything (expression and memory display).",
        "[DEL]": "Delete — erases the last entered character.",
        "[(] [)]": "Parentheses — group parts of expressions. Essential for order of operations.",
    },
    "Number & Navigation": {
        "[0-9]": "Number keys.",
        "[.]": "Decimal point.",
        "[(-)]": "Negative sign — makes a number negative. Different from the minus [-] key.",
        "[◀][▶][▲][▼]": "Navigation arrows — move cursor within an expression.",
        "[REPLAY]": "Recalls and re-edits the previous calculation.",
    },
    "Trigonometry": {
        "[sin][cos][tan]": "Sine, cosine, tangent of an angle.",
        "[SHIFT][sin] = sin⁻¹": "Inverse sine (arcsin).",
        "[SHIFT][cos] = cos⁻¹": "Inverse cosine (arccos).",
        "[SHIFT][tan] = tan⁻¹": "Inverse tangent (arctan).",
        "[hyp]": "Activate hyperbolic mode — then press sin/cos/tan for sinh/cosh/tanh.",
    },
    "Powers & Roots": {
        "[x²]": "Squares the current number.",
        "[x³]": "Cubes the current number (SHIFT function on some models).",
        "[^] or [xʸ]": "Raises to any power.",
        "[√]": "Square root.",
        "[SHIFT][√] = ∛": "Cube root.",
        "[x√y]": "General Nth root — enter n, press x√y, enter x.",
        "[x⁻¹]": "Reciprocal (1/x).",
    },
    "Logarithms & Exponentials": {
        "[log]": "Logarithm base 10.",
        "[ln]": "Natural logarithm (base e).",
        "[SHIFT][log] = 10ˣ": "10 to the power x (antilog).",
        "[SHIFT][ln] = eˣ": "e to the power x (exponential).",
    },
    "Fractions": {
        "[a b/c]": "Enter fractions. Press between numerator and denominator.",
        "[SHIFT][a b/c] = d/c": "Toggle between mixed number and improper fraction.",
        "[S⟺D]": "Convert between fraction/surd form and decimal.",
    },
    "Special Functions": {
        "[EXP]": "Scientific notation key. 3 EXP 5 = 3×10⁵.",
        "[%]": "Percentage (SHIFT function on some keys).",
        "[x!]": "Factorial. Enter n then press this key.",
        "[nPr]": "Permutation nPr.",
        "[nCr]": "Combination nCr.",
        "[Ran#]": "Random number — generates random decimal between 0 and 1.",
        "[π]": "Pi constant ≈ 3.14159265358979.",
        "[e]": "Euler's number ≈ 2.71828182845905.",
        "[Ans]": "Recalls the most recent answer for use in next calculation.",
    },
    "Memory Keys": {
        "[M+]": "Adds current result to memory.",
        "[M-]": "Subtracts current result from memory.",
        "[MR] or [RCL][M]": "Recalls the stored memory value.",
        "[MC] or [SHIFT][M+]": "Clears memory (sets to 0).",
        "[STO][A-F]": "Store result into a variable A, B, C, D, E, or F.",
        "[RCL][A-F]": "Recall a stored variable.",
    },
    "Mode & Setup Keys": {
        "[MODE]": "Opens mode selection menu.",
        "[SHIFT][MODE] = SETUP": "Opens calculator setup (angle unit, display format, etc.).",
        "[ON]": "Turns calculator on and resets display.",
        "[SHIFT][AC] = OFF": "Turns calculator off.",
    },
}

# ─────────────────────────────────────────────
#  ALPHA KEY DATABASE
# ─────────────────────────────────────────────
alpha_info = {
    "What is the ALPHA key?": (
        "The [ALPHA] key is a modifier key (like Shift on a keyboard) that gives each key a third function "
        "printed in green/red on or above the key. These are mostly letters (A, B, C...) and special characters."
    ),
    "How to use ALPHA": (
        "Press [ALPHA] once → next key pressed gives its ALPHA function. "
        "Press [ALPHA][ALPHA] → locks ALPHA mode (like Caps Lock) so you can type multiple letters. "
        "Press [ALPHA] again to unlock."
    ),
    "Common ALPHA uses": {
        "[ALPHA][A-F]": "Store/recall variables A through F for reuse in calculations.",
        "[ALPHA][X]": "Input variable X (used in TABLE and CALC modes).",
        "[ALPHA][Y]": "Input variable Y.",
        "[ALPHA][:] = COLON": "Used to separate multiple statements.",
        "[ALPHA][)] = ]": "Closing square bracket for some functions.",
        "[ALPHA][-] = ANS": "Alternative way to access the Ans memory.",
        "Hex digits A-F": "In BASE-N (HEX) mode, ALPHA keys are used to input letters A through F.",
    },
    "Tip": "When you see a green/red label above a key, that's its ALPHA function. Always look for what's printed above keys!"
}

# ─────────────────────────────────────────────
#  AI CHAT RESPONSE BUILDER
# ─────────────────────────────────────────────
def build_ai_response(user_input: str) -> str:
    """Match user question to function and return structured explanation."""
    q = user_input.lower().strip()

    # Mapping of keywords to function keys
    keyword_map = {
        "sin": "sin", "sine": "sin",
        "cos": "cos", "cosine": "cos",
        "tan": "tan", "tangent": "tan",
        "arcsin": "arcsin", "sin-1": "arcsin", "sin⁻¹": "arcsin", "inverse sin": "arcsin",
        "arccos": "arccos", "cos-1": "arccos", "inverse cos": "arccos",
        "arctan": "arctan", "tan-1": "arctan", "inverse tan": "arctan",
        "sinh": "sinh", "hyperbolic sin": "sinh",
        "cosh": "cosh", "hyperbolic cos": "cosh",
        "tanh": "tanh", "hyperbolic tan": "tanh",
        "log": "log", "logarithm": "log", "log10": "log",
        "ln": "ln", "natural log": "ln",
        "log base": "log_base_n", "change of base": "log_base_n",
        "sqrt": "square_root", "square root": "square_root", "√": "square_root",
        "cube root": "cube_root", "∛": "cube_root",
        "nth root": "nth_root",
        "power": "power", "exponent": "power", "^": "power",
        "square": "square", "x²": "square",
        "reciprocal": "reciprocal", "1/x": "reciprocal",
        "factorial": "factorial", "n!": "factorial",
        "probability": "probability",
        "permutation": "permutation", "npr": "permutation",
        "combination": "combination", "ncr": "combination",
        "mean": "mean", "average": "mean",
        "median": "median",
        "standard deviation": "std_dev", "std": "std_dev", "sigma": "std_dev",
        "matrix": "matrix", "matrices": "matrix",
        "determinant": "determinant", "det": "determinant",
        "inverse matrix": "inverse_matrix",
        "linear equation": "linear_equation",
        "quadratic": "quadratic_equation",
        "cubic": "cubic_equation",
        "fraction": "fraction", "fractions": "fraction",
        "mixed number": "mixed_number",
        "integration": "integration", "integral": "integration", "∫": "integration",
        "derivative": "derivative", "differentiation": "derivative", "d/dx": "derivative",
        "complex": "complex", "imaginary": "complex",
        "binary": "binary", "bin": "binary",
        "hex": "hexadecimal", "hexadecimal": "hexadecimal",
        "octal": "octal", "oct": "octal",
        "scientific notation": "scientific_notation", "exp": "scientific_notation",
        "percentage": "percentage", "percent": "percentage",
        "memory": "memory",
        "vector": "vector",
        "table": "table", "table mode": "table",
    }

    matched_key = None
    for kw, fn_key in keyword_map.items():
        if kw in q:
            matched_key = fn_key
            break

    if matched_key and matched_key in calculator_guide:
        d = calculator_guide[matched_key]
        lines = []
        lines.append(f"### 📘 {d['title']}  ·  *{d['category']}*\n")
        lines.append(f"**Purpose:** {d['purpose']}\n")
        lines.append(f"**Formula:** `{d['formula']}`\n")
        lines.append("**Steps on calculator:**")
        for i, s in enumerate(d["steps"], 1):
            lines.append(f"  {i}. {s}")
        lines.append("\n**Examples:**")
        for ex in d["examples"]:
            lines.append(f"  - `{ex['q']}` → **{ex['a']}**")
        lines.append(f"\n💡 **Tip:** {d['beginner_tip']}")
        return "\n".join(lines)

    # Mode questions
    if any(k in q for k in ["mode", "comp", "cmplx", "stat", "base-n", "eqn", "equation mode", "matrix mode", "table mode", "vector mode"]):
        lines = ["### 🔧 Calculator Modes\n"]
        for mode, info in calc_modes.items():
            lines.append(f"**{mode}** — {info['full_name']}")
            lines.append(f"  {info['desc']}")
            lines.append(f"  ➡ *How to access:* {info['how_to']}\n")
        return "\n".join(lines)

    if "alpha" in q:
        lines = ["### 🟢 ALPHA Key Guide\n"]
        for k, v in alpha_info.items():
            if isinstance(v, dict):
                lines.append(f"**{k}:**")
                for btn, desc in v.items():
                    lines.append(f"  `{btn}` — {desc}")
            else:
                lines.append(f"**{k}:** {v}")
        return "\n".join(lines)

    if "shift" in q and ("mode" in q or "setup" in q):
        lines = ["### ⚙️ SHIFT + MODE (Setup) Guide\n"]
        for section, settings in shift_mode_settings.items():
            lines.append(f"**{section}:**")
            for key, desc in settings.items():
                lines.append(f"  `{key}` — {desc}")
        return "\n".join(lines)

    if any(k in q for k in ["button", "key", "keys", "all buttons"]):
        lines = ["### 🎹 All Calculator Buttons\n"]
        for section, buttons in button_reference.items():
            lines.append(f"**{section}:**")
            for key, desc in buttons.items():
                lines.append(f"  `{key}` — {desc}")
        return "\n".join(lines)

    # Fallback
    return (
        "🤔 I couldn't find an exact match. Try asking about:\n\n"
        "- A function: *sin, cos, log, factorial, matrix, integration...*\n"
        "- A mode: *COMP, CMPLX, STAT, BASE-N, EQN, MATRIX, TABLE, VECTOR*\n"
        "- A key: *ALPHA, SHIFT, MODE, EXP, nCr, nPr...*\n"
        "- Setup: *shift mode, DEG RAD GRA, FIX SCI NORM*\n\n"
        "You can also browse all functions in the **Function Explorer** tab."
    )


# ─────────────────────────────────────────────
#  LIVE CALCULATOR
# ─────────────────────────────────────────────
def run_calculation(expr: str) -> str:
    try:
        if not expr.strip():
            return ""
        # Allow safe functions
        safe = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "sqrt": math.sqrt, "log": math.log10, "log10": math.log10,
            "log2": math.log2, "ln": math.log, "exp": math.exp,
            "abs": abs, "pi": math.pi, "e": math.e,
            "factorial": math.factorial, "degrees": math.degrees,
            "radians": math.radians, "floor": math.floor, "ceil": math.ceil,
            "round": round, "pow": pow,
        }
        result = eval(expr, {"__builtins__": {}}, safe)
        return str(result)
    except Exception as ex:
        return f"Error: {ex}"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🧮 Navigation")
page = st.sidebar.radio("", ["💬 AI Chat Assistant", "📖 Function Explorer", "🔧 Calculator Modes", "⚙️ SHIFT+MODE Setup", "🟢 ALPHA Key", "🎹 Button Reference", "🧪 Live Calculator"])

st.sidebar.markdown("---")
st.sidebar.markdown("**🔍 Quick Search**")
search = st.sidebar.text_input("Search any function...", placeholder="e.g. sin, matrix, log")
if search:
    found = [k for k in calculator_guide if search.lower() in k.lower() or search.lower() in calculator_guide[k]["title"].lower() or search.lower() in calculator_guide[k]["category"].lower()]
    if found:
        st.sidebar.success(f"Found {len(found)} result(s):")
        for item in found:
            st.sidebar.write(f"• **{calculator_guide[item]['title']}**  `({item})`")
    else:
        st.sidebar.error("No matching function found.")

# ─────────────────────────────────────────────
#  PAGE: AI CHAT ASSISTANT
# ─────────────────────────────────────────────
if page == "💬 AI Chat Assistant":
    st.markdown("## 💬 AI Learning Assistant")
    st.markdown('<div class="info-card">Ask me anything about your scientific calculator! Try: <em>"How do I use sin?"</em> · <em>"Explain matrix mode"</em> · <em>"What does ALPHA key do?"</em> · <em>"How to solve quadratic equation?"</em></div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑‍🎓 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("", placeholder="Type your question here... e.g. How do I find the determinant of a matrix?", label_visibility="collapsed")
        with col2:
            submitted = st.form_submit_button("Ask ➤")

    if submitted and user_input:
        response = build_ai_response(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "bot", "content": response})
        st.rerun()

    st.markdown("---")
    st.markdown("**💡 Try asking:**")
    quick_questions = [
        "How does sin work?", "Explain matrix multiplication", "What is ALPHA key?",
        "How to solve quadratic equation?", "Explain integration", "What is BASE-N mode?",
        "How does SHIFT MODE work?", "Explain complex numbers", "How to calculate factorial?"
    ]
    cols = st.columns(3)
    for i, qq in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(qq, key=f"quick_{i}"):
                response = build_ai_response(qq)
                st.session_state.chat_history.append({"role": "user", "content": qq})
                st.session_state.chat_history.append({"role": "bot", "content": response})
                st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────
#  PAGE: FUNCTION EXPLORER
# ─────────────────────────────────────────────
elif page == "📖 Function Explorer":
    st.markdown("## 📖 Function Explorer")

    categories = sorted(set(v["category"] for v in calculator_guide.values()))
    selected_cat = st.selectbox("Filter by Category", ["All Categories"] + categories)

    filtered = {k: v for k, v in calculator_guide.items() if selected_cat == "All Categories" or v["category"] == selected_cat}
    selected_fn = st.selectbox("Choose Function", list(filtered.keys()), format_func=lambda k: f"{filtered[k]['title']} ({filtered[k]['category']})")

    if selected_fn:
        d = filtered[selected_fn]
        st.markdown(f"## {d['title']}")
        st.markdown(f'<span class="mode-badge">{d["category"]}</span>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">🎯 Purpose</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card">{d["purpose"]}</div>', unsafe_allow_html=True)

            st.markdown('<p class="section-header">🧾 Formula</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="formula-box">{d["formula"]}</div>', unsafe_allow_html=True)

            st.markdown('<p class="section-header">💡 Beginner Tip</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="tip-box">💡 {d["beginner_tip"]}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<p class="section-header">🪜 Steps on Calculator</p>', unsafe_allow_html=True)
            for i, step in enumerate(d["steps"], 1):
                st.markdown(f'<div class="info-card">Step {i}: {step}</div>', unsafe_allow_html=True)

            st.markdown('<p class="section-header">✅ Examples</p>', unsafe_allow_html=True)
            for ex in d["examples"]:
                st.markdown(f'<div class="example-box">❓ <b>Q:</b> {ex["q"]}<br>✅ <b>A:</b> {ex["a"]}</div>', unsafe_allow_html=True)

        if d["try_expression"]:
            st.markdown('<p class="section-header">🧪 Live Result</p>', unsafe_allow_html=True)
            result = run_calculation(d["try_expression"])
            st.code(f"Expression: {d['try_expression']}\nResult:     {result}")

# ─────────────────────────────────────────────
#  PAGE: CALCULATOR MODES
# ─────────────────────────────────────────────
elif page == "🔧 Calculator Modes":
    st.markdown("## 🔧 Calculator Modes (MODE key)")
    st.markdown('<div class="info-card">Press <b>[MODE]</b> on your Casio fx-991 to access these modes. Each mode changes what the calculator can do.</div>', unsafe_allow_html=True)

    for mode_name, info in calc_modes.items():
        with st.expander(f"📌 {mode_name} — {info['full_name']}"):
            st.markdown(f'<div class="info-card">{info["desc"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="formula-box">How to access: {info["how_to"]}</div>', unsafe_allow_html=True)
            st.markdown("**Use cases:**")
            for uc in info["use_cases"]:
                st.write(f"  ✅ {uc}")
            st.markdown(f'<div class="tip-box">📝 Note: {info["note"]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: SHIFT+MODE SETUP
# ─────────────────────────────────────────────
elif page == "⚙️ SHIFT+MODE Setup":
    st.markdown("## ⚙️ SHIFT + MODE (Setup)")
    st.markdown('<div class="info-card">Press <b>[SHIFT][MODE]</b> to open the Setup menu. This controls how the calculator displays numbers and interprets angles.</div>', unsafe_allow_html=True)

    for section, settings in shift_mode_settings.items():
        st.markdown(f'<p class="section-header">{section}</p>', unsafe_allow_html=True)
        for key, desc in settings.items():
            st.markdown(f'<div class="info-card"><span class="button-key shift-key">{key}</span> &nbsp; {desc}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌡️ DEG vs RAD vs GRA — Quick Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="example-box"><b>DEG (Degrees)</b><br>Full circle = 360°<br>sin(30°) = 0.5<br>✅ Use for: school trig, everyday</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="example-box"><b>RAD (Radians)</b><br>Full circle = 2π ≈ 6.283<br>sin(π/6) = 0.5<br>✅ Use for: calculus, physics</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="example-box"><b>GRA (Gradians)</b><br>Full circle = 400 grad<br>sin(50g) = 0.5<br>✅ Use for: surveying (rare)</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: ALPHA KEY
# ─────────────────────────────────────────────
elif page == "🟢 ALPHA Key":
    st.markdown("## 🟢 The ALPHA Key")
    for section, content in alpha_info.items():
        if isinstance(content, dict):
            st.markdown(f'<p class="section-header">{section}</p>', unsafe_allow_html=True)
            for k, v in content.items():
                st.markdown(f'<div class="info-card"><span class="button-key alpha-key">{k}</span> &nbsp; {v}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="section-header">{section}</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card">{content}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔠 ALPHA Variables (A through F + X, Y, M)")
    cols = st.columns(4)
    vars_info = [
        ("A", "General purpose variable"),
        ("B", "General purpose variable"),
        ("C", "General purpose variable"),
        ("D", "General purpose variable"),
        ("E", "General purpose variable"),
        ("F", "General purpose variable"),
        ("X", "Variable X — used in TABLE and CALC modes"),
        ("Y", "Variable Y — used in some regression modes"),
        ("M", "Memory variable (also accessible via MR)"),
    ]
    for i, (var, desc) in enumerate(vars_info):
        with cols[i % 4]:
            st.markdown(f'<div class="example-box"><b>[ALPHA][{var}]</b><br>{desc}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: BUTTON REFERENCE
# ─────────────────────────────────────────────
elif page == "🎹 Button Reference":
    st.markdown("## 🎹 Complete Button Reference")
    st.markdown('<div class="info-card">Every button on your Casio fx-991 scientific calculator, explained clearly.</div>', unsafe_allow_html=True)

    for section, buttons in button_reference.items():
        st.markdown(f'<p class="section-header">{section}</p>', unsafe_allow_html=True)
        for key, desc in buttons.items():
            key_html = " ".join(f'<span class="button-key">{k.strip()}</span>' for k in key.split("]["))
            st.markdown(f'<div class="info-card">{key_html} &nbsp; {desc}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: LIVE CALCULATOR
# ─────────────────────────────────────────────
elif page == "🧪 Live Calculator":
    st.markdown("## 🧪 Live Calculator")
    st.markdown('<div class="info-card">Type a mathematical expression below and get the result instantly. Use Python math syntax: <code>sin(radians(30))</code>, <code>sqrt(25)</code>, <code>log10(100)</code>, <code>factorial(5)</code>, <code>exp(1)</code></div>', unsafe_allow_html=True)

    expr = st.text_input("Enter expression:", placeholder="e.g. sin(radians(45)), sqrt(144), factorial(6), log10(1000)")

    col1, col2 = st.columns([1, 3])
    with col1:
        angle_mode = st.selectbox("Angle Mode", ["Degrees", "Radians"])

    if expr:
        if angle_mode == "Degrees" and any(f in expr for f in ["sin(", "cos(", "tan(", "asin(", "acos(", "atan("]):
            if "radians" not in expr and "degrees" not in expr:
                st.info("💡 Tip: Your expression uses trig functions. If your angle is in degrees, wrap it: `sin(radians(30))`")

        result = run_calculation(expr)
        if result.startswith("Error"):
            st.error(f"❌ {result}")
        else:
            st.success(f"✅ Result: **{result}**")
            st.code(f">>> {expr}\n{result}")

    st.markdown("---")
    st.markdown("### 📋 Quick Examples — Click to Try")
    examples = [
        ("sin(radians(30))", "sin(30°)"),
        ("cos(radians(60))", "cos(60°)"),
        ("tan(radians(45))", "tan(45°)"),
        ("log10(1000)", "log(1000)"),
        ("ln(exp(1))", "ln(e)"),
        ("sqrt(144)", "√144"),
        ("factorial(7)", "7!"),
        ("pow(2, 10)", "2¹⁰"),
        ("pi * pow(5, 2)", "πr² r=5"),
        ("round(exp(1), 5)", "e value"),
    ]
    cols = st.columns(5)
    for i, (expr_val, label) in enumerate(examples):
        with cols[i % 5]:
            if st.button(label, key=f"ex_{i}"):
                r = run_calculation(expr_val)
                st.info(f"`{expr_val}` = **{r}**")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#a78bfa; font-size:0.9rem; padding:0.5rem'>
    🧮 <b>Scientific Calculator Learning Assistant</b> · Built with Python + Streamlit<br>
    Covers: Casio fx-991EX / fx-991ES Plus · All modes, functions, buttons and setup keys
</div>
""", unsafe_allow_html=True)



