import argparse, math, sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import sympy as sp

Number = float

@dataclass
class IterRow:
    it: int
    x: Optional[Number] = None
    a: Optional[Number] = None
    b: Optional[Number] = None
    fx: Optional[Number] = None
    ea: Optional[Number] = None
    note: str = ""

def make_f_and_df(expr_str: str):
    x = sp.symbols('x')
    expr = sp.sympify(expr_str)
    f = sp.lambdify(x, expr, 'numpy')
    df = sp.lambdify(x, sp.diff(expr, x), 'numpy')
    return f, df

def make_func(expr_str: str):
    x = sp.symbols('x')
    return sp.lambdify(x, sp.sympify(expr_str), 'numpy')

def bisection(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Bisection requires f(a) and f(b) to have opposite signs.")
    rows, c_old = [], None
    for it in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        ea = abs(c - c_old) if c_old is not None else None
        rows.append(IterRow(it, c, a, b, fc, ea))
        if abs(fc) < tol or (ea and ea < tol):
            return c, rows
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        c_old = c
    return c, rows

def regula_falsi(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Regula Falsi requires f(a) and f(b) to have opposite signs.")
    rows, c_old = [], None
    for it in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        ea = abs(c - c_old) if c_old is not None else None
        rows.append(IterRow(it, c, a, b, fc, ea))
        if abs(fc) < tol or (ea and ea < tol):
            return c, rows
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        c_old = c
    return c, rows

def secant(f, x0, x1, tol, max_iter):
    rows = []
    for it in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        if f1 == f0:
            raise ZeroDivisionError("Zero denominator.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        ea = abs(x2 - x1)
        rows.append(IterRow(it, x2, fx=f(x2), ea=ea, note=f"x0={x0}, x1={x1}"))
        if abs(f(x2)) < tol or ea < tol:
            return x2, rows
        x0, x1 = x1, x2
    return x2, rows

def newton(f, df, x0, tol, max_iter):
    rows = []
    x = x0
    for it in range(1, max_iter + 1):
        fx, dfx = f(x), df(x)
        if dfx == 0:
            raise ZeroDivisionError("Zero derivative.")
        x_new = x - fx / dfx
        ea = abs(x_new - x)
        rows.append(IterRow(it, x_new, fx=f(x_new), ea=ea))
        if abs(f(x_new)) < tol or ea < tol:
            return x_new, rows
        x = x_new
    return x, rows

def fixed_point(g, x0, tol, max_iter):
    rows, x = [], x0
    for it in range(1, max_iter + 1):
        x_new = g(x)
        ea = abs(x_new - x)
        rows.append(IterRow(it, x_new, ea=ea))
        if ea < tol:
            return x_new, rows
        x = x_new
    return x, rows

def modified_secant(f, x0, delta, tol, max_iter):
    rows, x = [], x0
    for it in range(1, max_iter + 1):
        fx = f(x)
        denom = f(x + delta * x) - fx
        if denom == 0:
            raise ZeroDivisionError("Zero denominator.")
        x_new = x - fx * (delta * x) / denom
        ea = abs(x_new - x)
        rows.append(IterRow(it, x_new, fx=f(x_new), ea=ea))
        if abs(f(x_new)) < tol or ea < tol:
            return x_new, rows
        x = x_new
    return x, rows

def rows_to_table(rows):
    return [{"iter": r.it, "a": r.a, "b": r.b, "x": r.x, "f(x)": r.fx, "error": r.ea, "note": r.note} for r in rows]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--func")
    p.add_argument("--g")
    p.add_argument("--a", type=float)
    p.add_argument("--b", type=float)
    p.add_argument("--x0", type=float)
    p.add_argument("--x1", type=float)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--max_iter", type=int, default=50)
    args = p.parse_args()

    tol, max_iter = args.tol, args.max_iter
    m = args.method.lower()
    if m == "fixed_point":
        if not args.g or args.x0 is None:
            sys.exit("Need g(x) and x0")
        g = make_func(args.g)
        root, rows = fixed_point(g, args.x0, tol, max_iter)
    else:
        if not args.func:
            sys.exit("Need f(x)")
        f, df = make_f_and_df(args.func)
        if m == "bisection":
            root, rows = bisection(f, args.a, args.b, tol, max_iter)
        elif m == "regula_falsi":
            root, rows = regula_falsi(f, args.a, args.b, tol, max_iter)
        elif m == "secant":
            root, rows = secant(f, args.x0, args.x1, tol, max_iter)
        elif m == "newton":
            root, rows = newton(f, df, args.x0, tol, max_iter)
        elif m == "modified_secant":
            root, rows = modified_secant(f, args.x0, args.delta, tol, max_iter)
        else:
            sys.exit("Unknown method")

    table = rows_to_table(rows)
    print("iter,a,b,x,f(x),error")
    for r in table:
        print(f"{r['iter']},{r['a']},{r['b']},{r['x']},{r['f(x)']},{r['error']}")
    print(f"\nRoot ≈ {root}, iterations = {len(rows)}")

if __name__ == "__main__":
    main()