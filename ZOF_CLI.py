#!/usr/bin/env python3
"""
ZOF_CLI.py
Command-line Zero-Of-Function (ZOF) solver implementing:
- Bisection
- Regula Falsi (False Position)
- Secant
- Newton-Raphson
- Fixed Point Iteration
- Modified Secant

Usage: run interactively:
    python ZOF_CLI.py

Or with flags (some example patterns supported interactively too).
"""

import sys
import argparse
import math
import numpy as np
from typing import Callable, Tuple, Optional

# -------------------------
# Safe eval environment
# -------------------------
_SAFE_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
# Allow numpy as np in expressions if user wants
_SAFE_ENV = {"np": np, **_SAFE_MATH}

def make_func(expr: str) -> Callable[[float], float]:
    """
    Create a function f(x) from a string expression, with a safe evaluation environment.
    Example: expr = "x**3 - x - 2"
    """
    expr = expr.strip()
    def f(x):
        local_env = {"x": x}
        return eval(expr, {"__builtins__": {} , **_SAFE_ENV}, local_env)
    return f

def numeric_derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Central difference derivative approximation"""
    return (f(x + h) - f(x - h)) / (2*h)

# -------------------------
# Root-finding methods
# -------------------------
def bisection(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int):
    if f(a) * f(b) > 0:
        raise ValueError("Bisection requires f(a) and f(b) to have opposite signs.")
    iterations = []
    fa = f(a)
    fb = f(b)
    for i in range(1, max_iter+1):
        c = (a + b) / 2.0
        fc = f(c)
        # approximate error as half interval length
        err = abs(b - a) / 2.0
        iterations.append((i, a, b, c, fa, fb, fc, err))
        if abs(fc) < tol or err < tol:
            return c, err, i, iterations
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    # max iterations reached
    c = (a + b) / 2.0
    return c, abs(b - a) / 2.0, max_iter, iterations

def regula_falsi(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int):
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("Regula Falsi requires f(a) and f(b) to have opposite signs.")
    iterations = []
    x_prev = a
    for i in range(1, max_iter+1):
        # x_r = b - fb*(b-a)/(fb-fa)
        xr = (a * fb - b * fa) / (fb - fa)
        fr = f(xr)
        err = abs(xr - x_prev)
        iterations.append((i, a, b, xr, fa, fb, fr, err))
        if abs(fr) < tol or err < tol:
            return xr, err, i, iterations
        if fa * fr < 0:
            b = xr
            fb = fr
        else:
            a = xr
            fa = fr
        x_prev = xr
    return xr, err, max_iter, iterations

def secant(f: Callable[[float], float], x0: float, x1: float, tol: float, max_iter: int):
    iterations = []
    f0 = f(x0)
    f1 = f(x1)
    for i in range(1, max_iter+1):
        if abs(f1 - f0) < 1e-14:
            raise ZeroDivisionError("Division by near-zero in secant method.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        err = abs(x2 - x1)
        iterations.append((i, x0, x1, x2, f0, f1, f2, err))
        if abs(f2) < tol or err < tol:
            return x2, err, i, iterations
        x0, x1 = x1, x2
        f0, f1 = f1, f2
    return x2, err, max_iter, iterations

def newton_raphson(f: Callable[[float], float], x0: float, tol: float, max_iter: int):
    iterations = []
    x = x0
    for i in range(1, max_iter+1):
        fx = f(x)
        dfx = numeric_derivative(f, x)
        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("Derivative near zero in Newton-Raphson.")
        x_new = x - fx / dfx
        err = abs(x_new - x)
        iterations.append((i, x, fx, dfx, x_new, err))
        if abs(fx) < tol or err < tol:
            return x_new, err, i, iterations
        x = x_new
    return x, err, max_iter, iterations

def fixed_point(g: Callable[[float], float], x0: float, tol: float, max_iter: int):
    iterations = []
    x = x0
    for i in range(1, max_iter+1):
        x_new = g(x)
        err = abs(x_new - x)
        iterations.append((i, x, x_new, err))
        if err < tol:
            return x_new, err, i, iterations
        x = x_new
    return x, err, max_iter, iterations

def modified_secant(f: Callable[[float], float], x0: float, delta: float, tol: float, max_iter: int):
    iterations = []
    x = x0
    for i in range(1, max_iter+1):
        fx = f(x)
        denom = f(x + delta * x) - fx
        if abs(denom) < 1e-14:
            raise ZeroDivisionError("Denominator near zero in modified secant.")
        x_new = x - (delta * x * fx) / denom
        err = abs(x_new - x)
        iterations.append((i, x, fx, x_new, err))
        if abs(fx) < tol or err < tol:
            return x_new, err, i, iterations
        x = x_new
    return x, err, max_iter, iterations

# -------------------------
# Printing helpers
# -------------------------
def print_bisection_iters(iters):
    print(f"{'it':>3} | {'a':>12} {'b':>12} {'c':>12} {'f(a)':>12} {'f(b)':>12} {'f(c)':>12} {'err':>12}")
    print("-"*96)
    for i,a,b,c,fa,fb,fc,err in iters:
        print(f"{i:3d} | {a:12.6g} {b:12.6g} {c:12.6g} {fa:12.6g} {fb:12.6g} {fc:12.6g} {err:12.6g}")

def print_regula_iters(iters):
    print(f"{'it':>3} | {'a':>12} {'b':>12} {'xr':>12} {'f(a)':>12} {'f(b)':>12} {'f(xr)':>12} {'err':>12}")
    print("-"*96)
    for i,a,b,xr,fa,fb,fr,err in iters:
        print(f"{i:3d} | {a:12.6g} {b:12.6g} {xr:12.6g} {fa:12.6g} {fb:12.6g} {fr:12.6g} {err:12.6g}")

def print_secant_iters(iters):
    print(f"{'it':>3} | {'x0':>12} {'x1':>12} {'x2':>12} {'f(x0)':>12} {'f(x1)':>12} {'f(x2)':>12} {'err':>12}")
    print("-"*108)
    for i,x0,x1,x2,f0,f1,f2,err in iters:
        print(f"{i:3d} | {x0:12.6g} {x1:12.6g} {x2:12.6g} {f0:12.6g} {f1:12.6g} {f2:12.6g} {err:12.6g}")

def print_newton_iters(iters):
    print(f"{'it':>3} | {'x':>12} {'f(x)':>12} {'f\\'(x)':>12} {'x_new':>12} {'err':>12}")
    print("-"*78)
    for i,x,fx,dfx,xnew,err in iters:
        print(f"{i:3d} | {x:12.6g} {fx:12.6g} {dfx:12.6g} {xnew:12.6g} {err:12.6g}")

def print_fixed_iters(iters):
    print(f"{'it':>3} | {'x':>12} {'g(x)':>12} {'err':>12}")
    print("-"*48)
    for i,x,xnew,err in iters:
        print(f"{i:3d} | {x:12.6g} {xnew:12.6g} {err:12.6g}")

def print_modified_iters(iters):
    print(f"{'it':>3} | {'x':>12} {'f(x)':>12} {'x_new':>12} {'err':>12}")
    print("-"*60)
    for i,x,fx,xnew,err in iters:
        print(f"{i:3d} | {x:12.6g} {fx:12.6g} {xnew:12.6g} {err:12.6g}")

# -------------------------
# Interactive CLI
# -------------------------
def interactive():
    print("=== ZOF Solver CLI ===")
    print("Enter the function f(x) whose root you want to find.")
    expr = input("f(x) = ")
    f = make_func(expr)
    print("\nChoose method:")
    print("1) Bisection")
    print("2) Regula Falsi (False Position)")
    print("3) Secant")
    print("4) Newton-Raphson")
    print("5) Fixed Point Iteration")
    print("6) Modified Secant")
    choice = input("Method [1-6]: ").strip()
    tol = float(input("Tolerance (e.g. 1e-6): ").strip() or 1e-6)
    max_iter = int(input("Max iterations (e.g. 50): ").strip() or 50)

    try:
        if choice == "1":
            a = float(input("a = "))
            b = float(input("b = "))
            root, err, used, iters = bisection(f, a, b, tol, max_iter)
            print_bisection_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        elif choice == "2":
            a = float(input("a = "))
            b = float(input("b = "))
            root, err, used, iters = regula_falsi(f, a, b, tol, max_iter)
            print_regula_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        elif choice == "3":
            x0 = float(input("x0 = "))
            x1 = float(input("x1 = "))
            root, err, used, iters = secant(f, x0, x1, tol, max_iter)
            print_secant_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        elif choice == "4":
            x0 = float(input("Initial guess x0 = "))
            root, err, used, iters = newton_raphson(f, x0, tol, max_iter)
            print_newton_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        elif choice == "5":
            print("Fixed Point requires g(x) such that x = g(x).")
            g_expr = input("g(x) = ")
            g = make_func(g_expr)
            x0 = float(input("Initial guess x0 = "))
            root, err, used, iters = fixed_point(g, x0, tol, max_iter)
            print_fixed_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        elif choice == "6":
            x0 = float(input("Initial guess x0 = "))
            delta = float(input("delta (fraction, e.g. 1e-3) = "))
            root, err, used, iters = modified_secant(f, x0, delta, tol, max_iter)
            print_modified_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        else:
            print("Invalid choice.")
    except Exception as e:
        print("Error during computation:", str(e))

# -------------------------
# CLI argument support (basic)
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="ZOF CLI - root finding methods")
    parser.add_argument("--expr", type=str, help="Function expression f(x) (e.g. 'x**3 - x - 2')")
    parser.add_argument("--method", type=str, choices=['bisection','regula','secant','newton','fixed','modified'], help="Method")
    parser.add_argument("--a", type=float, help="a for interval methods")
    parser.add_argument("--b", type=float, help="b for interval methods")
    parser.add_argument("--x0", type=float, help="initial guess x0")
    parser.add_argument("--x1", type=float, help="initial guess x1 (for secant)")
    parser.add_argument("--g", type=str, help="g(x) for fixed point iteration")
    parser.add_argument("--delta", type=float, help="delta for modified secant")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--maxit", type=int, default=50)
    args = parser.parse_args()

    if args.expr is None:
        interactive()
        return

    f = make_func(args.expr)
    tol = args.tol
    max_iter = args.maxit

    try:
        if args.method == 'bisection':
            if args.a is None or args.b is None:
                raise ValueError("Bisection requires --a and --b.")
            root, err, used, iters = bisection(f, args.a, args.b, tol, max_iter)
            print_bisection_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        elif args.method == 'regula':
            if args.a is None or args.b is None:
                raise ValueError("Regula Falsi requires --a and --b.")
            root, err, used, iters = regula_falsi(f, args.a, args.b, tol, max_iter)
            print_regula_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        elif args.method == 'secant':
            if args.x0 is None or args.x1 is None:
                raise ValueError("Secant requires --x0 and --x1.")
            root, err, used, iters = secant(f, args.x0, args.x1, tol, max_iter)
            print_secant_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        elif args.method == 'newton':
            if args.x0 is None:
                raise ValueError("Newton-Raphson requires --x0.")
            root, err, used, iters = newton_raphson(f, args.x0, tol, max_iter)
            print_newton_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        elif args.method == 'fixed':
            if args.g is None or args.x0 is None:
                raise ValueError("Fixed point requires --g and --x0.")
            g = make_func(args.g)
            root, err, used, iters = fixed_point(g, args.x0, tol, max_iter)
            print_fixed_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        elif args.method == 'modified':
            if args.x0 is None or args.delta is None:
                raise ValueError("Modified secant requires --x0 and --delta.")
            root, err, used, iters = modified_secant(f, args.x0, args.delta, tol, max_iter)
            print_modified_iters(iters)
            print(f"\nResult: root = {root}, error = {err}, iterations = {used}")

        else:
            raise ValueError("No or unknown method provided. Use interactive mode or pass --method.")
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
