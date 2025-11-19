import sympy as sp

# Convert string to callable function
def make_function(func_str):
    x = sp.symbols("x")
    expr = sp.sympify(func_str)
    return sp.lambdify(x, expr, "math"), expr, x


# -----------------------------------------
# 1. BISECTION METHOD
# -----------------------------------------
def bisection(func_str, a, b, tol=1e-6, max_iter=30):
    f, _, _ = make_function(func_str)
    rows = []

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fa = f(a)
        fc = f(c)
        err = abs(b - a)

        rows.append([i, a, b, c, fa, fc, err])

        if fc == 0 or err < tol:
            break

        if fa * fc < 0:
            b = c
        else:
            a = c

    return rows


# -----------------------------------------
# 2. REGULA FALSI (FALSE POSITION)
# -----------------------------------------
def regula_falsi(func_str, a, b, tol=1e-6, max_iter=30):
    f, _, _ = make_function(func_str)
    rows = []

    for i in range(1, max_iter + 1):
        fa = f(a)
        fb = f(b)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        err = abs(fc)

        rows.append([i, a, b, c, fa, fb, fc, err])

        if abs(fc) < tol:
            break

        if fa * fc < 0:
            b = c
        else:
            a = c

    return rows


# -----------------------------------------
# 3. SECANT METHOD
# -----------------------------------------
def secant(func_str, x0, x1, tol=1e-6, max_iter=30):
    f, _, _ = make_function(func_str)
    rows = []

    for i in range(1, max_iter + 1):
        f0 = f(x0)
        f1 = f(x1)

        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        err = abs(x_new - x1)

        rows.append([i, x0, x1, f0, f1, x_new, err])

        if err < tol:
            break

        x0, x1 = x1, x_new

    return rows


# -----------------------------------------
# 4. NEWTON–RAPHSON METHOD
# -----------------------------------------
def newton_raphson(func_str, x0, tol=1e-6, max_iter=30):
    f, expr, x = make_function(func_str)
    fprime = sp.lambdify(x, sp.diff(expr, x), "math")

    rows = []

    for i in range(1, max_iter + 1):
        fx = f(x0)
        fpx = fprime(x0)

        if fpx == 0:
            raise ZeroDivisionError("Derivative is zero!")

        x_new = x0 - fx / fpx
        err = abs(x_new - x0)

        rows.append([i, x0, fx, fpx, x_new, err])

        if err < tol:
            break

        x0 = x_new

    return rows


# -----------------------------------------
# 5. FIXED POINT ITERATION
# -----------------------------------------
def fixed_point(g_str, x0, tol=1e-6, max_iter=30):
    g, _, _ = make_function(g_str)
    rows = []

    for i in range(1, max_iter + 1):
        x_new = g(x0)
        err = abs(x_new - x0)

        rows.append([i, x0, x_new, err])

        if err < tol:
            break

        x0 = x_new

    return rows


# -----------------------------------------
# 6. MODIFIED SECANT METHOD
# -----------------------------------------
def modified_secant(func_str, x0, delta=0.01, tol=1e-6, max_iter=30):
    f, _, _ = make_function(func_str)
    rows = []

    for i in range(1, max_iter + 1):
        fx = f(x0)
        f_x_delta = f(x0 + delta * x0)

        derivative = (f_x_delta - fx) / (delta * x0)
        x_new = x0 - fx / derivative
        err = abs(x_new - x0)

        rows.append([i, x0, fx, derivative, x_new, err])

        if err < tol:
            break

        x0 = x_new

    return rows
