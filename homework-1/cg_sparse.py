import taichi as ti
import random
import time

ti.init(default_fp=ti.f64)

n = 1000

A = ti.field(dtype=ti.f64)
ti.root.pointer(ti.ij, (n, n)).place(A)

x = ti.field(dtype=ti.f64)
b = ti.field(dtype=ti.f64)
x_new = ti.field(dtype=ti.f64)
ti.root.pointer(ti.i, n).place(x, b, x_new)

# Vectors for cg solver
r = ti.field(dtype=ti.f64)
p = ti.field(dtype=ti.f64)
Ax = ti.field(dtype=ti.f64)
Ap = ti.field(dtype=ti.f64)
ti.root.pointer(ti.i, n).place(r, p, Ax, Ap)


@ti.kernel
def init():
    for i, j in ti.ndrange(n, n):
        if i == j:
            A[i, j] = 2.0
        elif ti.abs(i - j) == 1:
            A[i, j] = -1.0
    for i in ti.ndrange(n):
        b[i] = 0.0
        x[i] = 0.0
        Ax[i] = 0.0
        Ap[i] = 0.0
    b[0] = 100.0
    b[n - 1] = 0.0


@ti.kernel
def print_A():
    for i, j in A:
        print("A[", i, ",", j, "] = ", A[i, j])


@ti.func
def full_jacobian():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] * x[j]
        x_new[i] = r / A[i, i]
        x[i] = r / A[i, i]
    for i in range(n):
        x[i] = x_new[i]

    res = 0.0

    for i in range(n):
        r = b[i] * 1.0
        for j in range(n):
            r -= A[i, j] * x[j]
        res += r * r
    return ti.sqrt(res)


@ti.func
def jacobian_iterate():
    res = 1.0
    iter = 0
    while res > 1e-8:
        iter += 1
        # res = full_jacobian()
        res = full_jacobian()
        if iter % 1 == 0:
            print("Iteration = ", iter, "Residual = ", res)


@ti.func
def test():
    for i, j in A:
        if i != j:
            print(b[i])


@ti.kernel
def main():
    # conjgrad()
    # quick_conjgrad()
    # jacobian_iterate()
    test()


if __name__ == "__main__":
    init()
    main()
