# Purpose:
# Develop a bicg solver and compare the performance with Jacobian.

# Notes:
# M is the preconditioner. For the momoent, M = diag(A).

# Results:
# 1. For a 128 x 128 matrix A, the performance is 100x faster than Jacobian.
# => About 0.18sec vs. 18sec using Jacobian.
# 2. Compared with struct cg, bicg is about 60% slower because computing the A transpose
# times p.
# 3. At larger scale (>256), bicg with limited for loop range is much faster than
# full loop bicg. On 1024 A matrix, full loop takes 12 sec while limited for loop
# takes only 0.28sec.

import taichi as ti
import time

ti.init(default_fp=ti.f64)

# Choose n = 99 so that the exact solution will be 1,2,3,...99
n = 1024
A = ti.field(dtype=ti.f64)
M = ti.field(dtype=ti.f64)
ti.root.pointer(ti.ij, (n//8, n//8)).pointer(ti.ij, (8, 8)).place(A, M)
#ti.root.dense(ti.ij, (n, n)).place(A,M)
x = ti.field(dtype=ti.f64, shape=n)
b = ti.field(dtype=ti.f64, shape=n)
x_new = ti.field(dtype=ti.f64, shape=n)

# Vectors for bicg (or cg) solver
r = ti.field(dtype=ti.f64, shape=n)
z = ti.field(dtype=ti.f64, shape=n)
p = ti.field(dtype=ti.f64, shape=n)
Ax = ti.field(dtype=ti.f64, shape=n)
Ap = ti.field(dtype=ti.f64, shape=n)
Ap_tld = ti.field(dtype=ti.f64, shape=n)

r_tld = ti.field(dtype=ti.f64, shape=n)
p_tld = ti.field(dtype=ti.f64, shape=n)
z_tld = ti.field(dtype=ti.f64, shape=n)


@ti.kernel
def init():
    for i, j in ti.ndrange(n, n):
        if i == j:
            A[i, j] = 2.0
            M[i, j] = A[i, j]
        elif ti.abs(i - j) == 1:
            A[i, j] = -1.0
    for i in x:
        b[i] = 0.0
        x[i] = 0.0
        Ax[i] = 0.0
        Ap[i] = 0.0
    b[0] = 100.0
    b[n - 1] = 0.0


@ti.func
def print_A():
    for i in range(n):
        for j in range(n):
            print("A[", i, ",", j, "] = ", A[i, j])


# @ti.kernel
def check_sol() -> ti.f64:
    res = 0.0
    r = 0.0
    for i in range(n):
        r = b[i]
        for j in range(n):
            r -= A[i, j] * x[j]
        res += r * r
    return res


# Quick version of conjugate gradient
# Only multiply non-zero elements in A
# Other calculations are exactly same

@ti.func
def struct_cg():
    # dot(A,x)
    for i in range(n):
        Ax[i] = 0.0
        for j in range(i-1, i+2):
            Ax[i] += A[i, j] * x[j]
    # r = b - dot(A,x)
    # p = r
    for i in range(n):
        r[i] = b[i] - Ax[i]
        p[i] = r[i]
    rsold = 0.0
    for i in range(n):
        rsold += r[i] * r[i]

    for steps in range(100*n):
        # dot(A,p)
        for i in range(n):
            Ap[i] = 0.0
            for j in range(i-1, i+2):
                Ap[i] += A[i, j] * p[j]

        # dot(p, Ap) => pAp
        pAp = 0.0
        for i in range(n):
            pAp += p[i] * Ap[i]

        alpha = rsold / pAp

        # x = x + dot(alpha,p)
        # r = r - dot(alpha,Ap)
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]

        rsnew = 0.0
        for i in range(n):
            rsnew += r[i] * r[i]

        if ti.sqrt(rsnew) < 1e-8:
            print("The solution has converged...")
            break

        for i in range(n):
            p[i] = r[i] + (rsnew / rsold) * p[i]
        rsold = rsnew

        print("Iteration ", steps, ", residual = ", rsold)


@ti.func
def bicg():

    # dot(A,x)
    for i in range(n):
        Ax[i] = 0.0
        for j in range(i-1, i+2):
            #        for j in range(n):
            Ax[i] += A[i, j] * x[j]

    # r = b - dot(A,x)
    for i in range(n):
        r[i] = b[i] - Ax[i]
        r_tld[i] = r[i]

    rsold = 0.0
    for i in range(n):
        rsold += r[i] * r[i]

    rho_1 = 1.0
    for steps in range(100*n):

        for i in range(n):
            z[i] = 1.0 / M[i, i] * r[i]
            z_tld[i] = 1.0 / M[i, i] * r_tld[i]

        rho = 0.0
        for i in range(n):
            rho += z[i] * r_tld[i]
        if rho == 0.0:
            print("Bicg failed...")

        if steps == 0:
            for i in range(n):
                p[i] = z[i]
                p_tld[i] = z_tld[i]
        else:
            beta = rho / rho_1
            for i in range(n):
                p[i] = z[i] + beta * p[i]
                p_tld[i] = z_tld[i] + beta * p_tld[i]

        # dot(A,p)
        for i in range(n):
            Ap[i] = 0.0
            Ap_tld[i] = 0.0
            for j in range(i-1, i+2):
                # for j in range(n):
                # Ap => q
                Ap[i] += A[i, j] * p[j]
                # Ap_tld => q_tld
                Ap_tld[i] += A[j, i] * p_tld[j]

        # dot(p, Ap) => pAp
        pAp = 0.0
        for i in range(n):
            pAp += p_tld[i] * Ap[i]

        alpha = rho / pAp

        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
            r_tld[i] -= alpha * Ap_tld[i]

        rsnew = 0.0
        for i in range(n):
            rsnew += r[i] * r[i]
        rsold = rsnew
        print("Iteration ", steps, ", residual = ", rsold)

        if ti.sqrt(rsnew) < 1e-8:
            print("The solution has converged...")
            break

        rho_1 = rho


@ti.kernel
def main():
    bicg()
    # struct_cg()


if __name__ == "__main__":
    init()

    start = time.time()
    main()
    end = time.time()

    for i in range(n):
        print("x[", i, "] = ", x[i])

    #res = check_sol()
    #print("The final residual is ", res)
    print("Time collapsed is: ", end-start, " sec.")
