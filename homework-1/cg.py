import taichi as ti
import random
import time

ti.init(default_fp=ti.f64)

n = 16384

A = ti.field(dtype=ti.f64, shape=(n, n))
x = ti.field(dtype=ti.f64, shape=n)
b = ti.field(dtype=ti.f64, shape=n)
x_new = ti.field(dtype=ti.f64, shape=n)

# Vectors for cg solver
r = ti.field(dtype=ti.f64, shape=n)
p = ti.field(dtype=ti.f64, shape=n)
Ax = ti.field(dtype=ti.f64, shape=n)
Ap = ti.field(dtype=ti.f64, shape=n)


@ti.kernel
def init():
    for i, j in A:
        if i == j:
            A[i, j] = 2.0
        elif ti.abs(i - j) == 1:
            A[i, j] = -1.0
        else:
            A[i, j] = 0.0
    for i in x:
        b[i] = 0.0
        x[i] = 0.0
        Ax[i] = 0.0
        Ap[i] = 0.0
    b[0] = 100.0
    b[n - 1] = 0.0


# @ti.kernel
def init_rand():
    for i in range(n):
        for j in range(n):
            A[i, j] = random.random() - 0.5
        A[i, i] += n * 0.1
        b[i] = random.random() * 100


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


@ti.func
def partial_jacobian():
    res = 0.0
    for i in ti.ndrange(n):
        x[i] = 1 / A[i, i] * (b[i] - A[i, i - 1] * x[i - 1] -
                              A[i, i + 1] * x[i + 1])
    for i in ti.ndrange(n):
        res = res + (b[i] - A[i, i - 1] * x[i - 1] - A[i, i + 1] * x[i + 1] -
                     A[i, i] * x[i])**2
    return res


@ti.func
def full_jacobian():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] * x[j]
        x_new[i] = r / A[i, i]

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
        res = full_jacobian()
        # res = partial_jacobian()
        if iter % 1 == 0:
            print("Iteration = ", iter, "Residual = ", res)


@ti.func
def conjgrad():
    # dot(A,x)
    for i in range(n):
        Ax[i] = 0.0
        for j in range(n):
            Ax[i] += A[i, j] * x[j]
    # r = b - dot(A,x)
    # p = r
    for i in range(n):
        r[i] = b[i] - Ax[i]
        p[i] = r[i]
    rsold = 0.0
    for i in range(n):
        rsold += r[i] * r[i]

    for steps in range(n):
        # dot(A,p)
        for i in range(n):
            Ap[i] = 0.0
            for j in range(n):
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

        if steps == n-1 and rsold > 1e-8:
            print("The solution did NOT converge...")
        return steps

# Quick version of conjugate gradient
# Only multiply non-zero elements in A
# Other calculations are exactly same


@ti.func
def quick_conjgrad():
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

    for steps in range(n):
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

        if steps == n-1 and rsold > 1e-8:
            print("The solution did NOT converge...")
        return steps


@ti.kernel
def main():
    # conjgrad()
    quick_conjgrad()
    # jacobian_iterate()


if __name__ == "__main__":
    init()
    # init_rand()

    start = time.time()
    main()
    end = time.time()

    for i in range(n):
        print("x[", i, "] = ", x[i])

    #res = check_sol()
    #print("The final residual is ", res)
    print("Time collapsed is: ", end-start, " sec.")
