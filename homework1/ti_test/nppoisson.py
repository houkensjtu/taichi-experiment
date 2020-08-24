import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import cg
import time

n = 1000

A = np.zeros((n, n))
b = np.zeros(n)
x = np.zeros(n)


def init():
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 2.0
            elif np.abs(i-j) == 1:
                A[i, j] = -1.0
            else:
                A[i, j] = 0.0

    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[n-1, n-1] = 1.0
    A[n-1, n-2] = 0.0
    for i in range(n):
        b[i] = 0.0
        x[i] = 0.0
    b[0] = 100
    b[n-1] = 0
    # print("Converting...")
    A_sparse = csc_matrix(A)
    b_sparse = csc_matrix(b)


def disp():
    global x
    for i in range(n):
        print("x[", i, "] = ", x[i])


def jacobian_solve():
    res = 0.0
    x[0] = 1/A[0, 0] * (b[0] - A[0, 1]*x[1])
    for i in range(1, n-1):
        x[i] = 1/A[i, i] * (b[i] - A[i, i-1]*x[i-1] - A[i, i+1]*x[i+1])
    x[n-1] = 1/A[n-1, n-1] * (b[n-1] - A[n-1, n-2]*x[n-2])
    for i in range(1, n-1):
        res = res + b[i] - A[i, i-1]*x[i-1] - A[i, i+1]*x[i+1] - A[i, i]*x[i]
    res = res + (b[0] - A[0, 1]*x[1] - A[0, 0]*x[0])
    res = res + (b[n-1] - A[n-1, n-2]*x[n-2] - A[n-1, n-1] * x[n-1])
    return res


def conjgrad(A, b, x):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : matrix 
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(np.transpose(r), r)

    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < 1e-8:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x


def numsolve():
    return np.linalg.solve(A, b)


def custom_cg():
    return conjgrad(A, b, x)


def bicgsolve():
    x, exitCode = bicg(A, b, atol=1e-6)
    return x


def main():
    init()
    res = 1.0
    iter = 0
    print("Solving using Jacobian...")
    while res > 1e-6:
        iter += 1
        res = jacobian_solve()
        if iter % 1000 == 0:
            print("Iteration = ", iter, "Residual = ", res)
    # disp()


def main2():
    init()
    print("Solving using numpy solve...")
    x = numsolve()
    # for i in range(n):
    #print("x[", i, "] = ", x[i])


def main3():
    init()
    print("Solving using bicg...")
    x = bicgsolve()


def main4():
    init()
    x = custom_cg()


start = time.time()
main4()
print(time.time()-start)
