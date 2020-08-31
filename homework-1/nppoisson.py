import numpy as np
import random
from scipy.sparse.linalg import bicg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import cg
import time

n = 1024

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
    for i in range(n):
        b[i] = 0.0
        x[i] = 0.0
    b[0] = 100.0
    b[n-1] = 0.0


def init_rand():
    for i in range(n):
        for j in range(n):
            A[i, j] = random.random() - 0.5
        A[i, i] += n * 0.1
        b[i] = random.random() * 100


def disp(x):
    for i in range(n):
        print("x[", i, "] = ", x[i])


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
        # print("Iter = ", i, "Residual = ", rsold)
    return x


def main():
    # init_rand()
    # x = np.linalg.solve(A, b)
    # conjgrad(A, b, x)
    # x, exitCode = cg(A, b, tol=1e-8)
    x, exitCode = bicg(A, b, tol=1e-8)
    #x, exitCode = bicgstab(A, b, tol=1e-8)
    return x


if __name__ == "__main__":
    init()
    # init_rand()

    start = time.time()
    x = main()
    end = time.time()

    for i in range(n):
        print("x[", i, "] = ", x[i])

    print("Time collapsed is: ", end-start, " sec.")
