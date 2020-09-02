import taichi as ti
import random
import time

ti.init(default_fp=ti.f64)

n = 50

A = ti.field(dtype=ti.f64)
ti.root.pointer(ti.ij, (n, n)).place(A)

x = ti.field(dtype=ti.f64)
b = ti.field(dtype=ti.f64)
r = ti.field(dtype=ti.f64)
x_new = ti.field(dtype=ti.f64)
ti.root.pointer(ti.i, n).place(x, b, x_new,r)


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
        
    b[0] = 100.0


@ti.kernel
def full_jacobian()->ti.f64:
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


@ti.kernel
def full_jacobian_sparse()->ti.f64:
    for i in b:
        r[i] = b[i]
    for i,j in A:
        if i != j:
            r[i] -= A[i,j] * x[j]
    for i in r:
        x[i] = r[i] / A[i,i]

    res = 0.0
    
    for i in b:
        r[i] = b[i]
    for i,j in A:
        r[i] -= A[i,j] * x[j]

    for i in r:
        res += r[i] * r[i]
    return ti.sqrt(res)


@ti.kernel
def disp_x():
    for i in range(n):
        print("x[",i,"] = ", x[i])
    

if __name__ == "__main__":
    init()

    # Jacobian iteration
    res = 1.0
    iter_full = 0
    start_full_jacob = time.time()
    while res > 1e-8:
        iter_full += 1
        res = full_jacobian()
        print("Iteration = ", iter_full, "Residual = ", res)
    end_full_jacob = time.time()
    
    init()
    # Sparse Jacobian iteration
    res = 1.0
    iter_sparse = 0
    start_sparse_jacob = time.time()
    while res > 1e-8:
        iter_sparse += 1
        res = full_jacobian_sparse()
        print("Iteration = ", iter_sparse, "Residual = ", res)
    end_sparse_jacob = time.time()
    
    disp_x()

    print("Full Jacobian iteration took ", end_full_jacob - start_full_jacob , "sec and ", iter_full, "steps.")
    print("Sparse Jacobian iteration took ", end_sparse_jacob - start_sparse_jacob , "sec and ", iter_sparse,"steps.")    
