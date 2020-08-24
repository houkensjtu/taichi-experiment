import taichi as ti
import time

ti.init(default_fp=ti.f64, arch=ti.cpu)

n = 1000

A = ti.field(dtype=ti.f64, shape=(n, n))
b = ti.field(dtype=ti.f64, shape=n)
x = ti.field(dtype=ti.f64, shape=n)
x_new = ti.field(dtype=ti.f64, shape=n)


@ti.func
def init():
    for i, j in A:
        if i == j:
            A[i, j] = 2.0
        elif ti.abs(i-j) == 1:
            A[i, j] = -1.0
        else:
            A[i, j] = 0.0

    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[n-1, n-1] = 1.0
    A[n-1, n-2] = 0.0
    for i in b:
        b[i] = 0.0
        x[i] = 0.0
    b[0] = 100
    b[n-1] = 0


@ti.func
def disp():
    for i in x:
        print("x[", i, "] = ", x[i])


@ti.func
def jacobian_solve():
    res = 0.0
    for i in ti.ndrange(n):
        x[i] = 1/A[i, i] * (b[i] - A[i, i-1]*x[i-1] - A[i, i+1]*x[i+1])
    for i in ti.ndrange(n):
        res = res + (b[i] - A[i, i-1]*x[i-1] - A[i, i+1]
                     * x[i+1] - A[i, i]*x[i]) ** 2
    return res


@ti.func
def full_jacobian():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] * x[j]
        #x_new[i] = r / A[i, i]
        x[i] = r / A[i, i]
    # for i in range(n):
    #    x[i] = x_new[i]
    res = 0.0
    for i in range(n):
        r = b[i] * 1.0
        for j in range(n):
            r -= A[i, j] * x[j]
        res += r * r
    return res


@ti.kernel
def main():
    init()
    res = 1.0
    iter = 0
    while res > 1e-6:
        iter += 1
        res = jacobian_solve()
        #res = full_jacobian()
        # if iter % 100 == 0:
        #print("Iteration = ", iter, "Residual = ", res)
    # disp()


start = time.time()
main()
print(time.time()-start)
