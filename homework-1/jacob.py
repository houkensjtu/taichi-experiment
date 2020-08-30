import taichi as ti
import random

ti.init(default_fp=ti.f64)

n = 20

A = ti.field(dtype=ti.f64, shape=(n, n))
x = ti.field(dtype=ti.f64, shape=n)
new_x = ti.field(dtype=ti.f64, shape=n)
b = ti.field(dtype=ti.f64, shape=n)


@ti.kernel
def iterate():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] * x[j]

        # new_x[i] = r / A[i, i]
        x[i] = r / A[i, i]

    #for i in range(n):
    #x[i] = new_x[i]


@ti.kernel
def residual() -> ti.f32:
    res = 0.0

    for i in range(n):
        r = b[i] * 1.0
        for j in range(n):
            r -= A[i, j] * x[j]
        res += r * r

    return res


@ti.kernel
def print_A():
    for i in range(n):
        for j in range(n):
            print("A[", i, ",", j, "] = ", A[i, j])


for i in range(n):
    for j in range(n):
        A[i, j] = random.random() - 0.5

    A[i, i] += n * 0.1

    b[i] = random.random() * 100

print_A()
res = residual()
i = 0
while res > 1.0e-8:
    i += 1
    iterate()
    res = residual()
    print(f'iter {i}, residual={residual():0.10f}')

for i in range(n):
    lhs = 0.0
    for j in range(n):
        lhs += A[i, j] * x[j]
    assert abs(lhs - b[i]) < 1e-4
