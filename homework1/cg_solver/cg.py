import taichi as ti

ti.init()

n = 100

A = ti.field(dtype=ti.f64, shape=(n,n))
x = ti.field(dtype=ti.f64, shape=n)
b = ti.field(dtype=ti.f64, shape=n)

# Vectors for cg solver
r = ti.field(dtype=ti.f64, shape=n)
p = ti.field(dtype=ti.f64, shape=n)
Ax = ti.field(dtype=ti.f64, shape=n)
Ap = ti.field(dtype=ti.f64, shape=n)


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
        x[i] = 1.0
        Ax[i] = 0.0
        Ap[i] = 0.0
    b[0] = 100
    b[n-1] = 0

@ti.func
def test_access():
    for i in range(n):
        for j in range(n):
            print("A[",i,",",j,"] = ",A[i,j])

@ti.func
def dot():
    for i in range(n):
        Ax[i] = 0.0
        for j in range(n):
            Ax[i] += A[i,j] * x[j]

@ti.func
def cg2():
    # dot(A,x)
    for i in range(n):
        Ax[i] = 0.0
        for j in range(n):
            Ax[i] += A[i,j] * x[j]
    # r = b - dot(A,x)
    # p = r
    for i in range(n):
        r[i] = b[i] - Ax[i]
        p[i] = r[i]
    rsold = 0.0
    for i in range(n):
        rsold += r[i] * r[i]

    for _ in range(10000):
        # dot(A,p)
        for i in range(n):
            Ap[i] = 0.0
            for j in range(n):
                Ap[i] += A[i,j] * p[j]

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

        if ti.sqrt(rsnew) < 1e-6:
            break

        for i in range(n):
            p[i] = r[i] + (rsnew/rsold) * p[i]
        rsold = rsnew
        print("iter = ", _ ,  "residual = ", rsold)
            
            
@ti.kernel
def main():
    init()
    cg2()
    for i in range(n):
        print("x[",i,"] = ",x[i])

#    cg()

main()    
