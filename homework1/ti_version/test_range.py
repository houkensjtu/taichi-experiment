import taichi as ti

ti.init()

n = 10
a = ti.field(dtype = ti.f64, shape=(n,n))
b = [1,3,5,7]
c = ti.field(dtype = ti.i32, shape = 3)
@ti.kernel
def traverse():
    c[0] = 1
    c[1] = 3
    c[2] = 5
    for i in range(n):
        for j in c:
            print(a[i,j])

traverse()
