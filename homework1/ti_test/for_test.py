import taichi as ti

ti.init()

A = ti.field(dtype=ti.f32, shape=5)


@ti.func
def init():
    for i in A:
        A[i] = 1.0


@ti.kernel
def main():
    init()
    for i in A:
        print(A[i])


main()
