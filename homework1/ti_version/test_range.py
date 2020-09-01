import taichi as ti

ti.init()

n = 5

a = ti.field(dtype = ti.f64)
ti.root.pointer(ti.ij, (n,n)).place(a)

@ti.kernel
def jump():
    a[1,1] = 1
    a[2,2] = 1
    a[3,3] = 1
    for i,j in a:
        print("a[", i, ",", j,"] = ", a[i,j])
        
jump()
