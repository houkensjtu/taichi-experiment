import taichi as ti
import numpy as np

ti.init()

lx = 1.0
ly = 0.3

nx = 12
ny = 3

velo_rel = 0.01
p_rel = 0.03

# Add 1 cell padding to all directions.
p = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))
pcor = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))

u = ti.var(dt=ti.f32, shape=(nx + 3, ny + 2))
u0 = ti.var(dt=ti.f32, shape=(nx + 3, ny + 2))
u_post = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))

v = ti.var(dt=ti.f32, shape=(nx + 2, ny + 3))
v0 = ti.var(dt=ti.f32, shape=(nx + 2, ny + 3))
v_post = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))

# ct stands for Cell Type.
# ct = 0 -> Fluid
# ct = 1 -> Solid
ct = ti.var(dt=ti.i32, shape=(nx + 2, ny + 2))

rho = 100
mu = 0.1
dx = lx / nx
dy = ly / ny
dt = 0.1

Au = ti.var(dt=ti.f32, shape=((nx + 1) * ny, (nx + 1) * ny))
bu = ti.var(dt=ti.f32, shape=((nx + 1) * ny))
xu = ti.var(dt=ti.f32, shape=((nx + 1) * ny))

Av = ti.var(dt=ti.f32, shape=(nx * (ny + 1), nx * (ny + 1)))
bv = ti.var(dt=ti.f32, shape=(nx * (ny + 1)))
xv = ti.var(dt=ti.f32, shape=(nx * (ny + 1)))

Ap = ti.var(dt=ti.f32, shape=(nx * ny, nx * ny))
bp = ti.var(dt=ti.f32, shape=(nx * ny))
xp = ti.var(dt=ti.f32, shape=(nx * ny))

@ti.func
def init():
    print(" ...for i,j in ndrange...")
    for i, j in ti.ndrange(nx + 2, ny + 2):
        p[i, j] = 100 - i / nx
        print("P at i =", i, "j =", j, " = ", p[i,j])
    print(" ...for i,j in p...")
    for i, j in p:
        p[i, j] = 100 - i / nx
        print("P at i =", i, "j =", j, " = ", p[i,j])

@ti.kernel
def main():
     init()

if __name__=="__main__":
     main()     