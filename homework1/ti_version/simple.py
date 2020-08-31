import taichi as ti

ti.init(default_fp = ti.f64)

lx = 1.5
ly = 0.3

nx = 60
ny = 20

velo_rel = 0.01
p_rel = 0.03

# Add 1 cell padding to all directions.
p = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 2))
pcor = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 2))

u = ti.field(dtype=ti.f64, shape=(nx + 3, ny + 2))
u0 = ti.field(dtype=ti.f64, shape=(nx + 3, ny + 2))
ucor = ti.field(dtype=ti.f64, shape=(nx + 3, ny + 2))
u_post = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 2))

v = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 3))
vcor = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 3))
v0 = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 3))
v_post = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 2))

# ct stands for Cell Type.
# ct = 0 -> Fluid
# ct = 1 -> Solid
ct = ti.field(dtype=ti.i32, shape=(nx + 2, ny + 2))

rho = 100
mu = 0.1
dx = lx / nx
dy = ly / ny
dt = 0.001

Au = ti.field(dtype=ti.f64, shape=((nx + 1) * ny, (nx + 1) * ny))
bu = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))
xu = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))
xuold = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))

Av = ti.field(dtype=ti.f64, shape=(nx * (ny + 1), nx * (ny + 1)))
bv = ti.field(dtype=ti.f64, shape=(nx * (ny + 1)))
xv = ti.field(dtype=ti.f64, shape=(nx * (ny + 1)))
xvold = ti.field(dtype=ti.f64, shape=(nx * (ny + 1)))

Ap = ti.field(dtype=ti.f64, shape=(nx * ny, nx * ny))
bp = ti.field(dtype=ti.f64, shape=(nx * ny))
xp = ti.field(dtype=ti.f64, shape=(nx * ny))

@ti.kernel
def init():
    for i, j in ti.ndrange(nx + 2, ny + 2):
        p[i, j] = 100 - i / nx
    for i, j in ti.ndrange(nx + 3, ny + 2):
        u[i, j] = 5.0
        u0[i, j] = u[i, j]
    for i, j in ti.ndrange(nx + 2, ny + 3):
        v[i, j] = 0.0
        v0[i, j] = v[i, j]

    for i, j in ti.ndrange(nx + 2, ny + 2):
        ct[i, j] = 1  # "1" stands for solid
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 1)):
        ct[i, j] = -1  # "-1" stands for fluid

    for i, j in ti.ndrange(nx, ny):
        if (((i - 31)**2 + (j - 31)**2) < 36):
            ct[i, j] = 1
            u[i, j] = 0
            u0[i, j] = 0
            v[i, j] = 0
            v0[i, j] = 0


if __name__ == "__main__":
    init()
