import taichi as ti

ti.init(default_fp = ti.f64)

lx = 1.5
ly = 0.3

nx = 300
ny = 80

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

@ti.kernel            
def fill_Au():
    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)

        # Inlet and Outlet
        if (ct[i - 1, j]) == 1 or (ct[i, j] + ct[i - 1, j]) == 2:
            Au[k, k] = 1.0
            bu[k] = u[i, j]
        # Outlet
        elif (ct[i, j] == 1):
            Au[k, k] = 1.0
            Au[k, k - ny] = -1.0
            #bu[k] = u[i - 1, j]
            bu[k] = 0.0

        # Normal internal cells
        else:
            Au[k, k - 1] = -mu * dx / dy - ti.max(0, -rho * 0.5 * (v[i - 1, j] + v[i, j]) * dx)  # an
            Au[k, k + 1] = -mu * dx / dy - ti.max(0, rho * 0.5 * (v[i - 1, j + 1] + v[i, j + 1]) * dx)  # as
            Au[k, k - ny] = -mu * dy / dx - ti.max(0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy)  # aw
            Au[k, k + ny] = -mu * dy / dx - ti.max(0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy)  # ae
            Au[k, k] = -Au[k, k - 1] - Au[k, k + 1] - Au[k, k - ny] - Au[k, k + ny] + rho * dx * dy / dt  # ap
            bu[k] = (p[i - 1, j] - p[i, j]) * dy + rho * dx * dy / dt * u0[i, j]  # <= Unsteady term

    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        # Upper and lower boundary
        if (ct[i, j] + ct[i, j - 1]) == 0:
            Au[k, k] = Au[k, k] - Au[k, k - 1] + 2 * mu
            Au[k, k - 1] = 0
        elif (ct[i, j] + ct[i, j + 1]) == 0:
            Au[k, k] = Au[k, k] - Au[k, k + 1] + 2 * mu
            Au[k, k + 1] = 0

@ti.kernel
def fill_Av():
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 2)):
        k = (i - 1) * (ny + 1) + (j - 1)
        # Upper and lower boundary
        if (ct[i, j] + ct[i, j - 1]) == 0 or (ct[i, j] + ct[i, j - 1]) == 2:
            Av[k, k] = 1.0
            bv[k] = v[i, j]
        else:
            """
            TODO: Didn't cover inlet and outlet boundary. Actually accessing
            elements out of bound, for example, Av[1,-30].
            However, since in solve_v, when convert to numpy, A[1,-30] become
            0.0 automatically.
            """
            Av[k, k - 1] = -mu * dx / dy - ti.max(0, -rho * 0.5 * (v[i, j - 1] + v[i, j]) * dx)  # an
            Av[k, k + 1] = -mu * dx / dy - ti.max(0, rho * 0.5 * (v[i, j + 1] + v[i, j]) * dx)  # as
            Av[k, k - ny - 1] = -mu * dy / dx - ti.max(0, rho * 0.5 * (u[i, j] + u[i, j - 1]) * dy)  # aw
            Av[k, k + ny + 1] = -mu * dy / dx - ti.max(0, -rho * 0.5 * (u[i + 1, j - 1] + u[i + 1, j]) * dy)  # ae
            Av[k, k] = -Av[k, k - 1] - Av[k, k + 1] - Av[k, k - ny - 1] - Av[k, k + ny + 1] + rho * dx * dy / dt  # ap
            bv[k] = (p[i, j] - p[i, j - 1]) * dx + rho * dx * dy / dt * v0[i,j]



if __name__ == "__main__":
    init()
    fill_Au()
