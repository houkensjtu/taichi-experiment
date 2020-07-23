import taichi as ti
import numpy as np
import matplotlib.cm as cm

ti.init()

lx = 1.
ly = 0.2

nx = 100
ny = 30

# Add 1 cell padding to all directions.
p = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))
pcor = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))

# Velocity divergence
mdiv = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))

u = ti.var(dt=ti.f32, shape=(nx + 3, ny + 2))
u_post = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))

v = ti.var(dt=ti.f32, shape=(nx + 2, ny + 3))
v_post = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))

# ct stands for Cell Type.
# ct = 0 -> Fluid
# ct = 1 -> Solid
ct = ti.var(dt=ti.i32, shape=(nx + 2, ny + 2))

rho = 10
mu = 0.03
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


def init():
    for i, j in ti.ndrange(nx + 2, ny + 2):
        p[i, j] = 1000 - i / nx
        mdiv[i, j] = 0.0
    for i, j in ti.ndrange(nx + 3, ny + 2):
        u[i, j] = 2.0
    for i, j in ti.ndrange(nx + 2, ny + 3):
        v[i, j] = 0.0

    for i, j in ti.ndrange(nx + 2, ny + 2):
        ct[i, j] = 1  # "1" stands for solid
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 1)):
        ct[i, j] = -1  # "-1" stands for fluid

    for i, j in ti.ndrange((30, 40), (11, 21)):
        ct[i, j] = 1
        u[i, j] = 0


def fill_Au():
    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)

        # Inlet and Outlet
        if (ct[i, j] + ct[i - 1, j]) == 0:
            Au[k, k] = 1.0
            bu[k] = u[i, j]

        # Normal internal cells
        else:
            Au[k, k - 1] = -mu * dx / dy - max(
                [0, -rho * 0.5 * (v[i - 1, j] + v[i, j]) * dx])  # an
            Au[k, k + 1] = -mu * dx / dy - max(
                [0, rho * 0.5 * (v[i - 1, j + 1] + v[i, j + 1]) * dx])  # as
            Au[k, k - ny] = -mu * dy / dx - max(
                [0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy])  # aw
            Au[k, k + ny] = -mu * dy / dx - max(
                [0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy])  # ae
            Au[k, k] = -Au[k, k - 1] - Au[k, k + 1] - Au[k, k - ny] - Au[
                k, k + ny] + rho * dx * dy / dt  # ap
            bu[k] = (p[i - 1, j] - p[i, j]
                     ) * dy + rho * dx * dy / dt * u[i, j]  # <= Unsteady term

    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        # Upper and lower boundary
        if (ct[i, j] + ct[i, j - 1]) == 0:
            Au[k, k] = Au[k, k] - Au[k, k - 1] + 2 * mu
            Au[k, k - 1] = 0
        elif (ct[i, j] + ct[i, j + 1]) == 0:
            Au[k, k] = Au[k, k] - Au[k, k + 1] + 2 * mu
            Au[k, k + 1] = 0


def fill_Av():
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 2)):
        k = (i - 1) * (ny + 1) + (j - 1)
        # Upper and lower boundary
        if (ct[i, j] + ct[i, j - 1]) == 0:
            Av[k, k] = 1.0
            bv[k] = v[i, j]
        else:
            """
            TODO: Didn't cover inlet and outlet boundary. Actually accessing
            elements out of bound, for example, Av[1,-30].
            However, since in solve_v, when convert to numpy, A[1,-30] become
            0.0 automatically.
            """
            Av[k, k - 1] = -mu * dx / dy - max(
                [0, -rho * 0.5 * (v[i, j - 1] + v[i, j]) * dx])  # an
            Av[k, k + 1] = -mu * dx / dy - max(
                [0, rho * 0.5 * (v[i, j + 1] + v[i, j]) * dx])  # as

            Av[k, k - ny - 1] = -mu * dy / dx - max(
                [0, rho * 0.5 * (u[i, j] + u[i, j - 1]) * dy])  # aw
            Av[k, k + ny + 1] = -mu * dy / dx - max(
                [0, -rho * 0.5 * (u[i + 1, j - 1] + u[i + 1, j]) * dy])  # ae
            Av[k, k] = -Av[k, k - 1] - Av[k, k + 1] - Av[k, k - ny - 1] - Av[
                k, k + ny + 1] + rho * dx * dy / dt  # ap
            bv[k] = (p[i, j] - p[i, j - 1]) * dx + rho * dx * dy / dt * v[i, j]


def solve_axb(A, b):
    A_np = A.to_numpy()
    b_np = b.to_numpy()
    return np.linalg.solve(A_np, b_np)


def sol_back_matrix(mat, sol):
    mat_width = mat.shape()[0] - 2
    mat_height = mat.shape()[1] - 2
    for i, j in ti.ndrange(mat_width, mat_height):
        mat[i + 1, j + 1] = sol[i * mat_height + j]


def xu_back():
    for i, j in ti.ndrange(nx + 1, ny):
        u[i + 1, j + 1] = xu[i * ny + j]


def xv_back():
    for i, j in ti.ndrange(nx, ny + 1):
        v[i + 1, j + 1] = xv[i * ny + j]


def solve_moment_x():
    fill_Au()
    # solve_axb returns a numpy array
    # needs to convert back to taichi
    xu.from_numpy(solve_axb(Au, bu))
    sol_back_matrix(u, xu)


def solve_moment_y():
    fill_Av()
    xv.from_numpy(solve_axb(Av, bv))
    sol_back_matrix(v, xv)


def mass_div():
    for i, j in ti.ndrange(nx + 2, ny + 2):
        mdiv[i, j] = rho * dy * (u[i + 1, j] -
                                 u[i, j]) + rho * dx * (v[i, j] - v[i, j + 1])


def fill_Ap():
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        bp[k] = rho * (u[i + 1, j] - u[i, j]) * dy + rho * (v[i, j] -
                                                            v[i, j + 1]) * dx


def visual(mat):
    A = mat.to_numpy()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # 'nearest' interpolation - faithful but blocky
    plt.imshow(A, interpolation='nearest', cmap=cm.rainbow)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    init()
    gui = ti.GUI('2D Heat conduction', (nx + 3, ny + 2))
    for iter in range(1000):
        solve_moment_x()
        solve_moment_y()
        visual(u)
        #u_img = cm.terrain(u.to_numpy())
        #gui.set_image(u_img)
        #filename = f'frame_{iter:05d}.png'
        #gui.show(filename)
