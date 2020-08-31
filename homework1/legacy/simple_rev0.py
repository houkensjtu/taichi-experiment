import taichi as ti
import numpy as np
import matplotlib.cm as cm

ti.init()

lx = 1.
ly = 0.2

nx = 100
ny = 20

p = ti.var(dt=ti.f32, shape=(nx, ny))

u = ti.var(dt=ti.f32, shape=(nx + 1, ny))
u_post = ti.var(dt=ti.f32, shape=(nx, ny))

v = ti.var(dt=ti.f32, shape=(nx, ny + 1))
v_post = ti.var(dt=ti.f32, shape=(nx, ny))

# Boundary condition will be specified by a 4 element vec
# [west, north, east, south]
# 0 stands for fluid, 1 stands for solid.
bc = ti.Vector(4, dt=ti.i32, shape=(nx, ny))

rho = 10
mu = 0.01
dx = lx / nx
dy = ly / ny
dt = 0.01

Au = ti.var(dt=ti.f32, shape=((nx + 1) * ny, (nx + 1) * ny))
bu = ti.var(dt=ti.f32, shape=((nx + 1) * ny))
xu = ti.var(dt=ti.f32, shape=((nx + 1) * ny))

Av = ti.var(dt=ti.f32, shape=(nx * (ny + 1), nx * (ny + 1)))
bv = ti.var(dt=ti.f32, shape=(nx * (ny + 1)))
xv = ti.var(dt=ti.f32, shape=(nx * (ny + 1)))


@ti.kernel
def fill_p():
    for i, j in ti.ndrange(nx, ny):
        p[i, j] = 1 - i / nx


@ti.kernel
def fill_u():
    for i, j in ti.ndrange(nx + 1, ny):
        u[i, j] = 1.0


@ti.kernel
def fill_v():
    for i, j in ti.ndrange(nx, ny + 1):
        v[i, j] = 0.0


def fill_bc():
    for i, j, k in ti.ndrange(nx, ny, 4):
        bc[i, j][k] = 0
    for j in range(ny):
        bc[0, j][0] = 1
        bc[nx - 1, j][2] = 1
    for i in range(1, nx - 1):
        bc[i, 0][1] = 1
        bc[i, ny - 1][3] = 1
    for i, j, k in ti.ndrange(nx, ny, 4):
        #print("bc[", i, ",", j, ",", k, "] = ", bc[i, j][k])
        pass


def fill_xu():
    for i, j in ti.ndrange(nx + 1, ny):
        xu[i * ny + j] = u[i, j]


def xu_back():
    for i, j in ti.ndrange(nx + 1, ny):
        u[i, j] = xu[i * ny + j]


def fill_xv():
    for i, j in ti.ndrange(nx, ny + 1):
        xv[i * ny + j] = v[i, j]


def xv_back():
    for i, j in ti.ndrange(nx, ny + 1):
        v[i, j] = xv[i * ny + j]


def fill_Au():
    for i, j in ti.ndrange(nx, ny):
        k = i * ny + j

        if bc[i, j][0] == 1:

            Au[k, k] = 1.0
            bu[k] = u[i, j]

        elif bc[i, j][1] == 1:

            Au[k, k - 1] = 0.0  # an
            Au[k, k + 1] = -mu * dx / dy - max(
                [0, rho * 0.5 * (v[i - 1, j + 1] + v[i, j + 1]) * dx])  # as
            Au[k, k - ny] = -mu * dy / dx - max(
                [0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy])  # aw
            Au[k, k + ny] = -mu * dy / dx - max(
                [0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy])  # ae
            Au[k, k] = -Au[k, k - 1] - Au[k, k + 1] - Au[k, k - ny] - Au[
                k, k + ny] + 2 * mu  # ap
            bu[k] = (p[i - 1, j] - p[i, j]) * dy

        elif bc[i, j][3] == 1:

            Au[k, k - 1] = -mu * dx / dy - max(
                [0, -rho * 0.5 * (v[i - 1, j] + v[i, j]) * dx])  # an
            Au[k, k + 1] = 0.0  # as
            Au[k, k - ny] = -mu * dy / dx - max(
                [0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy])  # aw
            Au[k, k + ny] = -mu * dy / dx - max(
                [0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy])  # ae
            Au[k, k] = -Au[k, k - 1] - Au[k, k + 1] - Au[k, k - ny] - Au[
                k, k + ny] + 2 * mu  # ap
            bu[k] = (p[i - 1, j] - p[i, j]) * dy

        elif bc[i, j][2] == 1:
            # i = [0, nx-1], j = [0,ny-1] => k = (nx-1) * ny + [0,ny-1]
            # k + ny :=> [nx*ny, (nx+1)*ny-1]

            Au[k + ny, k + ny] = 1.0
            bu[k + ny] = u[i + 1, j]
            Au[k, k - 1] = -mu * dx / dy - max(
                [0, -rho * 0.5 * (v[i - 1, j] + v[i, j]) * dx])  # an
            Au[k, k + 1] = -mu * dx / dy - max(
                [0, rho * 0.5 * (v[i - 1, j + 1] + v[i, j + 1]) * dx])  # as
            Au[k, k - ny] = -mu * dy / dx - max(
                [0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy])  # aw
            Au[k, k + ny] = -mu * dy / dx - max(
                [0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy])  # ae
            Au[k,
               k] = -Au[k, k - 1] - Au[k, k + 1] - Au[k, k - ny] - Au[k, k +
                                                                      ny]  # ap
            bu[k] = (p[i - 1, j] - p[i, j]) * dy

        else:

            Au[k, k - 1] = -mu * dx / dy - max(
                [0, -rho * 0.5 * (v[i - 1, j] + v[i, j]) * dx])  # an
            Au[k, k + 1] = -mu * dx / dy - max(
                [0, rho * 0.5 * (v[i - 1, j + 1] + v[i, j + 1]) * dx])  # as
            Au[k, k - ny] = -mu * dy / dx - max(
                [0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy])  # aw
            Au[k, k + ny] = -mu * dy / dx - max(
                [0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy])  # ae
            Au[k,
               k] = -Au[k, k - 1] - Au[k, k + 1] - Au[k, k - ny] - Au[k, k +
                                                                      ny]  # ap
            bu[k] = (p[i - 1, j] - p[i, j]) * dy


def fill_Av():
    for i, j in ti.ndrange(nx, ny):
        k = i * (ny + 1) + j
        if bc[i, j][0] == 1:
            # print("case 1")
            Av[k, k] = 1.0
            bv[k] = v[i, j]
            Av[k + 1, k + 1] = 1.0
            bv[k + 1] = v[i, j + 1]
            # print("i = ", i, "j = ", j, "k = ", k, "Av[k,k] = ", Av[k, k])
        elif bc[i, j][2] == 1:
            # print("case 2")
            Av[k, k] = 1.0
            bv[k] = v[i, j]
            Av[k + 1, k + 1] = 1.0
            bv[k + 1] = v[i, j + 1]
            # print("i = ", i, "j = ", j, "k = ", k, "Av[k,k] = ", Av[k, k])
        elif bc[i, j][1] == 1:
            # print("case 3")
            Av[k, k] = 1.0
            bv[k] = v[i, j]
            # print("i = ", i, "j = ", j, "k = ", k, "Av[k,k] = ", Av[k, k])
        elif bc[i, j][3] == 1:
            # print("case 3")
            Av[k, k - 1] = -mu * dx / dy - max(
                [0, -rho * 0.5 * (v[i, j - 1] + v[i, j]) * dx])  # an
            Av[k, k + 1] = -mu * dx / dy - max(
                [0, rho * 0.5 * (v[i, j + 1] + v[i, j]) * dx])  # as
            Av[k, k - ny - 1] = -mu * dy / dx - max(
                [0, rho * 0.5 * (u[i, j] + u[i, j - 1]) * dy])  # aw
            Av[k, k + ny + 1] = -mu * dy / dx - max(
                [0, -rho * 0.5 * (u[i + 1, j - 1] + u[i + 1, j]) * dy])  # ae
            Av[k,
               k] = -Av[k, k - 1] - Av[k, k + 1] - Av[k, k - ny -
                                                      1] - Av[k,
                                                              k + ny + 1]  # ap
            bv[k] = (p[i, j] - p[i, j - 1]) * dx

            Av[k + 1, k + 1] = 1.0
            bv[k + 1] = v[i, j + 1]
            # print("i = ", i, "j = ", j, "k = ", k, "Av[k,k] = ", Av[k, k])
        else:
            Av[k, k - 1] = -mu * dx / dy - max(
                [0, -rho * 0.5 * (v[i, j - 1] + v[i, j]) * dx])  # an
            Av[k, k + 1] = -mu * dx / dy - max(
                [0, rho * 0.5 * (v[i, j + 1] + v[i, j]) * dx])  # as
            Av[k, k - ny - 1] = -mu * dy / dx - max(
                [0, rho * 0.5 * (u[i, j] + u[i, j - 1]) * dy])  # aw
            Av[k, k + ny + 1] = -mu * dy / dx - max(
                [0, -rho * 0.5 * (u[i + 1, j - 1] + u[i + 1, j]) * dy])  # ae
            Av[k,
               k] = -Av[k, k - 1] - Av[k, k + 1] - Av[k, k - ny -
                                                      1] - Av[k,
                                                              k + ny + 1]  # ap
            bv[k] = (p[i, j] - p[i, j - 1]) * dx


def solve_u():
    A = Au.to_numpy()
    b = bu.to_numpy()
    print(np.linalg.matrix_rank(A))
    return np.linalg.solve(A, b)


def solve_v():
    A = Av.to_numpy()
    b = bv.to_numpy()
    print(np.linalg.matrix_rank(A))
    return np.linalg.solve(A, b)


@ti.kernel
def iter_solve_u():
    #A = Au.to_numpy()
    #b = bu.to_numpy()
    for i, j in ti.ndrange(nx + 1, ny):
        k = i * ny + j
        #print("k = ", k, "ny = ", ny, "k-ny = ", k - ny, "Au[k-ny] = ",
        #Au[k - ny])
        xu[k] = 1 / Au[k, k] * (
            -Au[k, k - 1] * u[i, j - 1] - Au[k, k + 1] * u[i, j + 1] -
            Au[k, k - ny] * u[i - 1, j] - Au[k, k + ny] * u[i + 1, j] + bu[k])


def visual(mat):
    A = mat.to_numpy()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # 'nearest' interpolation - faithful but blocky
    plt.imshow(A, interpolation='nearest', cmap=cm.rainbow)
    plt.colorbar()
    plt.show()


@ti.kernel
def post_u():
    for i, j in ti.ndrange(nx, ny):
        u_post[i, j] = 0.5 * (u[i, j] + u[i + 1, j])


@ti.kernel
def post_v():
    for i, j in ti.ndrange(nx, ny):
        v_post[i, j] = 0.5 * (v[i, j] + v[i, j + 1])


def display():
    gui = ti.GUI('2D Simple viewer', (nx, 3 * ny))
    p_img = cm.terrain(p.to_numpy())
    u_img = cm.terrain(u_post.to_numpy())
    v_img = cm.terrain(v_post.to_numpy())
    img = np.concatenate((p_img, v_img, u_img), axis=1)
    while True:
        gui.set_image(img)
        gui.show()


if __name__ == "__main__":
    fill_p()
    fill_u()
    fill_v()
    fill_bc()
    fill_xu()
    fill_Au()
    #    visual(Au)
    xu = solve_u()

    fill_xv()
    fill_Av()
    # visual(Av)
    xv = solve_v()
    xu_back()
    xv_back()
    post_u()
    post_v()
    visual(u)
    visual(v)