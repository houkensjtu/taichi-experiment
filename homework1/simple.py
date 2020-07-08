import taichi as ti
import numpy as np
import matplotlib.cm as cm

ti.init()

lx = 1.
ly = 0.2

nx = 200
ny = 40

p = ti.var(dt=ti.f32, shape=(nx, ny))

u = ti.var(dt=ti.f32, shape=(nx + 1, ny))
u_post = ti.var(dt=ti.f32, shape=(nx, ny))

v = ti.var(dt=ti.f32, shape=(nx, ny + 1))
v_post = ti.var(dt=ti.f32, shape=(nx, ny))

# Boundary condition will be specified by a 4 element vec
# [west, north, east, south]
# 0 stands for fluid, 1 stands for solid.
bc = ti.Vector(4, dt=ti.i32, shape=(nx, ny))

rho = 1.0
mu = 0.1
dx = lx / nx
dy = ly / ny
dt = 0.01

Au = ti.var(dt=ti.f32, shape=((nx + 1) * ny, (nx + 1) * ny))
bu = ti.var(dt=ti.f32, shape=((nx + 1) * ny))
xu = ti.var(dt=ti.f32, shape=((nx + 1) * ny))


@ti.kernel
def fill_p():
    for i, j in ti.ndrange(nx, ny):
        p[i, j] = 1.0 - i / nx


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
    for i in range(nx):
        bc[i, 0][1] = 1
        bc[i, ny - 1][3] = 1
    for i, j, k in ti.ndrange(nx, ny, 4):
        #print("bc[", i, ",", j, ",", k, "] = ", bc[i, j][k])
        pass


def fill_xu():
    for i, j in ti.ndrange(nx + 1, ny):
        xu[i * ny + j] = u[i, j]


def fill_Au():
    for i, j in ti.ndrange(nx, ny):
        k = i * ny + j
        if bc[i, j][0] == 1:
            Au[k, k] = 1.0
            bu[k] = u[i, j]
        elif bc[i, j][2] == 1:
            Au[k + ny, k + ny] = 1.0
            bu[k + ny] = u[i + 1, j]
        elif bc[i, j][1] == 1:
            Au[k, k - 1] = mu * dx / dy + max(
                [0, -rho * 0.5 * (v[i - 1, j] + v[i, j] * dx)])
            print("i=", i, "j= ", j, "k= ", k, Au[k, k - 1])


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
    fill_xu()
    fill_bc()
    fill_Au()
    post_u()
    post_v()

    display()