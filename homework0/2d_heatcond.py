import taichi as ti
import numpy as np
import matplotlib.cm as cm

ti.init()

nx = 500
ny = 500

T = ti.var(dt=ti.f32, shape=(nx, ny))


@ti.func
def Vector2(x, y):
    return ti.Vector([x, y])


@ti.func
def inside(p, c, r):
    return (p - c).norm_sqr() <= r * r


@ti.func
def inside_taichi(p):
    p = Vector2(0.5, 0.5) + (p - Vector2(0.5, 0.5)) * 1.3
    ret = -1
    if not inside(p, Vector2(0.50, 0.50), 0.55):
        if ret == -1:
            ret = 0
    if not inside(p, Vector2(0.50, 0.50), 0.50):
        if ret == -1:
            ret = 1
    if inside(p, Vector2(0.50, 0.25), 0.09):
        if ret == -1:
            ret = 1
    if inside(p, Vector2(0.50, 0.75), 0.09):
        if ret == -1:
            ret = 0
    if inside(p, Vector2(0.50, 0.25), 0.25):
        if ret == -1:
            ret = 0
    if inside(p, Vector2(0.50, 0.75), 0.25):
        if ret == -1:
            ret = 1
    if p[0] < 0.5:
        if ret == -1:
            ret = 1
    else:
        if ret == -1:
            ret = 0
    return ret


@ti.kernel
def temp_fill():
    for i, j in ti.ndrange(nx, ny):
        T[i, j] = inside_taichi(Vector2(i / nx, j / ny))


@ti.kernel
def temp_print():
    for i, j in ti.ndrange(nx, ny):
        print(T[i, j])


@ti.kernel
def temp_update():
    for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
        T[i,
          j] = 0.2499 * (T[i - 1, j] + T[i + 1, j] + T[i, j - 1] + T[i, j + 1])


def main():
    gui = ti.GUI('2D heat conduction', (nx, ny))
    temp_fill()

    while True:
        T_img = cm.GnBu(T.to_numpy())
        gui.set_image(T_img)
        gui.show()
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.LMB:
                print("Clicked on: ", e.pos[0], e.pos[1])
                T[int(e.pos[0] * nx // 1), int(e.pos[1] * ny // 1)] += 1000
        temp_update()


if __name__ == "__main__":
    main()
