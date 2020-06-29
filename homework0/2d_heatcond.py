import taichi as ti
import numpy as np
import matplotlib.cm as cm

ti.init()

nx = 640
ny = 640

T = ti.var(dt=ti.f32, shape=(nx, ny))


@ti.kernel
def temp_fill():
    for i, j in ti.ndrange((1, nx), (1, ny)):
        T[i, j] = 1 / nx * i


@ti.kernel
def temp_disp():
    for i, j in ti.ndrange((1, nx), (1, ny)):
        print(T[i, j])


def main():
    gui = ti.GUI('2D heat conduction', (nx, ny))
    temp_fill()
    T_img = cm.coolwarm(T.to_numpy())

    #temp_disp()
    while True:
        gui.set_image(T_img)
        gui.show()


if __name__ == "__main__":
    main()
