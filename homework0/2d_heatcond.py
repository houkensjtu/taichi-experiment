import taichi as ti
import numpy as np
import matplotlib.cm as cm

ti.init()


@ti.data_oriented
class heatcond:
    def __init__(self, nx, ny, iter):
        self.nx = nx
        self.ny = ny
        self.iter = iter
        self.k = 1.0
        self.cp = 1.0
        self.rho = 1.0
        self.dx = 1 / nx
        self.dy = 1 / ny
        self.dt = 0.01
        self.T = ti.var(dt=ti.f32, shape=(nx, ny))

    # Vector2, inside and inside_taichi are used to paint the Taichi logo
    # Borrowed from advection.py from Lecture4.
    @ti.func
    def Vector2(self, x, y):
        return ti.Vector([x, y])

    @ti.func
    def inside(self, p, c, r):
        return (p - c).norm_sqr() <= r * r

    @ti.func
    def inside_taichi(self, p):
        p = self.Vector2(0.5, 0.5) + (p - self.Vector2(0.5, 0.5)) * 1.3
        ret = -1
        if not self.inside(p, self.Vector2(0.50, 0.50), 0.55):
            if ret == -1:
                ret = 0
        if not self.inside(p, self.Vector2(0.50, 0.50), 0.50):
            if ret == -1:
                ret = 1
        if self.inside(p, self.Vector2(0.50, 0.25), 0.09):
            if ret == -1:
                ret = 1
        if self.inside(p, self.Vector2(0.50, 0.75), 0.09):
            if ret == -1:
                ret = 0
        if self.inside(p, self.Vector2(0.50, 0.25), 0.25):
            if ret == -1:
                ret = 0
        if self.inside(p, self.Vector2(0.50, 0.75), 0.25):
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
    def init(self):
        """
        Fill the temperature field with all 1s.
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            self.T[i, j] = self.inside_taichi(
                self.Vector2(i / self.nx, j / self.ny))

    @ti.kernel
    def temp_update(self):
        """
        Update the temperature field using 2d heat conduction equation.
        Explicit method is used for time integration.
        k : heat conduction
        cp : heat capacity
        rho : density
        dx, dy : grid size
        dt : time step
        """
        a_w = self.k / self.dx
        a_e = self.k / self.dx
        a_n = self.k / self.dy
        a_s = self.k / self.dy
        a_p = a_w + a_e + a_n + a_s + self.rho * self.cp * self.dx / self.dt
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.T[i,
                   j] = (a_w * self.T[i - 1, j] + a_e * self.T[i + 1, j] +
                         a_s * self.T[i, j - 1] + a_n * self.T[i, j + 1]) / a_p

    @ti.kernel
    def on_click(self, e: ti.template()):
        """
        Paint a dot with radius = 0.03 on the canvas.
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.inside(self.Vector2(i / self.nx, j / self.ny),
                           self.Vector2(e.pos[0], e.pos[1]), 0.03):
                self.T[i, j] = 1

    def solve(self):
        gui = ti.GUI('2D Heat conduction', (self.nx, self.ny))
        self.init()

        for i in range(self.iter):
            T_img = cm.terrain(self.T.to_numpy())
            gui.set_image(T_img)
            filename = f'frame_{i:05d}.png'
            gui.show(filename)
            for e in gui.get_events(ti.GUI.PRESS):
                if e.key == ti.GUI.LMB:
                    print("Clicked on: ", e.pos[0], e.pos[1])
                    self.on_click(e)
            self.temp_update()


def main():
    heat_cond = heatcond(320, 320, 1000)
    heat_cond.solve()


if __name__ == "__main__":
    main()
