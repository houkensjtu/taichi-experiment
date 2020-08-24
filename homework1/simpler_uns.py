import taichi as ti
import numpy as np
import matplotlib.cm as cm

ti.init()

lx = 1.0
ly = 0.3

nx = 120
ny = 30

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


def init():
    for i, j in ti.ndrange(nx + 2, ny + 2):
        p[i, j] = 100 - i / nx
        mdiv[i, j] = 0.0
    for i, j in ti.ndrange(nx + 3, ny + 2):
        u[i, j] = 1.0
        u0[i, j] = u[i, j]
    for i, j in ti.ndrange(nx + 2, ny + 3):
        v[i, j] = 0.0
        v0[i, j] = v[i, j]

    for i, j in ti.ndrange(nx + 2, ny + 2):
        ct[i, j] = 1  # "1" stands for solid
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 1)):
        ct[i, j] = -1  # "-1" stands for fluid

    for i, j in ti.ndrange((35, 40), (14, 18)):
        ct[i, j] = 1
        u[i, j] = 0
        u0[i, j] = 0
        v[i, j] = 0
        v0[i, j] = 0
        pass


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
                     ) * dy + rho * dx * dy / dt * u0[i, j]  # <= Unsteady term

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
            bv[k] = (p[i, j] - p[i, j - 1]) * dx + rho * dx * dy / dt * v0[i,
                                                                           j]


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
    import numpy.linalg as npl
    print("Shape of Au is", Au.shape(), "Rank of Au is:",
          npl.matrix_rank(Au.to_numpy()))
    xu.from_numpy(solve_axb(Au, bu))
    sol_back_matrix(u, xu)


def solve_moment_y():
    fill_Av()
    import numpy.linalg as npl
    print("Shape of Av is", Av.shape(), "Rank of Av is:",
          npl.matrix_rank(Av.to_numpy()))
    xv.from_numpy(solve_axb(Av, bv))
    sol_back_matrix(v, xv)


def correct_u():
    ucor_max = 0.0
    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        # Upper and lower boundary
        if (ct[i - 1, j] + ct[i, j]) == 0 or (ct[i - 1, j] + ct[i, j]) == 2:
            pass
        else:
            ucor = (pcor[i - 1, j] - pcor[i, j]) * dy / Au[k, k]
            u[i, j] = u[i, j] + ucor * velo_rel
            if np.abs(ucor / (u[i, j] + 1.0e-9)) >= ucor_max:
                ucor_max = np.abs(ucor / (u[i, j] + 1.0e-9))
    return ucor_max


def correct_v():
    vcor_max = 0.0
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 2)):
        k = (i - 1) * (ny + 1) + (j - 1)
        # Upper and lower boundary
        if (ct[i, j] + ct[i, j - 1]) == 0 or (ct[i, j] + ct[i, j - 1]) == 2:
            pass
        else:
            vcor = (pcor[i, j] - pcor[i, j - 1]) * dx / Av[k, k]
            v[i, j] = v[i, j] + vcor * velo_rel
            if np.abs(vcor / (v[i, j] + 1.0e-9)) >= vcor_max:
                vcor_max = np.abs(vcor / (v[i, j] + 1.0e-9))
    return vcor_max


def correct_uconserv():
    inlet_flux = 0.0
    outlet_flux = 0.0
    for i in range(1, ny + 1):
        inlet_flux = inlet_flux + u[1, i]
        outlet_flux = outlet_flux + u[nx + 1, i]
    print("Inlet flux = ", inlet_flux, "; Outlet flux = ", outlet_flux)

    coef = inlet_flux / outlet_flux
    for i in range(1, ny + 1):
        u[nx + 1, i] = coef * u[nx + 1, i]


def check_uconserv():
    inlet_flux = 0.0
    outlet_flux = 0.0
    for i in range(1, ny + 1):
        inlet_flux = inlet_flux + u[1, i]
        outlet_flux = outlet_flux + u[nx + 1, i]
    print("Inlet flux = ", inlet_flux, "; Outlet flux = ", outlet_flux)


def fill_Ap():
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        bp[k] = rho * (u[i, j] - u[i + 1, j]) * dy + rho * (v[i, j + 1] -
                                                            v[i, j]) * dx
        # Go back to Av matrix, find the corresponding v
        vk = (i - 1) * (ny + 1) + (j - 1)
        Ap[k, k - 1] = -rho * dx * dx / Av[vk, vk]
        Ap[k, k + 1] = -rho * dx * dx / Av[vk + 1, vk + 1]
        # Go back to Au matrix
        uk = k
        Ap[k, k - ny] = -rho * dy * dy / Au[uk, uk]
        Ap[k, k + ny] = -rho * dy * dy / Au[uk + ny, uk + ny]

        if (ct[i, j] + ct[i, j - 1]) == 0:
            Ap[k, k - 1] = 0
        elif (ct[i, j] + ct[i, j + 1]) == 0:
            Ap[k, k + 1] = 0
        elif (ct[i, j] + ct[i - 1, j]) == 0:
            Ap[k, k - ny] = 0
        elif (ct[i, j] + ct[i + 1, j]) == 0:
            Ap[k, k + ny] = 0
        Ap[k, k] = -Ap[k, k - 1] - Ap[k, k + 1] - Ap[k, k - ny] - Ap[k, k + ny]


def solve_pcor():
    fill_Ap()
    import numpy.linalg as npl
    print("Shape of Ap is", Ap.shape(), "Rank of Ap is:",
          npl.matrix_rank(Ap.to_numpy()))
    sumbp = 0.0
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        sumbp = sumbp + bp[k]
    print("Sum bp before solving pcorr was", sumbp)

    print("Now solving pcor...")
    xp.from_numpy(solve_axb(Ap, bp))
    sol_back_matrix(pcor, xp)

    for i, j in ti.ndrange(nx + 2, ny + 2):
        p[i, j] = p[i, j] + p_rel * pcor[i, j]


def visual(mat):
    A = mat.to_numpy()
    import matplotlib.pyplot as plt
    # 'nearest' interpolation - faithful but blocky
    plt.imshow(A, interpolation='nearest', cmap=cm.rainbow)
    # plt.colorbar()
    # plt.show()
    plt.savefig("karmen" + str(iter) + ".png", dpi=300)


def display():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4)

    pcm = ax[0, 0].pcolormesh(u.to_numpy(), cmap=cm.rainbow)
    ax[0, 0].set_title("U velocity")
    fig.colorbar(pcm, ax=ax[0, 0])

    pcm = ax[0, 1].pcolormesh(v.to_numpy(), cmap=cm.rainbow)
    ax[0, 1].set_title("V velocity")
    fig.colorbar(pcm, ax=ax[0, 1])

    pcm = ax[0, 2].pcolormesh(p.to_numpy(), cmap=cm.rainbow)
    ax[0, 2].set_title("Pressure")
    fig.colorbar(pcm, ax=ax[0, 2])

    pcm = ax[0, 3].pcolormesh(pcor.to_numpy(), cmap=cm.rainbow)
    ax[0, 3].set_title("p correction")
    fig.colorbar(pcm, ax=ax[0, 3])

    ax[1, 0].plot(p.to_numpy()[1:int(nx + 1), int(ny / 2)])
    ax[1, 0].set_title("pressure drop")

    ax[1, 1].plot(u.to_numpy()[int(0.2 * nx), 1:int(ny + 1)])
    ax[1, 1].set_title("U profile at 60")

    ax[1, 2].plot(u.to_numpy()[int(0.5 * nx), 1:int(ny + 1)])
    ax[1, 2].set_title("U profile at 100")

    ax[1, 3].plot(u.to_numpy()[int(0.8 * nx), 1:int(ny + 1)])
    ax[1, 3].set_title("U profile at 120")

    fig.set_size_inches(11, 8.5)
    fig.tight_layout()

    plt.savefig("Iteration_i" + str(iter) + "_t" + str(jter) + ".png", dpi=400)


if __name__ == "__main__":
    init()

    check_uconserv()
    for jter in range(1000):
        print("Solving the outer loop", jter, "th iteration...")
        for iter in range(10):
            print("Solving the inner loop", iter, "th iteration...")
            solve_moment_x()
            solve_moment_y()
            correct_uconserv()
            check_uconserv()
            solve_pcor()
            resu = correct_u()
            resv = correct_v()
        u0 = u
        v0 = v
        display()
    #u_img = cm.terrain(u.to_numpy())
    # gui.set_image(u_img)
    #filename = f'frame_{iter:05d}.png'
    # gui.show(filename)
