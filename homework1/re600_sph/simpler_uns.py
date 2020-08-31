import taichi as ti
import numpy as np
import matplotlib.cm as cm

ti.init()

lx = 1.5
ly = 0.3

nx = 60
ny = 20

velo_rel = 0.01
p_rel = 0.03

# Add 1 cell padding to all directions.
p = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))
pcor = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))

u = ti.var(dt=ti.f32, shape=(nx + 3, ny + 2))
u0 = ti.var(dt=ti.f32, shape=(nx + 3, ny + 2))
ucor = ti.var(dt=ti.f32, shape=(nx + 3, ny + 2))
u_post = ti.var(dt=ti.f32, shape=(nx + 2, ny + 2))

v = ti.var(dt=ti.f32, shape=(nx + 2, ny + 3))
vcor = ti.var(dt=ti.f32, shape=(nx + 2, ny + 3))
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
dt = 0.001

Au = ti.var(dt=ti.f32, shape=((nx + 1) * ny, (nx + 1) * ny))
bu = ti.var(dt=ti.f32, shape=((nx + 1) * ny))
xu = ti.var(dt=ti.f32, shape=((nx + 1) * ny))
xuold = ti.var(dt=ti.f32, shape=((nx + 1) * ny))

Av = ti.var(dt=ti.f32, shape=(nx * (ny + 1), nx * (ny + 1)))
bv = ti.var(dt=ti.f32, shape=(nx * (ny + 1)))
xv = ti.var(dt=ti.f32, shape=(nx * (ny + 1)))
xvold = ti.var(dt=ti.f32, shape=(nx * (ny + 1)))

Ap = ti.var(dt=ti.f32, shape=(nx * ny, nx * ny))
bp = ti.var(dt=ti.f32, shape=(nx * ny))
xp = ti.var(dt=ti.f32, shape=(nx * ny))


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
    from scipy.sparse.linalg import qmr, bicg
    from scipy.sparse import csc_matrix
    print("Now converting A and b to numpy...")
    A_np = A.to_numpy()
    b_np = b.to_numpy()
    print("Finished converting A and b to numpy...")
    print("Now solving Ax=b...")
    ans = np.linalg.solve(A_np, b_np)
    print("Finished solving Ax=b...")
    return ans
    #ans, exitCode = bicg(A_np, b_np, atol='legacy', tol=1e-3)
    # return ans


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


def iter_solve_u():
    #A = Au.to_numpy()
    #b = bu.to_numpy()
    res = 100.0

    while np.abs(res) > 1e-3:
        res = 0.0
        for i, j in ti.ndrange(nx + 1, ny):
            k = i * ny + j
            # print("k = ", k, "ny = ", ny, "k-ny = ", k - ny, "Au[k-ny] = ",
            # Au[k - ny])
            xu[k] = 1 / Au[k, k] * (-Au[k, k - 1] * u[i, j - 1] -
                                    Au[k, k + 1] * u[i, j + 1] -
                                    Au[k, k - ny] * u[i - 1, j] -
                                    Au[k, k + ny] * u[i + 1, j] + bu[k])

            res = res + xu[k] - xuold[k]
        xu_back()
        for i, j in ti.ndrange(nx + 1, ny):
            k = i * ny + j
            xuold[k] = xu[k]
        print("Solving x momentum, the residul is now ", res)


def iter_solve_v():
    #A = Au.to_numpy()
    #b = bu.to_numpy()
    res = 100.0

    while np.abs(res) > 1e-3:
        res = 0.0
        for i, j in ti.ndrange(nx, ny + 1):
            k = i * (ny + 1) + j
            # print("k = ", k, "ny = ", ny, "k-ny = ", k - ny, "Au[k-ny] = ",
            # Au[k - ny])
            xv[k] = 1 / Av[k, k] * (-Av[k, k - 1] * v[i, j - 1] -
                                    Av[k, k + 1] * v[i, j + 1] -
                                    Av[k, k - ny - 1] * v[i - 1, j] -
                                    Av[k, k + ny + 1] * v[i + 1, j] + bv[k])

            res = res + xv[k] - xvold[k]
        xv_back()
        for i, j in ti.ndrange(nx, ny + 1):
            k = i * (ny + 1) + j
            xvold[k] = xv[k]
        print("Solving y momentum, the residual is now ", res)


def solve_moment_x():
    print("Now filling Au...")
    fill_Au()
    print("Finished filling Au...")
    
    print("Solving x momentum...")
    # solve_axb returns a numpy array
    # needs to convert back to taichi
    #import numpy.linalg as npl
    # print("Shape of Au is", Au.shape(), "Rank of Au is:",
    #      npl.matrix_rank(Au.to_numpy()))
    xu.from_numpy(solve_axb(Au, bu))
    # iter_solve_u()
    print("Converting xu to u...")
    sol_back_matrix(u, xu)
    print("Finished converting xu to u...")

def solve_moment_y():
    fill_Av()
    print("Solving y momentum...")
    #import numpy.linalg as npl
    # print("Shape of Av is", Av.shape(), "Rank of Av is:",
    #      npl.matrix_rank(Av.to_numpy()))
    xv.from_numpy(solve_axb(Av, bv))
    # iter_solve_v()
    sol_back_matrix(v, xv)


def correct_u():
    ucor_max = 0.0
    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        # Upper and lower boundary
        if (ct[i - 1, j] + ct[i, j]) == 0 or (ct[i - 1, j] + ct[i, j]) == 2:
            pass
        else:
            ucor[i, j] = (pcor[i - 1, j] - pcor[i, j]) * dy / Au[k, k]
            u[i, j] = u[i, j] + ucor[i, j] * velo_rel
            if np.abs(ucor[i, j] / (u[i, j] + 1.0e-9)) >= ucor_max:
                ucor_max = np.abs(ucor[i, j] / (u[i, j] + 1.0e-9))
    return ucor_max


def correct_v():
    vcor_max = 0.0
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 2)):
        k = (i - 1) * (ny + 1) + (j - 1)
        # Upper and lower boundary
        if (ct[i, j] + ct[i, j - 1]) == 0 or (ct[i, j] + ct[i, j - 1]) == 2:
            pass
        else:
            vcor[i, j] = (pcor[i, j] - pcor[i, j - 1]) * dx / Av[k, k]
            v[i, j] = v[i, j] + vcor[i, j] * velo_rel
            if np.abs(vcor[i, j] / (v[i, j] + 1.0e-9)) >= vcor_max:
                vcor_max = np.abs(vcor[i, j] / (v[i, j] + 1.0e-9))
    return vcor_max


def correct_uconserv():
    inlet_flux = 0.0
    outlet_flux = 0.0
    for i in range(1, ny + 1):
        inlet_flux = inlet_flux + u[1, i]
        outlet_flux = outlet_flux + u[nx + 1, i]
    print("Inlet flux = ", inlet_flux, "; Outlet flux = ", outlet_flux)

    coef = inlet_flux / (outlet_flux + 1.0e-9)
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
    #import numpy.linalg as npl
    # print("Shape of Ap is", Ap.shape(), "Rank of Ap is:",
    #      npl.matrix_rank(Ap.to_numpy()))
    sumbp = 0.0
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        sumbp = sumbp + bp[k]
    print("Sum bp before solving pcorr was", sumbp)

    print("Now solving pcor...")
    xp.from_numpy(solve_axb(Ap, bp))
    sol_back_matrix(pcor, xp)

    for i, j in ti.ndrange(nx + 2, ny + 2):
        if ct[i, j] == 1:
            pass
        else:
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
    fig, ax = plt.subplots(2, 6)

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

    pcm = ax[0, 4].pcolormesh(ucor.to_numpy(), cmap=cm.rainbow)
    ax[0, 4].set_title("u correction")
    fig.colorbar(pcm, ax=ax[0, 4])

    pcm = ax[0, 5].pcolormesh(vcor.to_numpy(), cmap=cm.rainbow)
    ax[0, 5].set_title("v correction")
    fig.colorbar(pcm, ax=ax[0, 5])

    ax[1, 0].plot(p.to_numpy()[1:int(nx + 1), int(ny / 2)])
    ax[1, 0].set_title("pressure drop")

    ax[1, 1].plot(u.to_numpy()[int(0.2 * nx), 1:int(ny + 1)])
    ax[1, 1].set_title("U profile at 60")

    ax[1, 2].plot(u.to_numpy()[int(0.5 * nx), 1:int(ny + 1)])
    ax[1, 2].set_title("U profile at 100")

    ax[1, 3].plot(u.to_numpy()[int(0.8 * nx), 1:int(ny + 1)])
    ax[1, 3].set_title("U profile at 120")

    fig.set_size_inches(16, 9)
    fig.tight_layout()

    plt.savefig("Iteration_i" + str(iter) + "_t" + str(jter) + ".png", dpi=400)


if __name__ == "__main__":
    init()

    check_uconserv()
    for jter in range(1000):
        print("Solving the next time step, currently the ", jter, "th iteration...")
        for iter in range(10000):
            print("Looping through the inner loop, it's the ", iter, "th iteration out of 10...")
            solve_moment_x()
            solve_moment_y()
            correct_uconserv()
            check_uconserv()
            solve_pcor()
            resu = correct_u()
            resv = correct_v()
            print("Resu = ", resu, "Resv = ", resv)
        u0 = u
        v0 = v
        display()
