import taichi as ti

ti.init(default_fp=ti.f64, arch=ti.cpu)

# Notes:
# >>> for i in ti.ndrange(5):
# ...     print(i)
# ...
# (0,)
# (1,)
# (2,)
# (3,)
# (4,)


# Problems:

# fill_Au
# 1. Inlet and outlet setting.
# 2. Upper and lower wall boundary setting.
# 3. Investigate the property of Au, is it symmetry? Is it positive definite?
# 4. Depends on dt, which is quicker? CG or Jacobian?
# 5. When boundary is implemented as simple 1 = u, what's the results on A's property?
#    Does it affect p correction equation?


lx = 1.0
ly = 0.1

nx = 200
ny = 60

rho = 1
mu = 0.01
dx = lx / nx
dy = ly / ny
dt = 20000

# Relaxation factors
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
v0 = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 3))
vcor = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 3))
v_post = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 2))

# ct stands for Cell Type.
# ct = 0 -> Fluid
# ct = 1 -> Solid
ct = ti.field(dtype=ti.i32, shape=(nx + 2, ny + 2))

Au = ti.field(dtype=ti.f64, shape=((nx + 1) * ny, (nx + 1) * ny))
bu = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))
xu = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))
xu_new = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))
xuold = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))

# for solving u momentum using CG
Auxu = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))
ru = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))
pu = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))
Aupu = ti.field(dtype=ti.f64, shape=((nx + 1) * ny))
# def conjgrad(A, x, Ax, r, p, Ap):


Av = ti.field(dtype=ti.f64, shape=(nx * (ny + 1), nx * (ny + 1)))
bv = ti.field(dtype=ti.f64, shape=(nx * (ny + 1)))
xv = ti.field(dtype=ti.f64, shape=(nx * (ny + 1)))
xv_new = ti.field(dtype=ti.f64, shape=(nx * (ny + 1)))
xvold = ti.field(dtype=ti.f64, shape=(nx * (ny + 1)))

Ap = ti.field(dtype=ti.f64, shape=(nx * ny, nx * ny))
bp = ti.field(dtype=ti.f64, shape=(nx * ny))
xp = ti.field(dtype=ti.f64, shape=(nx * ny))


@ti.kernel
def init():
    for i, j in ti.ndrange(nx + 2, ny + 2):
        p[i, j] = 100 - 12 * i / nx
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


@ti.kernel
def fill_Au():
    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)

        # Inlet and Outlet
        # ct[i-1,j] is the left cell of u[i,j]
        # ct[i,j] + ct[i-1,j] = 2 means the u is inside a block
        if (ct[i - 1, j]) == 1 or (ct[i, j] + ct[i - 1, j]) == 2:
            Au[k, k] = 1.0
            bu[k] = u[i, j]

        # Outlet
        # ct[i,j] is the right cell of u[i,j]
        elif (ct[i, j] == 1):
            Au[k, k] = 1.0
            Au[k, k - ny] = -1.0
            # bu[k] = u[i - 1, j]
            # bu[k] = u[i, j]
            bu[k] = 0.0

        # Normal internal cells
        else:
            Au[k, k - 1] = -mu * dx / dy - \
                ti.max(0, -rho * 0.5 * (v[i - 1, j] + v[i, j]) * dx)  # an
            Au[k, k + 1] = -mu * dx / dy - \
                ti.max(0, rho * 0.5 *
                       (v[i - 1, j + 1] + v[i, j + 1]) * dx)  # as
            Au[k, k - ny] = -mu * dy / dx - \
                ti.max(0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy)  # aw
            Au[k, k + ny] = -mu * dy / dx - \
                ti.max(0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy)  # ae
            Au[k, k] = -Au[k, k - 1] - Au[k, k + 1] - Au[k, k - ny] - \
                Au[k, k + ny] + rho * dx * dy / dt  # ap
            bu[k] = (p[i - 1, j] - p[i, j]) * dy + rho * dx * \
                dy / dt * u0[i, j]  # <= Unsteady term

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
            Av[k, k - 1] = -mu * dx / dy - \
                ti.max(0, -rho * 0.5 * (v[i, j - 1] + v[i, j]) * dx)  # an
            Av[k, k + 1] = -mu * dx / dy - \
                ti.max(0, rho * 0.5 * (v[i, j + 1] + v[i, j]) * dx)  # as
            Av[k, k - ny - 1] = -mu * dy / dx - \
                ti.max(0, rho * 0.5 * (u[i, j] + u[i, j - 1]) * dy)  # aw
            Av[k, k + ny + 1] = -mu * dy / dx - \
                ti.max(0, -rho * 0.5 *
                       (u[i + 1, j - 1] + u[i + 1, j]) * dy)  # ae
            Av[k, k] = -Av[k, k - 1] - Av[k, k + 1] - Av[k, k - ny - 1] - \
                Av[k, k + ny + 1] + rho * dx * dy / dt  # ap
            bv[k] = (p[i, j] - p[i, j - 1]) * dx + \
                rho * dx * dy / dt * v0[i, j]


@ti.kernel
def full_jacobian(A: ti.template(), b: ti.template(), x: ti.template(), x_new: ti.template()) -> ti.f64:
    for i in range(x.shape[0]):
        r = b[i]
        for j in range(x.shape[0]):
            if i != j:
                r -= A[i, j] * x[j]
        x_new[i] = r / A[i, i]
        x[i] = r / A[i, i]
    for i in range(x.shape[0]):
        x[i] = x_new[i]

    res = 0.0

    for i in range(x.shape[0]):
        r = b[i] * 1.0
        for j in range(x.shape[0]):
            r -= A[i, j] * x[j]
        res += r * r
    return ti.sqrt(res)


@ti.kernel
def quick_jacobian(A: ti.template(), b: ti.template(), x: ti.template(), x_new: ti.template()) -> ti.f64:
    for i in range(x.shape[0]):
        r = b[i]
        for j in range(i-ny-1, i+ny+2):
            if i != j:
                r -= A[i, j] * x[j]
        x_new[i] = r / A[i, i]
        x[i] = r / A[i, i]
    for i in range(x.shape[0]):
        x[i] = x_new[i]

    res = 0.0

    for i in range(x.shape[0]):
        r = b[i] * 1.0
        for j in range(i-ny-1, i+ny+2):
            r -= A[i, j] * x[j]
        res += r * r
    return ti.sqrt(res)


@ti.kernel
def conjgrad(A: ti.template(), x: ti.template(), b: ti.template(), Ax: ti.template(), r: ti.template(), p: ti.template(), Ap: ti.template()):
    # dot(A,x)
    for i in range(x.shape[0]):
        Ax[i] = 0.0
        for j in range(i-ny-1, i+ny+2):
            Ax[i] += A[i, j] * x[j]
    # r = b - dot(A,x)
    # p = r
    for i in range(x.shape[0]):
        r[i] = b[i] - Ax[i]
        p[i] = r[i]
    rsold = 0.0
    for i in range(x.shape[0]):
        rsold += r[i] * r[i]

    for steps in range(x.shape[0]):
        # dot(A,p)
        for i in range(x.shape[0]):
            Ap[i] = 0.0
            for j in range(i-ny-1, i+ny+2):
                Ap[i] += A[i, j] * p[j]

        # dot(p, Ap) => pAp
        pAp = 0.0
        for i in range(x.shape[0]):
            pAp += p[i] * Ap[i]

        alpha = rsold / pAp

        # x = x + dot(alpha,p)
        # r = r - dot(alpha,Ap)
        for i in range(x.shape[0]):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]

        rsnew = 0.0
        for i in range(x.shape[0]):
            rsnew += r[i] * r[i]

        if ti.sqrt(rsnew) < 1e-6:
            print("The solution has converged...")
            break

        for i in range(x.shape[0]):
            p[i] = r[i] + (rsnew / rsold) * p[i]
        rsold = rsnew
        print(steps, rsold)


@ti.kernel
def xu_back():
    for i, j in ti.ndrange(nx + 1, ny):
        u[i + 1, j + 1] = xu[i * ny + j]


@ti.kernel
def xv_back():
    for i, j in ti.ndrange(nx, ny + 1):
        v[i + 1, j + 1] = xv[i * ny + j]


@ti.kernel
def proc_u():
    for i, j in u:
        u[i, j] = u[i, j] / 2.0


if __name__ == "__main__":
    init()
    fill_Au()
    fill_Av()

    residual_x = 10.0
    # conjgrad(Au, xu, bu, Auxu, ru, pu, Aupu)
    while residual_x > 1e-8:
        print("Residual x = ", residual_x)
        residual_x = quick_jacobian(Au, bu, xu, xu_new)

    residual_y = 10.0
    while residual_y > 1e-8:
        print("Residual y = ", residual_y)
        residual_y = quick_jacobian(Av, bv, xv, xv_new)

    xu_back()
    xv_back()

    for j in range(ny+2):
        print("i = ", nx+1, ", j = ", j, ", u = ", u[nx+1, j])
    for j in range(ny+2):
        print("i = ", 1, ", j = ", j, ", u = ", u[1, j])


#    for j in range(ny+2):
#        print("i = 201, j = ",j, "v = ", v[nx,j])
#    for j in range(ny+2):
#        print("i = 1, j = ",j, "v = ", v[1,j])
