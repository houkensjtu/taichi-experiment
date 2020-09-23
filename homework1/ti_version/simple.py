import taichi as ti
import time
import numpy as np

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
# => For inlet, Au[k,k] = 1 and bu[k] = inlet velocity. Will not affect p correction.
# => For outlet, Au[k,k] = Au[k-ny,k-ny], and Au[k,k-ny] = -Au[k,k].

# 2. Upper and lower wall boundary setting.
# => For upper bound, an = 0 and ap += 2*mu*dx/dy

# 3. Investigate the property of Au, is it symmetry? Is it positive definite?
# => Au is not symmetry, but it is full rank and all eig values are positive.

# 4. Depends on dt, which is quicker? CG or Jacobian?

# 5. When inlet boundary is implemented as simple Au[i,i] = 1, what's the results on A's property?
#    Does it affect p correction equation?

# fill_Av
# 1. Accessing out of bound elements is causing unexpected write-in in Av.
# => Make sure all access (k-1, k-ny-1, k+1, k+ny+1) are within boundary.

# Problems almost solved; Confirmed that the solution is correct for 2D plane hagen-posiule flow.
# Next step will be implementing the BiCGSTAB solver to replace Jacobian.
# One remaining issue: In quick Jacobian, solver is accessing elements out of bounds.

# Linear solver
# 1. Normal CG will diverge on u momentum eqn. because it's not symmetric.
# 2. BiCG can converge on u momentum eqn. but the converge is slow and spiky.

# 3. On a 128 x 64 field (eps = 1e-5), BiCG (with M) takes 77sec, Jacobian takes 57 sec to converge.
#    So BiCG (with diag(A) as preconditioner) is no better than Jacobian.
#    BiCG (with no preconditioner) took 19 sec and is much better than Jacobian.

#    For BiCGSTAB with M=diag(A), it took about 7 sec (10x faster than BiCG!)
#    And with M=1.0, it took about 12 sec, showing different behavior than BiCG.
#    Generally, the convergence is much much smoother than BiCG.

# 4. For the v momentum, bicg will fail when all v are zero because the initial rho = 0.

# GUI Display
# 1. Can only write a ti.GUI in main routine, cannot write as a separate function. Don't know why yet.
# 2. u and v are interpolated into u_post and v_post, and then shown in gui.
# 3. Image size needs to adjusted so that show actual aspect ratio lx/ly.

# When lx and ly changed, the pressure drop also needs to be changed to reproduce the hagen flow cond.
lx = 0.5
ly = 0.1

nx = 275
ny = 55

rho = 1
mu = 0.01
dx = lx / nx
dy = ly / ny
dt = 10000000

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
u_disp = ti.field(dtype=ti.f64, shape=(3 *(nx + 2), 3*(ny + 2)))

v = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 3))
v0 = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 3))
vcor = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 3))
v_post = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 2))
v_disp = ti.field(dtype=ti.f64, shape=(3*(nx + 2), 3*(ny + 2)))

# ct stands for Cell Type.
# ct = 0 -> Fluid
# ct = 1 -> Solid
ct = ti.field(dtype=ti.i32, shape=(nx + 2, ny + 2))

# for solving u momentum using Jacobian
Au = ti.field(dtype=ti.f64, shape=((nx + 1) * ny, (nx + 1) * ny))
Mu = ti.field(dtype=ti.f64, shape=((nx + 1) * ny, (nx + 1) * ny))
bu = ti.field(dtype=ti.f64)
xu = ti.field(dtype=ti.f64)
xu_new = ti.field(dtype=ti.f64)
xuold = ti.field(dtype=ti.f64)
ti.root.dense(ti.i, (nx+1)*ny).place(bu, xu, xu_new, xuold)
# Additional vectors for BiCG
Auxu = ti.field(dtype=ti.f64)
Aupu = ti.field(dtype=ti.f64)
Aupu_tld = ti.field(dtype=ti.f64)
ru = ti.field(dtype=ti.f64)
pu = ti.field(dtype=ti.f64)
zu = ti.field(dtype=ti.f64)
ru_tld = ti.field(dtype=ti.f64)
pu_tld = ti.field(dtype=ti.f64)
zu_tld = ti.field(dtype=ti.f64)
ti.root.dense(ti.i, (nx+1)*ny).place(Auxu, Aupu, Aupu_tld)
ti.root.dense(ti.i, (nx+1)*ny).place(ru, pu, zu, ru_tld, pu_tld, zu_tld)
# More additional vectors for BiCGSTAB
pu_hat = ti.field(dtype=ti.f64)
su = ti.field(dtype=ti.f64)
su_hat = ti.field(dtype=ti.f64)
tu = ti.field(dtype=ti.f64)
ti.root.dense(ti.i, (nx+1)*ny).place(pu_hat, su, su_hat, tu)

# for solving v momentum using Jacobian
Av = ti.field(dtype=ti.f64, shape=(nx * (ny + 1), nx * (ny + 1)))
Mv = ti.field(dtype=ti.f64, shape=(nx * (ny + 1), nx * (ny + 1)))
bv = ti.field(dtype=ti.f64)
xv = ti.field(dtype=ti.f64)
xv_new = ti.field(dtype=ti.f64)
xvold = ti.field(dtype=ti.f64)
ti.root.dense(ti.i, nx*(ny+1)).place(bv, xv, xv_new, xvold)
# Additional vectors for BiCG
Avxv = ti.field(dtype=ti.f64)
Avpv = ti.field(dtype=ti.f64)
Avpv_tld = ti.field(dtype=ti.f64)
rv = ti.field(dtype=ti.f64)
pv = ti.field(dtype=ti.f64)
zv = ti.field(dtype=ti.f64)
rv_tld = ti.field(dtype=ti.f64)
pv_tld = ti.field(dtype=ti.f64)
zv_tld = ti.field(dtype=ti.f64)
ti.root.dense(ti.i, nx*(ny+1)).place(Avxv, Avpv, Avpv_tld)
ti.root.dense(ti.i, nx*(ny+1)).place(rv, pv, zv, rv_tld, pv_tld, zv_tld)
# More additional vectors for BiCGSTAB
pv_hat = ti.field(dtype=ti.f64)
sv = ti.field(dtype=ti.f64)
sv_hat = ti.field(dtype=ti.f64)
tv = ti.field(dtype=ti.f64)
ti.root.dense(ti.i, nx*(ny+1)).place(pv_hat, sv, sv_hat, tv)


# For pressure correction equation.
Ap = ti.field(dtype=ti.f64, shape=(nx * ny, nx * ny))
bp = ti.field(dtype=ti.f64, shape=(nx * ny))
xp = ti.field(dtype=ti.f64, shape=(nx * ny))


@ti.kernel
def init():
    for i, j in ti.ndrange(nx + 2, ny + 2):
        p[i, j] = 100 - 6.0 * i / nx
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


def write_matrix(mat, name):
    np.savetxt(name + ".csv", mat.to_numpy(), delimiter = ",")

def visual_matrix(mat, name):
    a = mat.to_numpy()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # 'nearest' interpolation - faithful but blocky
    plt.imshow(a, interpolation='nearest', cmap=cm.rainbow)
    plt.colorbar()
    plt.show()    

@ti.kernel
def fill_Au():
    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        # Inlet
        # ct[i-1,j] is the left cell of u[i,j]
        # ct[i,j] + ct[i-1,j] = 2 means the u is inside a block
        if (ct[i - 1, j]) == 1 or (ct[i, j] + ct[i - 1, j]) == 2:
            Au[k, k] = 1.0
            bu[k] = u[i, j]
        # Outlet
        # ct[i,j] is the right cell of u[i,j]
        elif (ct[i, j] == 1):
            Au[k, k] = 1.0  # Au[k-ny,k-ny]
            Au[k, k - ny] = -1.0  # -Au[k,k]
            bu[k] = 0.0
        else:
            # Normal internal cells
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

        # Upper and lower boundary
        # Excluded the inlet and outlet
        if (ct[i, j] + ct[i, j - 1]) == 0 and ct[i, j] != 1 and ct[i-1, j] != 1:
            # Be careful it should be + Au[k,k-1] because it is minus.
            # Also, notice that 2*mu should be followed by dx/dy.
            Au[k, k] = Au[k, k] + Au[k, k - 1] + 2 * mu * dx/dy
            Au[k, k - 1] = 0
            # For second order wall visc: bu[k] += (p[i-1,j] - p[i,j])*rho/mu*dy/4
        elif (ct[i, j] + ct[i, j + 1]) == 0 and ct[i, j] != 1 and ct[i-1, j] != 1:
            Au[k, k] = Au[k, k] + Au[k, k + 1] + 2 * mu * dx/dy
            Au[k, k + 1] = 0
    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        Mu[k,k] = Au[k,k]
        # Mu[k,k] = 1.0




@ti.kernel
def fill_Av():
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 2)):
        k = (i - 1) * (ny + 1) + (j - 1)
        # Upper and lower boundary
        if (ct[i, j] + ct[i, j - 1]) == 0 or (ct[i, j] + ct[i, j - 1]) == 2:
            Av[k, k] = 1.0
            bv[k] = v[i, j]
        # Inlet: do not access west cell A[k,k-ny-1], treat as a wall boundary
        elif (ct[i, j]+ct[i-1, j]) == 0:
            Av[k, k - 1] = -mu * dx / dy - \
                ti.max(0, -rho * 0.5 * (v[i, j - 1] + v[i, j]) * dx)  # an
            Av[k, k + 1] = -mu * dx / dy - \
                ti.max(0, rho * 0.5 * (v[i, j + 1] + v[i, j]) * dx)  # as
            Av[k, k + ny + 1] = -mu * dy / dx - \
                ti.max(0, -rho * 0.5 *
                       (u[i + 1, j - 1] + u[i + 1, j]) * dy)  # ae
            Av[k, k] = -Av[k, k - 1] - Av[k, k + 1] - \
                Av[k, k + ny + 1] + rho * dx * dy / dt + 2*mu*dy/dx  # ap
            bv[k] = (p[i, j] - p[i, j - 1]) * dx + \
                rho * dx * dy / dt * v0[i, j]
        # Outlet: do not access east cell, treat as a wall boundary
        elif (ct[i, j] + ct[i+1, j]) == 0:
            Av[k, k - 1] = -mu * dx / dy - \
                ti.max(0, -rho * 0.5 * (v[i, j - 1] + v[i, j]) * dx)  # an
            Av[k, k + 1] = -mu * dx / dy - \
                ti.max(0, rho * 0.5 * (v[i, j + 1] + v[i, j]) * dx)  # as
            Av[k, k - ny - 1] = -mu * dy / dx - \
                ti.max(0, rho * 0.5 * (u[i, j] + u[i, j - 1]) * dy)  # aw
            Av[k, k] = -Av[k, k - 1] - Av[k, k + 1] - Av[k, k - ny - 1] \
                + rho * dx * dy / dt + 2*mu*dy/dx  # ap
            bv[k] = (p[i, j] - p[i, j - 1]) * dx + \
                rho * dx * dy / dt * v0[i, j]
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
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 2)):
        k = (i - 1) * (ny + 1) + (j - 1)
        Mv[k,k] = Av[k,k]


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



# Quick version of conjugate gradient
# Only multiply non-zero elements in A
# Other calculations are exactly same

@ti.kernel
def struct_cg(
         A: ti.template(),
         b: ti.template(),
         x: ti.template(),
         Ax: ti.template(),
         Ap: ti.template(),
         r: ti.template(),
         p: ti.template(),
         nx: ti.i32,
         ny: ti.i32):
    n = (nx+1) * ny
    # dot(A,x)
    for i in range(n):
        Ax[i] = 0.0
        for j in range(i-ny, i+ny+1):
            Ax[i] += A[i, j] * x[j]
    # r = b - dot(A,x)
    # p = r
    for i in range(n):
        r[i] = b[i] - Ax[i]
        p[i] = r[i]
    rsold = 0.0
    for i in range(n):
        rsold += r[i] * r[i]

    for steps in range(100*n):
        # dot(A,p)
        for i in range(n):
            Ap[i] = 0.0
            for j in range(i-ny, i+ny+1):
                Ap[i] += A[i, j] * p[j]

        # dot(p, Ap) => pAp
        pAp = 0.0
        for i in range(n):
            pAp += p[i] * Ap[i]

        alpha = rsold / pAp

        # x = x + dot(alpha,p)
        # r = r - dot(alpha,Ap)
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]

        rsnew = 0.0
        for i in range(n):
            rsnew += r[i] * r[i]

        if ti.sqrt(rsnew) < 1e-8:
            print("The solution has converged...")
            break

        for i in range(n):
            p[i] = r[i] + (rsnew / rsold) * p[i]
        rsold = rsnew

        print("Iteration ", steps, ", residual = ", rsold)



@ti.kernel
def bicg(A: ti.template(),
         b: ti.template(),
         x: ti.template(),
         M: ti.template(),
         Ax: ti.template(),
         Ap: ti.template(),
         Ap_tld: ti.template(),
         r: ti.template(),
         p: ti.template(),
         z: ti.template(),
         r_tld: ti.template(),
         p_tld: ti.template(),
         z_tld: ti.template(),
         nx: ti.i32,
         ny: ti.i32,
         n: ti.i32):

    # dot(A,x)
    for i in range(n):
        Ax[i] = 0.0
        for j in range(i-ny-1, i+ny+2):
            Ax[i] += A[i, j] * x[j]

    # r = b - dot(A,x)
    for i in range(n):
        r[i] = b[i] - Ax[i]
        r_tld[i] = r[i]

    rsold = 0.0
    for i in range(n):
        rsold += r[i] * r[i]

    print("The initial res is ", rsold)

    rho_1 = 1.0
    for steps in range(n):

        for i in range(n):
            z[i] = 1.0 / M[i, i] * r[i]
            z_tld[i] = 1.0 / M[i, i] * r_tld[i]

        rho = 0.0
        for i in range(n):
            rho += z[i] * r_tld[i]
        if rho == 0.0:
            print("Bicg failed...")
            break

        if steps == 0:
            for i in range(n):
                p[i] = z[i]
                p_tld[i] = z_tld[i]
        else:
            beta = rho / rho_1
            for i in range(n):
                p[i] = z[i] + beta * p[i]
                p_tld[i] = z_tld[i] + beta * p_tld[i]

        # dot(A,p)
        for i in range(n):
            Ap[i] = 0.0
            Ap_tld[i] = 0.0
            for j in range(i-ny-1, i+ny+2):
                # Ap => q
                Ap[i] += A[i, j] * p[j]
                # Ap_tld => q_tld
                Ap_tld[i] += A[j, i] * p_tld[j]

        # dot(p, Ap) => pAp
        pAp = 0.0
        for i in range(n):
            pAp += p_tld[i] * Ap[i]

        alpha = rho / pAp

        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
            r_tld[i] -= alpha * Ap_tld[i]

        rsnew = 0.0
        for i in range(n):
            rsnew += r[i] * r[i]
        rsold = rsnew
        print("Iteration ", steps, ", residual = ", rsold)

        if ti.sqrt(rsnew) < 1e-5:
            print("The solution has converged...")
            break
        rho_1 = rho


@ti.kernel
def bicgstab(A:ti.template(),
             b:ti.template(),
             x:ti.template(),
             M:ti.template(),
             Ax:ti.template(),
             r:ti.template(),
             r_tld:ti.template(),
             p:ti.template(),
             p_hat:ti.template(),
             Ap:ti.template(),
             s:ti.template(),
             s_hat:ti.template(),
             t:ti.template(),
             nx:ti.i32,
             ny:ti.i32,
             n:ti.i32):
    # dot(A,x)
    for i in range(n):
        Ax[i] = 0.0
        for j in range(i-ny-1, i+ny+2):
            #        for j in range(n):
            Ax[i] += A[i, j] * x[j]

    # r = b - dot(A,x)
    for i in range(n):
        r[i] = b[i] - Ax[i]
        r_tld[i] = r[i]

    omega = 1.0
    alpha = 1.0
    beta = 1.0
    rho_1 = 1.0    
    for steps in range(100*n):
        
        rho = 0.0
        for i in range(n):
            rho += r[i] * r_tld[i]
        if rho == 0.0:
            print("Bicgstab failed...")
            break
            
        if steps == 0:
            for i in range(n):
                p[i] = r[i]
        else:
            beta = (rho / rho_1) * (alpha/omega)
            for i in range(n):
                p[i]  = r[i] + beta*(p[i] - omega*Ap[i])
        for i in range(n):
            p_hat[i] = 1/M[i,i] * p[i]
        
        # dot(A,p)
        # Ap => v        
        for i in range(n):
            Ap[i] = 0.0
            for j in range(i-ny-1, i+ny+2):
                # for j in range(n):
                Ap[i] += A[i, j] * p_hat[j]

        alpha_lower = 0.0
        for i in range(n):
            alpha_lower += r_tld[i] * Ap[i]
        alpha = rho / alpha_lower

        for i in range(n):
            s[i] = r[i] - alpha * Ap[i]

        # Early convergnece check...
        for i in range(n):
            s_hat[i] = 1/M[i, i]*s[i]

        for i in range(n):
            t[i] = 0.0
            for j in range(i-ny-1, i+ny+2):
                # for j in range(n):
                t[i] += A[i, j] * s_hat[j]

        omega_upper = 0.0
        omega_lower = 0.0
        for i in range(n):
            omega_upper += t[i] * s[i]
            omega_lower += t[i] * t[i]
        omega = omega_upper / omega_lower

        for i in range(n):
            x[i] += alpha* p_hat[i] + omega*s_hat[i]

        for i in range(n):
            r[i] = s[i] - omega*t[i]

        residual = 0.0
        for i in range(n):
            residual += r[i] * r[i]
        print("Iteration ", steps, ", residual = ", residual)
        if ti.sqrt(residual) < 1e-5:
            print("The solution has converged...")
            break
        if omega==0.0:
            print("Omega = 0.0 ...")
            break

        rho_1 = rho

        

@ti.kernel
def xu_back():
    for i, j in ti.ndrange(nx + 1, ny):
        u[i + 1, j + 1] = xu[i * ny + j]


@ti.kernel
def xv_back():
    for i, j in ti.ndrange(nx, ny + 1):
        v[i + 1, j + 1] = xv[i * ny + j]


def solve_momentum_jacob():
    for steps in range(50):
        residual = 10.0
        residual_x = 0.0
        residual_y = 0.0
        while residual > 1e-5:
            fill_Au()
            fill_Av()
            print("Residual = ", residual)
            residual_x = quick_jacobian(Au, bu, xu, xu_new)
            residual_y = quick_jacobian(Av, bv, xv, xv_new)
            residual = residual_x + residual_y
        xu_back()
        xv_back()


def solve_momentum_bicg():
    for steps in range(50):
        fill_Au()
        fill_Av()
        # Confirmed that bicg will give an answer for Au but the
        # convergence is slow and spiky.
        bicg(Au, bu, xu, Mu, Auxu, Aupu, Aupu_tld, ru,
             pu, zu, ru_tld, pu_tld, zu_tld, nx, ny, (nx+1)*ny)
        bicg(Av, bv, xv, Mv, Avxv, Avpv, Avpv_tld, rv,
            pv, zv, rv_tld, pv_tld, zv_tld, nx, ny, nx*(ny+1))
        # Confirmed that simple CG will diverge on this Au matrix.
        # struct_cg(Au, bu, xu, Auxu, Aupu, ru, pu, nx, ny)
        xu_back()
        xv_back()

        
def solve_momentum_bicgstab():
    for steps in range(50):
        fill_Au()
        fill_Av()
        bicgstab(Au, bu, xu, Mu, Auxu, ru, ru_tld, pu, pu_hat,
                 Aupu, su, su_hat, tu, nx, ny, (nx+1)*ny)
        bicgstab(Av, bv, xv, Mv, Avxv, rv, rv_tld, pv, pv_hat,
                 Avpv, sv, sv_hat, tv, nx, ny, nx*(ny+1))
        xu_back()
        xv_back()
    # write_matrix(Au, "Au")
    # write_matrix(Av, "Av")    

@ti.kernel        
def post_velocity():
    # Scale the velocity field so that max is <= 1.0.
    u_scale = 0.6
    v_scale = 0.6
    for i,j in ti.ndrange(nx+2,ny+2):
        u_post[i,j] = 0.5 * (u[i,j] + u[i+1,j]) * u_scale
        v_post[i,j] = 0.5 * (v[i,j] + v[i,j+1]) * v_scale
    # Exterpolate velocity to a bigger canvas.        
    for i,j in ti.ndrange(3*(nx+2),3*(ny+2)):
        u_disp[i,j] = u_post[i//3, j//3]
        v_disp[i,j] = v_post[i//3, j//3]        
        

if __name__ == "__main__":
    init()
    start = time.time()
    
    #solve_momentum_jacob()
    #solve_momentum_bicg()
    solve_momentum_bicgstab()
    
    print("The velocity profile at the inlet:")
    for j in range(ny+2):
        print("i = ", 1, ", j = ", j, ", u = ", u[1, j])
    print("The velocity profile at the outlet:")        
    for j in range(ny+2):
        print("i = ", nx+1, ", j = ", j, ", u = ", u[nx+1, j])
    print("It took ", time.time()-start,"sec to solve the momentum.")

    post_velocity()
    gui = ti.GUI("velocity plot", (3*(nx+2),6*(ny+2)))
    img = np.concatenate((u_disp.to_numpy(), v_disp.to_numpy()), axis =1)
    gui.set_image(img)
    gui.show("sample.png")
    
    
        
