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

# fill_Ap
# 1. Be careful with the nx and ny settings. Because you might lose cells if nx or ny is not divisible
#    by 8 when allocating pointer structures.
# 2. Accessing elements out of bound is causing error in Ap calculations. For example when A[-1,0] != 0.0,
#    it will be added to Ap[0, 0].

# Problems with p correction:
# 1. P correction seems to be proportional to grid size. The smaller the grid, the bigger the correction.
#    => My current explanation: the fix velocity inlet is creating 2 singular point on the corner, and
#       generating unlimited source of p correction.
#    => OpenFOAM's simpleFOAM showed similar trend, the pressure on the corner will grow proportional
#       to the grid size, the smaller the grid, the higher the pressure on corner.
#    => The pressure on the inlet and outlet are fixed by setting the pressure correction to zero.
#       See the fill_Ap() func, where Ap[i,j] = 1 and bp[] = 0.0
#       Accordingly, u velocity on inlet and outlet are set to be zero gradient.
#       Give the intended the pressure difference in init(), and velocity will develop itselt.
# 2. V momentum xv_back() was wrong => Now corrected.
# 3. The under-relaxation of velocity was wrong. Velocity correction does NOT need any relaxation.
#    Instead, under relax the velocity change inside the solving of momentum eqn.
#    This is implemented in the xu_back() and xv_back().
# 4. Implemented a convergence checker in puv_correction: return the max p correction.


lx = 0.5
ly = 0.1

# nx and ny have to be multiples of 8.
nx = 320
ny = 64
ptr_size = 16

rho = 1
mu = 0.01
dx = lx / nx
dy = ly / ny
dt = 0.001

# Relaxation factors
velo_rel = 0.5
p_rel = 0.08

# Add 1 cell padding to all directions.
p = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 2))
pcor = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 2))
p_disp = ti.field(dtype=ti.f64, shape=(3 *(nx + 2), 3*(ny + 2)))
pcor_disp = ti.field(dtype=ti.f64, shape=(3 *(nx + 2), 3*(ny + 2)))
udiv = ti.field(dtype=ti.f64, shape=(nx + 2, ny + 2))
udiv_disp = ti.field(dtype=ti.f64, shape=(3 *(nx + 2), 3*(ny + 2)))

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
# Au = ti.field(dtype=ti.f64, shape=((nx + 1) * ny, (nx + 1) * ny))
# Mu = ti.field(dtype=ti.f64, shape=((nx + 1) * ny, (nx + 1) * ny))
# Au and Mu declared as multiple layers to avoid virtual memory error.
Au = ti.field(dtype=ti.f64)
Mu = ti.field(dtype=ti.f64)
ti.root.pointer(ti.ij,((nx+1)*ny//ptr_size,(nx+1)*ny//ptr_size)).dense(ti.ij,(ptr_size,ptr_size)).place(Au, Mu)

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
# Av = ti.field(dtype=ti.f64, shape=(nx * (ny + 1), nx * (ny + 1)))
# Mv = ti.field(dtype=ti.f64, shape=(nx * (ny + 1), nx * (ny + 1)))
Av = ti.field(dtype=ti.f64)
Mv = ti.field(dtype=ti.f64)
ti.root.pointer(ti.ij,(nx*(ny+1)//ptr_size,nx*(ny+1)//ptr_size)).dense(ti.ij,(ptr_size,ptr_size)).place(Av, Mv)

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
Ap = ti.field(dtype=ti.f64)
Mp = ti.field(dtype=ti.f64)
ti.root.pointer(ti.ij,(nx*ny//ptr_size,nx*ny//ptr_size)).dense(ti.ij,(ptr_size,ptr_size)).place(Ap, Mp)

bp = ti.field(dtype=ti.f64)
xp = ti.field(dtype=ti.f64)
Apxp = ti.field(dtype=ti.f64)
rp = ti.field(dtype=ti.f64)
rp_tld = ti.field(dtype=ti.f64)
pp = ti.field(dtype=ti.f64)
pp_hat = ti.field(dtype=ti.f64)
Appp = ti.field(dtype=ti.f64)
sp = ti.field(dtype=ti.f64)
sp_hat = ti.field(dtype=ti.f64)
tp = ti.field(dtype=ti.f64)
ti.root.dense(ti.i, nx * ny).place(bp, xp, Apxp, rp, rp_tld, pp, pp_hat, Appp, sp, sp_hat, tp)


@ti.kernel
def init():
    for i, j in ti.ndrange(nx + 2, ny + 2):
        # Give the intended pressure drop here
        # p[i, j] = 100 - 6.0 * i / nx
        p[i, j] = 100 - 240.0 * i / nx        
    for i, j in ti.ndrange(nx + 3, ny + 2):
        u[i, j] = 0.0
        u0[i, j] = 0.0 # u[i, j]
    for i, j in ti.ndrange(nx + 2, ny + 3):
        v[i, j] = 0.0
        v0[i, j] = 0.0 # v[i, j]

    for i, j in ti.ndrange(nx + 2, ny + 2):
        ct[i, j] = 1  # "1" stands for solid
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 1)):
        ct[i, j] = -1  # "-1" stands for fluid
        
    for i, j in ti.ndrange(nx, ny):
        if (((i - 64)**2 + (j - 32)**2) < 64):
            ct[i, j] = 1
            u[i, j] = 0
            u0[i, j] = 0
            v[i, j] = 0
            v0[i, j] = 0

def write_matrix(mat, name):
    print("    >> Writing matrix data to", name, ".csv ...")
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
        # ct[i - 1, j] is the left cell of u[i,j]
        # ct[i,j] + ct[i-1,j] = 2 means the u is inside a block
        if ct[i - 1, j] == 1:
            # Au[k, k] = 1.0
            # bu[k] = u[i, j]
            # For pressure inlet
            Au[k, k] = 1.0
            Au[k, k+ny] = -1.0
            bu[k] = 0.0
            
        # Outlet
        # ct[i,j] is the right cell of u[i,j]
        elif ct[i, j] == 1:
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
def fill_Ap():
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        bp[k] = rho * (u[i, j] - u[i + 1, j]) * dy + rho * (v[i, j + 1] - v[i, j]) * dx
        # The following change does reduce the magnitude of pcor, but is incorrect...?
        # bp[k] = rho * (u[i, j] - u[i + 1, j]) * dy *dx + rho * (v[i, j + 1] - v[i, j]) * dx*dy
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
        if (ct[i, j] + ct[i, j + 1]) == 0:
            Ap[k, k + 1] = 0
        if (ct[i, j] + ct[i - 1, j]) == 0:
            Ap[k, k - ny] = 0
        if (ct[i, j] + ct[i + 1, j]) == 0:
            Ap[k, k + ny] = 0
        Ap[k, k] = -Ap[k, k - 1] - Ap[k, k + 1] - Ap[k, k - ny] - Ap[k, k + ny]
        # if k==0:
        #     print(Ap[k,k-1], Ap[k,k+1], Ap[k,k-ny], Ap[k,k+ny], Ap[k,k])
        Mp[k, k] = Ap[k, k]
        
    # Inlet and outlet pressure correction all equal zero. (Fixed pressure)
    for i,j in ti.ndrange(ny, nx * ny):
        if i == j:
            # Inlet
            Ap[i, j] = 1.0
            Mp[i, j] = 1.0
            bp[j] = 0.0
            # Outlet
            Ap[nx * ny - i, nx * ny - j] = 1.0                        
            Mp[nx * ny - i, nx * ny - j] = 1.0                        
            bp[nx*ny - j] = 0.0
        else:
            Ap[i, j] = 0.0
            Mp[i, j] = 0.0            


@ti.kernel
def fill_Au_ver2():
    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        
        # Inlet
        # ct[i - 1, j] is the left cell of u[i,j]
        # ct[i,j] + ct[i-1,j] = 2 means the u is inside a block
        if ct[i - 1, j] == 1:
            # Au[k, k] = 1.0
            # bu[k] = u[i, j]
            # For pressure inlet
            Au[k, k] = 1.0
            Au[k, k+ny] = -1.0
            bu[k] = 0.0
            
        # Outlet
        # ct[i,j] is the right cell of u[i,j]
        elif ct[i, j] == 1:
            Au[k, k] = 1.0  # Au[k-ny,k-ny]
            Au[k, k - ny] = -1.0  # -Au[k,k]
            bu[k] = 0.0

        # Upper boundary
        # ct[i, j - 1] is the upper cell of u[i,j]
        elif (ct[i, j] + ct[i, j - 1]) == 0:
            # Notice that 2*mu should be followed by dx/dy.
            Au[k, k + 1] = -mu * dx / dy - \
                ti.max(0, rho * 0.5 *
                       (v[i - 1, j + 1] + v[i, j + 1]) * dx)  # as
            Au[k, k - ny] = -mu * dy / dx - \
                ti.max(0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy)  # aw
            Au[k, k + ny] = -mu * dy / dx - \
                ti.max(0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy)  # ae
            Au[k, k] = - Au[k, k + 1] - Au[k, k - ny] - \
                Au[k, k + ny] + rho * dx * dy / dt  + 2 * mu * dx/dy # ap
            bu[k] = (p[i - 1, j] - p[i, j]) * dy + rho * dx * \
                dy / dt * u0[i, j]  # <= Unsteady term

        # Lower boundary
        elif (ct[i, j] + ct[i, j + 1]) == 0:
            Au[k, k - 1] = -mu * dx / dy - \
                ti.max(0, -rho * 0.5 * (v[i - 1, j] + v[i, j]) * dx)  # an
            Au[k, k - ny] = -mu * dy / dx - \
                ti.max(0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy)  # aw
            Au[k, k + ny] = -mu * dy / dx - \
                ti.max(0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy)  # ae
            Au[k, k] = -Au[k, k - 1] - Au[k, k - ny] - \
                Au[k, k + ny] + rho * dx * dy / dt  + 2* mu * dx/dy # ap
            bu[k] = (p[i - 1, j] - p[i, j]) * dy + rho * dx * \
                dy / dt * u0[i, j]  # <= Unsteady term
            
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
            
        Mu[k,k] = Au[k,k]

    # Internal obstacle
    for i, j in ti.ndrange((2, nx + 1), (2, ny)):
        k = (i - 1) * ny + (j - 1)
        
        # For u velocity on the interface or inside the obstacle
        if (ct[i - 1, j] + ct[i, j]) == 0 or (ct[i - 1, j] + ct[i, j]) == 2:
            for nb in ti.static([k-ny,k-1,k+1,k+ny]):
                Au[k, nb] = 0.0
            Au[k, k] = 1.0
            bu[k] = 0.0

        elif (ct[i, j] + ct[i, j - 1]) == 0 or (ct[i - 1, j] + ct[i - 1, j - 1]) == 0:
            # Notice that 2*mu should be followed by dx/dy.
            Au[k, k + 1] = -mu * dx / dy - \
                ti.max(0, rho * 0.5 *
                       (v[i - 1, j + 1] + v[i, j + 1]) * dx)  # as
            Au[k, k - ny] = -mu * dy / dx - \
                ti.max(0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy)  # aw
            Au[k, k + ny] = -mu * dy / dx - \
                ti.max(0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy)  # ae
            Au[k, k] = - Au[k, k + 1] - Au[k, k - ny] - \
                Au[k, k + ny] + rho * dx * dy / dt  + 2 * mu * dx/dy # ap
            bu[k] = (p[i - 1, j] - p[i, j]) * dy + rho * dx * \
                dy / dt * u0[i, j]  # <= Unsteady term
            
        elif (ct[i, j] + ct[i, j - 1]) == 0 or (ct[i - 1, j] + ct[i - 1, j - 1]) == 0:
            Au[k, k - 1] = -mu * dx / dy - \
                ti.max(0, -rho * 0.5 * (v[i - 1, j] + v[i, j]) * dx)  # an
            Au[k, k - ny] = -mu * dy / dx - \
                ti.max(0, rho * 0.5 * (u[i, j] + u[i - 1, j]) * dy)  # aw
            Au[k, k + ny] = -mu * dy / dx - \
                ti.max(0, -rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy)  # ae
            Au[k, k] = -Au[k, k - 1] - Au[k, k - ny] - \
                Au[k, k + ny] + rho * dx * dy / dt  + 2* mu * dx/dy # ap
            bu[k] = (p[i - 1, j] - p[i, j]) * dy + rho * dx * \
                dy / dt * u0[i, j]  # <= Unsteady term


@ti.kernel
def fill_Av_ver2():
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
def fill_Ap_ver2():
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        bp[k] = rho * (u[i, j] - u[i + 1, j]) * dy + rho * (v[i, j + 1] - v[i, j]) * dx
        # The following change does reduce the magnitude of pcor, but is incorrect...?
        # bp[k] = rho * (u[i, j] - u[i + 1, j]) * dy *dx + rho * (v[i, j + 1] - v[i, j]) * dx*dy
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
        if (ct[i, j] + ct[i, j + 1]) == 0:
            Ap[k, k + 1] = 0
        if (ct[i, j] + ct[i - 1, j]) == 0:
            Ap[k, k - ny] = 0
        if (ct[i, j] + ct[i + 1, j]) == 0:
            Ap[k, k + ny] = 0
        Ap[k, k] = -Ap[k, k - 1] - Ap[k, k + 1] - Ap[k, k - ny] - Ap[k, k + ny]
        # if k==0:
        #     print(Ap[k,k-1], Ap[k,k+1], Ap[k,k-ny], Ap[k,k+ny], Ap[k,k])
        Mp[k, k] = Ap[k, k]
        
    # Inlet and outlet pressure correction all equal zero. (Fixed pressure)
    for i,j in ti.ndrange(ny, nx * ny):
        if i == j:
            # Inlet
            Ap[i, j] = 1.0
            Mp[i, j] = 1.0
            bp[j] = 0.0
            # Outlet
            Ap[nx * ny - i, nx * ny - j] = 1.0                        
            Mp[nx * ny - i, nx * ny - j] = 1.0                        
            bp[nx*ny - j] = 0.0
        else:
            Ap[i, j] = 0.0
            Mp[i, j] = 0.0            

            

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
         ny: ti.i32,
         eps: ti.f64,
         output: ti.i32):
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

        if ti.sqrt(rsnew) < eps:
            if output:
                print("        >> The solution has converged...")
            break

        for i in range(n):
            p[i] = r[i] + (rsnew / rsold) * p[i]
        rsold = rsnew
        
        if output:
            print("        >> Iteration ", steps, ", residual = ", rsold)



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
         n: ti.i32,
         eps: ti.f64,
         output: ti.i32):

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

    if output:
        print("        >> The initial res is ", rsold)

    rho_1 = 1.0
    for steps in range(n):

        for i in range(n):
            z[i] = 1.0 / M[i, i] * r[i]
            z_tld[i] = 1.0 / M[i, i] * r_tld[i]

        rho = 0.0
        for i in range(n):
            rho += z[i] * r_tld[i]
        if rho == 0.0:
            if output:
                print("        >> Bicg failed...")
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
        if output:
            print("        >> Iteration ", steps, ", residual = ", rsold)

        if ti.sqrt(rsnew) < eps:
            if output:
                print("        >> The solution has converged...")
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
             n:ti.i32,
             eps: ti.f64,
             output:ti.i32):
    # dot(A,x)
    for i in range(n):
        Ax[i] = 0.0
        # Only traverse certain elements. Need to use ti.static() to convert python list.        
        for j in ti.static([i-ny-1,i-ny,i-1,i,i+1,i+ny,i+ny+1]):        
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
            if output:
                print("        >> Bicgstab failed...")
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
            # Only traverse certain elements. Need to use ti.static() to convert python list.
            for j in ti.static([i-ny-1,i-ny,i-1,i,i+1,i+ny,i+ny+1]):
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
            # Only traverse certain elements. Need to use ti.static() to convert python list.            
            for j in ti.static([i-ny-1,i-ny,i-1,i,i+1,i+ny,i+ny+1]):
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
        if output:            
            print("        >> Iteration ", steps, ", residual = ", residual)
            
        if ti.sqrt(residual) < eps:
            if output:
                print("        >> The solution has converged...")
            break
        
        if omega==0.0:
            if output:
                print("        >> Omega = 0.0 ...")
            break
        
        rho_1 = rho
        

@ti.kernel
def xu_back():
    for i, j in ti.ndrange(nx + 1, ny):
        # u[i + 1, j + 1] = xu[i * ny + j]
        # New velocity under-relaxation
        u[i + 1, j + 1] = u[i + 1, j + 1] + velo_rel * (xu[i * ny + j] - u[i + 1, j + 1])
            


@ti.kernel
def xv_back():
    for i, j in ti.ndrange(nx, ny + 1):
        # v[i + 1, j + 1] = xv[i * ( ny + 1 ) + j]
        # New velocity under-relaxation
        v[i + 1, j + 1] = v[i + 1, j + 1] + velo_rel * (xv[i * ( ny + 1 ) + j] - v[i + 1, j + 1])


@ti.kernel
def xp_back():
    for i, j in ti.ndrange(nx, ny):
        pcor[i + 1, j + 1] = xp[i * ny + j]


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


def solve_momentum_bicg(eps, max_iterations, solve_output, linalg_output):
    print("    >> Now Solving momentum equations using BiCG...")    
    for steps in range(max_iterations):
        if solve_output:
            print("    [", steps,"/",max_iterations, "] Iteratively solving momentum equation using BiCG...")            
        fill_Au()
        fill_Av()
        # Confirmed that bicg will give an answer for Au but the
        # convergence is slow and spiky.
        bicg(Au, bu, xu, Mu, Auxu, Aupu, Aupu_tld, ru,
             pu, zu, ru_tld, pu_tld, zu_tld, nx, ny, (nx+1)*ny, eps, linalg_output)
        bicg(Av, bv, xv, Mv, Avxv, Avpv, Avpv_tld, rv,
             pv, zv, rv_tld, pv_tld, zv_tld, nx, ny, nx*(ny+1), eps, linalg_output)
        # Confirmed that simple CG will diverge on this Au matrix.
        # struct_cg(Au, bu, xu, Auxu, Aupu, ru, pu, nx, ny)
        xu_back()
        xv_back()

        
# solve_output contorls whether show iteration history of momentum eqn.
# linalg_output controls whether show convergence of BiCGSTAB solver.
def solve_momentum_bicgstab(eps, max_iterations, solve_output, linalg_output):
    print("    >> Now Solving momentum equations using BiCGSTAB...")
    for steps in range(max_iterations):
        if solve_output:
            print("    [", steps,"/",max_iterations, "] Iteratively solving momentum equation using BiCGSTAB...")
        fill_Au_ver2()
        fill_Av_ver2()
        bicgstab(Au, bu, xu, Mu, Auxu, ru, ru_tld, pu, pu_hat,
                 Aupu, su, su_hat, tu, nx, ny, (nx+1)*ny, eps, linalg_output)
        bicgstab(Av, bv, xv, Mv, Avxv, rv, rv_tld, pv, pv_hat,
                 Avpv, sv, sv_hat, tv, nx, ny, nx*(ny+1), eps, linalg_output)
        xu_back()
        xv_back()
    # write_matrix(Au, "Au")
    # write_matrix(Av, "Av")    


def solve_pcorrection_bicgstab(eps, linalg_output):
    print("    >> Now Solving pressure correction using BiCGSTAB...")
    fill_Ap_ver2()

    # write_matrix(Au, "Au")
    # write_matrix(Av, "Av")
    bicgstab(Ap, bp, xp, Mp, Apxp, rp, rp_tld, pp, pp_hat,
             Appp, sp, sp_hat, tp, nx, ny, nx * ny, eps, linalg_output)
    xp_back()
    # write_matrix(Ap, "Ap")
    # write_matrix(bp, "bp")
    # write_matrix(xp, "xp")            
    # write_matrix(pcor, "pcor")


@ti.kernel
def puv_correction()->ti.f64:
    pcor_max = 0.0
    
    ucor_max = 0.0
    for i, j in ti.ndrange((1, nx + 2), (1, ny + 1)):
        k = (i - 1) * ny + (j - 1)
        # Upper and lower boundary
        if (ct[i - 1, j] + ct[i, j]) == 0 or (ct[i - 1, j] + ct[i, j]) == 2:
            pass
        else:
            ucor[i, j] = (pcor[i - 1, j] - pcor[i, j]) * dy / Au[k, k]
            u[i, j] = u[i, j] + ucor[i, j]
            if np.abs(ucor[i, j] / (u[i, j] + 1.0e-9)) >= ucor_max:
                ucor_max = np.abs(ucor[i, j] / (u[i, j] + 1.0e-9))
                
    vcor_max = 0.0
    for i, j in ti.ndrange((1, nx + 1), (1, ny + 2)):
        k = (i - 1) * (ny + 1) + (j - 1)
        # Upper and lower boundary
        if (ct[i, j] + ct[i, j - 1]) == 0 or (ct[i, j] + ct[i, j - 1]) == 2:
            pass
        else:
            vcor[i, j] = (pcor[i, j] - pcor[i, j - 1]) * dx / Av[k, k]
            v[i, j] = v[i, j] + vcor[i, j]
            if np.abs(vcor[i, j] / (v[i, j] + 1.0e-9)) >= vcor_max:
                vcor_max = np.abs(vcor[i, j] / (v[i, j] + 1.0e-9))

    for i, j in ti.ndrange(nx + 2, ny + 2):
        if ct[i, j] == 1:
            pass
        else:
            p[i, j] = p[i, j] + p_rel * pcor[i, j]
        if ti.abs(pcor[i, j]) > pcor_max:
            pcor_max = ti.abs(pcor[i, j])
    return pcor_max
    
    
@ti.kernel        
def post_process_field():
    # First, interpolate u,v onto the center of a cell
    for i,j in ti.ndrange(nx+2,ny+2):
        u_post[i,j] = 0.5 * (u[i,j] + u[i+1,j])
        v_post[i,j] = 0.5 * (v[i,j] + v[i,j+1])
        
    # Calculate the divergence of the velocity field
    # udiv is of [nx+2, ny+2] size but only calculated in [(1,nx+1), (1,ny+1)] area.
    for i,j in ti.ndrange((1,nx+1), (1,ny+1)):
        udiv[i,j] = (u[i,j] - u[i+1,j]) * dy + (v[i,j+1] - v[i,j]) * dx
        
    # Exterpolate velocity to a bigger canvas.        
    for i,j in ti.ndrange(3*(nx+2),3*(ny+2)):
        u_disp[i,j] = u_post[i//3, j//3]
        v_disp[i,j] = v_post[i//3, j//3]
        udiv_disp[i,j] = udiv[i//3, j//3]
        p_disp[i,j] = p[i//3, j//3]
        pcor_disp[i,j] = pcor[i//3, j//3]

    # Scale the value of display field to range [0,1]
    scale_field(u_disp)
    scale_field(v_disp)
    scale_field(p_disp)
    scale_field(pcor_disp)
    scale_field(udiv_disp)


@ti.func
def scale_field(f):
    f_max = 0.0
    f_min = 1.0e9
    for i,j in f:
        if f[i,j] > f_max:
            f_max = f[i,j]
        if f[i,j] < f_min:
            f_min = f[i,j]
    for i,j in f:
        f[i,j] = (f[i,j] - f_min) / (f_max - f_min + 1.0e-9)

        
@ti.kernel
def correct_conserv():
    mdot_inlet = 0.0
    mdot_outlet = 0.0
    coef = 1.0
    
    for j in range(1,ny+1):
        mdot_inlet += u[1, j]
        mdot_outlet += u[nx+1, j]
    
    # print("        >> Before correction, the mass flow at the inlet is", mdot_inlet)
    # print("        >> Before correction, the mass flow at the outlet is", mdot_outlet)    
    coef = mdot_inlet / mdot_outlet
    
    for j in range(1,ny+1):
        u[nx+1, j] = coef * u[nx+1, j]

    mdot_outlet = 0.0
    for j in range(1,ny+1):
        mdot_outlet += u[nx+1, j]
    # print("        >> After correction, the mass flow at the outlet is", mdot_outlet)

    # Check the overall divergence of the whole velocity field. Should be zero
    # after correction.
    udiv_overall = 0.0
    for i,j in ti.ndrange((1,nx+1), (1,ny+1)):
        udiv[i, j] = (u[i, j] - u[i+1, j]) * dy + (v[i, j+1] - v[i, j]) * dx
        udiv_overall += udiv[i, j]
    # print("The overall divergence of velocity after correction is", udiv_overall)


@ti.kernel
def time_forward():
    for i,j in ti.ndrange(nx+3, ny+2):
        u0[i, j] = u[i, j]
    for i,j in ti.ndrange(nx+2, ny+3):
        v0[i, j] = v[i, j]


if __name__ == "__main__":
    init()
    
    for time_step in range(5000):
        # subtime_step is the iteration cycle inside a time-step.
        # maximum steps is 50, or iteration will end when the overall residual is < 1.0e-8
        for subtime_step in range(50):
            pcor_max = 0.0
            start = time.time()
            #solve_momentum_jacob()
            #solve_momentum_bicg()
            # (eps, max_iterations, solver_output, linalg_output)
            solve_momentum_bicgstab(1e-8, 30, 0, 0)
            print("    >> [ time =", time_step * dt, "] It took ", time.time()-start,"sec to solve the momentum equation.")

            # Use write function to output matrix as desired.    
            # write_matrix(Au, "Au")
            # write_matrix(Av, "Av")
            # print("THe shape of Au is ", Au.shape)
            # print("The shape of Av is ", Av.shape)
    
            correct_conserv()
            solve_pcorrection_bicgstab(1e-8, 0)
            
            pcor_max = puv_correction()
            post_process_field()
            print("    >> [ time =", time_step * dt, "] The current max p-correction is", pcor_max)            
            if pcor_max < 1.0e-2:
                print("    >> [ time =", time_step * dt, "] The flow field has converged on this time step.")
                break
            
        time_forward()
        
        # Do not show the gui, only save the png when gui.show() is called
        gui = ti.GUI("velocity plot", (3*(nx+2),15*(ny+2)), show_gui=False)
        img = np.concatenate((pcor_disp.to_numpy(), udiv_disp.to_numpy(), p_disp.to_numpy(), u_disp.to_numpy(), v_disp.to_numpy()), axis =1)
        gui.set_image(img)

        text_color = 0x3355ff
        gui.text(content = "P correction"  , pos = (0.5,0.1), font_size = 20, color=text_color )
        gui.text(content = "Velocity div", pos = (0.5,0.3), font_size = 20, color=text_color )
        gui.text(content = "Pressure", pos = (0.5,0.5), font_size = 20, color=text_color )
        gui.text(content = "U velocity", pos = (0.5,0.7), font_size = 20, color=text_color )
        gui.text(content = "V velocity", pos = (0.5,0.9), font_size = 20, color=text_color )
        
        filename = "timestep" + str(time_step) + ".png"
        gui.show(filename)

    # Write the fields at the last time step
    write_matrix(pcor, "pcor")
    write_matrix(p, "p")
    write_matrix(u, "u")
    write_matrix(v, "v")
    write_matrix(udiv, "udiv")
