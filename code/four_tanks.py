from casadi import *
import scipy as sp

# system identification
system = 'four_tanks'

# system dimensions
n = 4
m = 2
p = 2

# system matrices
A = DM([
    [0.921, 0.0,   0.041, 0.0],
    [0.0,   0.918, 0.0,   0.033],
    [0.0,   0.0,   0.924, 0.0],
    [0.0,   0.0,   0.0,   0.937]])
B = DM([
    [0.017, 0.001],
    [0.001, 0.023],
    [0.0,   0.061],
    [0.072, 0.0]])
C = DM([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])
D = DM.zeros((p, m))

# # prints
# print(f"eigs of A: {np.linalg.eigvals(A.full())}")

# # check controllability
# Cr = B
# tmp = B
# for i in range(1,n):
#     tmp = mtimes(A,tmp)
#     Cr = horzcat(Cr, tmp)
# print(f"rank controllability matrix: {np.linalg.matrix_rank(Cr.full())}")

# # check observability
# Ob = C
# tmp = C
# for i in range(1,n):
#     tmp = mtimes(tmp,A)
#     Ob = vertcat(Ob, tmp)
# print(f"rank observability matrix: {np.linalg.matrix_rank(Ob.full())}")

# # search lag (l=2)
# l_list = np.arange(1, n+1)
# for l in l_list:
#     obs = DM(C)
#     tmp = DM(C)
#     for i in range(1,l):
#         tmp = mtimes(tmp, A)
#         obs = vertcat(obs, tmp)
#     print(f"lag: {l}, rank obs: {np.linalg.matrix_rank(obs.full())}")

def tanks_dyn():
    # symbolic state, input and output
    x = SX.sym('x_', n)
    u = SX.sym('u_', m)
    y = SX.sym('y_', p)

    # discrete time dynamics with integrator
    x_kp_sym = A@x + B@u
    y_k_sym = C@x + D@u

    f_dis = Function('f', [x, u], [x_kp_sym])
    g_dis = Function('g', [x, u], [y_k_sym])

    return f_dis, g_dis, x, u, y

# def LQR_controller(
#         x_k, 
#         x_ref, 
#         u_max,
#         err_int, 
#         dt
# ):
#     # compute stabilising K
#     Q_sp = 1e1*np.eye(n)
#     R_sp = 1e-2*np.eye(m)
#     P_sp = sp.linalg.solve_discrete_are(A, B, Q_sp, R_sp)
#     K_sp = -np.linalg.inv(R_sp + B.T @ P_sp @ B) @ (B.T @ P_sp @ A)

#     # print(f"K shape: {K_sp.shape}, K: {K_sp}")

#     # print(f"A-BK stable: {np.all(np.linalg.norm(np.linalg.eigvals(A+B@K_sp).reshape((n,1)),axis=1) <= 1.0)}")
#     print(x_ref.shape);exit()
#     u_k = np.clip(K_sp @ (x_k[:,np.newaxis] - x_ref), -u_max, u_max)

#     return u_k, err_int
