import numpy as np
from scipy.linalg import solve, lu_factor, lu_solve, cho_factor, cho_solve, eigvalsh
from scipy.sparse import issparse, diags, csr_matrix
from scipy.sparse.linalg import splu, SuperLU, minres

def mydot(A, B):
    r"""Dot-product that can handle dense and sparse arrays

    Parameters
    ----------
    A : numpy ndarray or scipy sparse matrix
        The first factor
    B : numpy ndarray or scipy sparse matrix
        The second factor

    Returns
    C : numpy ndarray or scipy sparse matrix
        The dot-product of A and B

    """
    if issparse(A) :
        return A.dot(B)
    elif issparse(B):
        return (B.T.dot(A.T)).T
    else:
        return np.dot(A, B)

def myfactor(A):
    if issparse(A):
        return splu(A.tocsc())
    else:
        return lu_factor(A)

def mysolve(LU, b):
    if isinstance(LU, SuperLU):
        return LU.solve(b)
    else:
        return lu_solve(LU, b)    

###############################################################################
# Solve via full system
###############################################################################

def factor_full(z, DPhival, G, A):
    return DPhival

def solve_full(z, Fval, DPhival, G, A):    
    M, N=G.shape
    P, N=A.shape

    """Total number of inequality constraints"""
    m=M    

    """Primal variable"""
    x=z[0:N]

    """Multiplier for equality constraints"""
    nu=z[N:N+P]

    """Multiplier for inequality constraints"""
    l=z[N+P:N+P+M]

    """Slacks"""
    s=z[N+P+M:]

    """Dual infeasibility"""
    rd = Fval[0:N]
    
    """Primal infeasibility"""
    rp1 = Fval[N:N+P]
    rp2 = Fval[N+P:N+P+M]

    """Centrality"""
    rc = Fval[N+P+M:]

    """Sigma matrix"""
    SIG = np.diag(l/s)

    # DPhival = DFval[0:N, 0:N]

    """Condensed system"""
    J = np.zeros((N+P, N+P))
    J[0:N, 0:N] = DPhival + mydot(G.T, mydot(SIG, G))
    J[0:N, N:] = A.T
    J[N:, 0:N] = A

    # Hxx = J[0:N/2,0:N/2]
    # Hyx = J[0:N/2,N/2:N]
    # Hyy = J[N/2:N,N/2:N]
    # CH_xx = cho_factor(Hxx)
    # S = -(Hyy + mydot(Hyx.T, cho_solve(CH_xx, Hyx)))
    # evs = eigvalsh(S)
    # print evs.max(), evs.min()
    # Ay = A[:, N/2:]
    # W = np.zeros((N/2+1, N/2+1))
    # W[0:N/2,0:N/2] = -Hyy
    # W[0:N/2,N/2:] = -Ay.T
    # W[N/2:,0:N/2] = -Ay
    # evs = eigvalsh(W)
    # print evs.min(), evs.max()

    b1 = -rd - mydot(G.T, mydot(SIG, rp2)) + mydot(G.T, rc/s)
    b2 = -rp1
    b = np.hstack((b1, b2))

    sign = np.zeros(N+P)
    sign[0:N/2] = 1.0
    sign[N/2:] = -1.0
    S = diags(sign, 0)
    J_new = mydot(S, csr_matrix(J))
    b_new = mydot(S, b)

    dJ_new = np.abs(J_new.diagonal())
    dPc = np.ones(J_new.shape[0])
    ind = (dJ_new > 0.0)
    dPc[ind] = 1.0/dJ_new[ind]
    Pc = diags(dPc, 0)    
    dxnu, info = minres(J_new, b_new, tol=1e-8, M=Pc)
    
    # dxnu = solve(J, b)
    dx = dxnu[0:N]
    dnu = dxnu[N:]

    """Obtain search directions for l and s"""
    ds = -rp2 - mydot(G, dx)
    dl = -mydot(SIG, ds) - rc/s

    dz = np.hstack((dx, dnu, dl, ds))
    return dz 

###############################################################################
# Solve via augmented system
###############################################################################

def factor_aug(z, DPhival, G, A):
    M, N = G.shape
    P, N = A.shape
    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks"""
    s = z[N+P+M:]

    """Sigma matrix"""
    SIG = diags(l/s, 0)

    # evs = eigvalsh(DPhival)
    # print evs.max(), evs.min()
    # print evs
    # tmp = np.zeros(N)
    # tmp[N/2:] = 1.0
    # print np.linalg.norm(np.dot(DPhival, tmp))
    # print tmp
    # print np.dot(DPhival, tmp)
    
    """Condensed system"""
    J = np.zeros((N+P, N+P))
    J[0:N, 0:N] = DPhival + mydot(G.T, mydot(SIG, G))    
    # """This is dirty, but would give big speedup => use sparse/block matrices"""
    # J[0:N, 0:N] = DPhival
    # J[np.diag_indices(N/2)] += l/s
    J[0:N, N:] = A.T
    J[N:, 0:N] = A

    LU = myfactor(J)    
    return LU

def solve_factorized_aug(z, Fval, LU, G, A):
    M, N=G.shape
    P, N=A.shape

    """Total number of inequality constraints"""
    m = M    

    """Primal variable"""
    x = z[0:N]

    """Multiplier for equality constraints"""
    nu = z[N:N+P]

    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks"""
    s = z[N+P+M:]

    """Dual infeasibility"""
    rd = Fval[0:N]
    
    """Primal infeasibility"""
    rp1 = Fval[N:N+P]
    rp2 = Fval[N+P:N+P+M]

    """Centrality"""
    rc = Fval[N+P+M:]

    """Sigma matrix"""
    SIG = diags(l/s, 0)

    """RHS for condensed system"""
    b1 = -rd - mydot(G.T, mydot(SIG, rp2)) + mydot(G.T, rc/s)
    b2 = -rp1
    b = np.hstack((b1, b2))
    dxnu = mysolve(LU, b)
    dx = dxnu[0:N]
    dnu = dxnu[N:]

    """Obtain search directions for l and s"""
    ds = -rp2 - mydot(G, dx)
    dl = -mydot(SIG, ds) - rc/s

    dz = np.hstack((dx, dnu, dl, ds))
    return dz

###############################################################################
# Solve via normal equations (Schur complement)
###############################################################################
    
def factor_schur(z, DPhival, G, A):
    M, N = G.shape
    P, N = A.shape
    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks"""
    s = z[N+P+M:]

    """Sigma matrix"""
    SIG = diags(l/s, 0)

    """Augmented Jacobian"""
    H = DPhival + mydot(G.T, mydot(SIG, G))

    """Factor H"""
    LU_H = myfactor(H)

    """Compute H^{-1}A^{T}"""
    HinvAt = mysolve(LU_H, A.T)

    """Compute Schur complement AH^{-1}A^{T}"""
    S = mydot(A, HinvAt)

    """Factor Schur complement"""
    LU_S = myfactor(S)

    LU = (LU_S, LU_H)
    return LU

def solve_factorized_schur(z, Fval, LU, G, A):
    M, N=G.shape
    P, N=A.shape

    """Total number of inequality constraints"""
    m = M    

    """Primal variable"""
    x = z[0:N]

    """Multiplier for equality constraints"""
    nu = z[N:N+P]

    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks"""
    s = z[N+P+M:]

    """Dual infeasibility"""
    rd = Fval[0:N]
    
    """Primal infeasibility"""
    rp1 = Fval[N:N+P]
    rp2 = Fval[N+P:N+P+M]

    """Centrality"""
    rc = Fval[N+P+M:]

    """Sigma matrix"""
    SIG = diags(l/s, 0)

    """Assemble right hand side of augmented system"""
    r1 = rd + mydot(G.T, mydot(SIG, rp2)) - mydot(G.T, rc/s)
    r2 = rp1

    """Unpack LU-factors"""
    LU_S, LU_H = LU

    """Assemble right hand side for normal equation"""
    b = r2 - mydot(A, mysolve(LU_H, r1))  

    """Solve for dnu"""
    dnu = mysolve(LU_S, b)
       
    """Solve for dx"""
    dx = mysolve(LU_H, -(r1 + mydot(A.T, dnu)))    
    
    """Obtain search directions for l and s"""
    ds = -rp2 - mydot(G, dx)
    dl = -mydot(SIG, ds) - rc/s

    dz = np.hstack((dx, dnu, dl, ds))
    return dz
    
