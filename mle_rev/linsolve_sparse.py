import numpy as np
from scipy.sparse import issparse, csr_matrix, diags
from scipy.sparse.linalg import minres, aslinearoperator, LinearOperator

class AugmentedSystem(LinearOperator):
    def __init__(self, H, A):
        self.H2 = H
        self.A = A
        self.AT = A.T.tocsr()
        self.M1, self.N1 = self.H2.shape
        self.M2, self.N2 = self.A.shape
        self.shape = (self.M1 + self.M2, self.N1 + self.M2)
        self.dtype = self.H2.dtype
        self.diag = np.hstack((self.H2.diagonal(), 0))

    def _matvec(self, v):        
        v1 = v[0:self.N1]
        v2 = v[self.N1:]
        y = np.zeros(self.shape[0])
        y[0:self.N1] = self.H2.dot(v1) + self.AT.dot(v2)
        y[self.N1:] = self.A.dot(v1)
        return y

    def diagonal(self):
        return self.diag

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

def factor_aug(z, DPhival, G, A):
    r"""Set up augmented system and return.

    Parameters
    ----------
    z : (N+P+M+M,) ndarray
        Current iterate, z = (x, nu, l, s)
    DPhival : LinearOperator
        Jacobian of the variational inequality mapping
    G : (M, N) ndarray or sparse matrix
        Inequality constraints
    A : (P, N) ndarray or sparse matrix
        Equality constraints  

    Returns
    -------
    J : LinearOperator
        Augmented system
    
    """
    M, N = G.shape
    P, N = A.shape
    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks"""
    s = z[N+P+M:]

    """Sigma matrix"""
    SIG = diags(l/s, 0)

    """Convert A"""
    if not issparse(A):
        A = csr_matrix(A)

    """Convert G"""
    if not issparse(G):
        G = csr_matrix(G)

    # """Ensure linear operator"""
    # DPhival = aslinearoperator(DPhival)

    """Set up H"""
    H = DPhival + G.T.dot(SIG).dot(G)

    J = AugmentedSystem(H, A)
    return J

def solve_factorized_aug(z, Fval, LU, G, A):
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

    """LU is actually the augmented system J"""
    J = LU

    b1 = -rd - mydot(G.T, mydot(SIG, rp2)) + mydot(G.T, rc/s)
    b2 = -rp1
    b = np.hstack((b1, b2))

    """Prepare iterative solve via MINRES"""
    sign = np.zeros(N+P)
    sign[0:N/2] = 1.0
    sign[N/2:] = -1.0
    S = diags(sign, 0)
    # J_new = mydot(S, csr_matrix(J))
    def mv(v):
        x = J.dot(v)
        return S.dot(x)
        
    # J_new = LinearOperator(J.shape, matvec=mv)
    J_new = aslinearoperator(S).dot(J)
    
    b_new = mydot(S, b)

    dJ_new = S.dot(J.diagonal())
    # dJ_new = np.abs(J_new.diagonal())
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
