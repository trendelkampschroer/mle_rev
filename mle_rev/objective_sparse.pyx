import numpy as np
import scipy.sparse

cimport numpy as np

from libc.math cimport exp

ctypedef np.int_t DTYPE_INT_t
ctypedef np.float_t DTYPE_FLOAT_t

def f(z, C):
    N=z.shape[0]
    x=z[0:N/2]
    y=z[N/2:]
    q=np.exp(y)    
    W=x[:,np.newaxis]*q[np.newaxis,:]
    Z=W+W.transpose()
    return -1.0*np.sum(C*np.log(Z))+np.sum(x)+np.sum(C*y[np.newaxis,:])

def F(z, C):
    r"""Monotone mapping for the reversible MLE problem.

    Parameters
    ----------
    z : (2*M,) ndarray
        Point at which to evaluate mapping, z=(x, y)
    C : (M, M) scipy.sparse matrix
        Count matrix of reversible chain

    Returns
    -------
    Fval : (2*M,) ndarray
        Value of the mapping at z
    
    """
    cdef size_t M, k, l, j
    cdef double cs_kj, ekj
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] x, y
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] data, c, Fval
    # cdef np.ndarray[np.int, ndim=1] indices, indptr
    
    M = C.shape[0]
    c = C.sum(axis=0).A1
    Cs = C + C.T
    Cs = Cs.tocsr()

    x = z[0:M]
    y = z[M:]

    data = Cs.data
    indptr = Cs.indptr
    indices = Cs.indices

    Fval = np.zeros(2*M,)

    """Loop over rows of Cs"""
    for k in range(M):
        Fval[k] += 1.0
        Fval[k+M] -= c[k]

        """Loop over nonzero entries in row of Cs"""
        for l in range(indptr[k], indptr[k+1]):
            """Column index of current element"""
            j = indices[l]
            """Current element of Cs at (k, j)"""
            cs_kj = data[l]
            """Exponential of difference"""
            ekj = exp(y[k]-y[j])
            """Update Fx"""
            Fval[k] += -cs_kj/(x[k]+x[j]*ekj)
            """Update Fy"""
            Fval[k+M] -= -cs_kj*x[j]/(x[k]/ekj + x[j])               

    return Fval    
    
def DF(z, C):
    r"""Jacobian of the monotone mapping.

    Parameters
    ----------
    z : (2*M,) ndarray
        Point at which to evaluate mapping, z=(x, y)
    C : (M, M) scipy.sparse matrix
        Count matrix of reversible chain

    Returns
    -------
    DFval : (2*M, 2*M) scipy.sparse matrix
        Value of the Jacobian at z
    
    """
    cdef size_t M, k, l, j
    cdef double cs_kj, ekj, tmp1, tmp2
    cdef np.ndarray[np.float_t, ndim=1] x, y
    cdef np.ndarray[np.float_t, ndim=1] data, data_Hxx, data_Hyy, data_Hyx
    cdef np.ndarray[np.float_t, ndim=1] diag_Dxx, diag_Dyy, diag_Dyx
    cdef np.ndarray[np.int_t, ndim=1] indices, indptr
    
    M = C.shape[0]
    Cs = C + C.T
    Cs = Cs.tocsr()

    x = z[0:M]
    y = z[M:]

    data = Cs.data
    indptr = Cs.indptr
    indices = Cs.indices

    """All subblocks DF_ij can be written as follows, DF_ij = H_ij +
    D_ij. H_ij has the same sparsity structure as C+C.T and D_ij is a
    diagonal matrix, i, j \in {x, y}

    """
    data_Hxx = np.zeros_like(data)
    data_Hyx = np.zeros_like(data)
    data_Hyy = np.zeros_like(data)

    diag_Dxx = np.zeros(M)
    diag_Dyx = np.zeros(M)
    diag_Dyy = np.zeros(M)

    """Loop over rows of Cs"""
    for k in range(M):
        """Loop over nonzero entries in row of Cs"""
        for l in range(indptr[k], indptr[k+1]):
            """Column index of current element"""
            j = indices[l]
            """Current element of Cs at (k, j)"""
            cs_kj = data[l]

            ekj = np.exp(y[k]-y[j])

            tmp1 = cs_kj/((x[k]+x[j]*ekj)*(x[k]/ekj+x[j]))
            tmp2 = cs_kj/(x[k] + x[j]*ekj)**2

            data_Hxx[l] = tmp1
            diag_Dxx[k] += tmp2

            data_Hyy[l] = tmp1*x[k]*x[j]
            diag_Dyy[k] -= tmp1*x[k]*x[j]

            data_Hyx[l] = -tmp1*x[k]
            diag_Dyx[k] += tmp1*x[j]

    Hxx = scipy.sparse.csr_matrix((data_Hxx, indices, indptr), shape=(M, M))
    Dxx = scipy.sparse.diags(diag_Dxx, 0)
    DFxx = Hxx + Dxx

    Hyy = scipy.sparse.csr_matrix((data_Hyy, indices, indptr), shape=(M, M))
    Dyy = scipy.sparse.diags(diag_Dyy, 0)
    DFyy = Hyy + Dyy

    Hyx = scipy.sparse.csr_matrix((data_Hyx, indices, indptr), shape=(M, M))
    Dyx = scipy.sparse.diags(diag_Dyx, 0)
    DFyx = Hyx + Dyx
    
    DFval = scipy.sparse.bmat([[DFxx, DFyx.T], [-1.0*DFyx, -1.0*DFyy]])
    return DFval.toarray()

