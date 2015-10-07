import numpy as np

###############################################################################
# Objective, Gradient, and Hessian
###############################################################################

def f(z, C):
    N=z.shape[0]
    x=z[0:N/2]
    y=z[N/2:]
    q=np.exp(y)    
    W=x[:,np.newaxis]*q[np.newaxis,:]
    Z=W+W.transpose()
    return -1.0*np.sum(C*np.log(Z))+np.sum(x)+np.sum(C*y[np.newaxis,:])

def F(z, C):
    N=z.shape[0]
    x=z[0:N/2]
    y=z[N/2:]
    q=np.exp(y)
    C_sym=C+C.transpose()
    W=x[:,np.newaxis]*q[np.newaxis,:]
    Z=W+W.transpose()   
    Fx=-1.0*np.sum(C_sym*q[np.newaxis, :]/Z, axis=1)+1.0
    Fy= -1.0*np.sum(C_sym*W.transpose()/Z, axis=1)+np.sum(C, axis=0)
    return np.hstack((Fx, -1.0*Fy))

def DF(z, C):
    N=z.shape[0]
    x=z[0:N/2]
    y=z[N/2:]
    
    q=np.exp(y)

    C_sym=C+C.transpose()
    W=x[:,np.newaxis]*q[np.newaxis,:]
    Wt=W.transpose()
    Z=W+Wt

    Z2=Z**2
    Q=q[:,np.newaxis]*q[np.newaxis,:]

    dxx=np.sum(C_sym*(q**2)[np.newaxis,:]/Z2, axis=1)
    DxDxf= np.diag(dxx)+C_sym*Q/Z2

    dxy=np.sum(C_sym*(x*q)[:,np.newaxis]*q[np.newaxis,:]/Z2, axis=0)
    DyDxf=-1.0*C_sym*q[np.newaxis,:]/Z + C_sym*(W*q[np.newaxis,:])/Z2+np.diag(dxy)
    
    DxDyf=DyDxf.transpose()
    
    Dyy1=-1.0*C_sym*W/Z
    Dyy2=C_sym*W**2/Z2
    dyy=np.sum(Dyy1, axis=0)+np.sum(Dyy2, axis=0)
    
    DyDyf=np.diag(dyy)+C_sym*W*Wt/Z2

    J=np.zeros((N, N))
    J[0:N/2, 0:N/2]=DxDxf
    J[0:N/2, N/2:]=DyDxf
    J[N/2:, 0:N/2]=-1.0*DxDyf
    J[N/2:, N/2:]=-1.0*DyDyf
    
    return J

