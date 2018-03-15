import numpy as np
from scipy.special import logsumexp

_R0 = 8.
_z0 = 0.025

def gaussian_1d(v_R, v_z, R, z, med_z, params=[30.]):
    sigmaR = params[0]
    vo = 0.
    A_R = (1./(sigmaR*np.sqrt(2*np.pi)))
    E_R = (-(v_R-vo)**2)/(2*sigmaR**2)
    p = A_R*np.exp(E_R)
    logp = np.log(p)
    logp = np.sum(logp[np.isfinite(logp)])
    return logp

def gaussian_fixedv0(v_R, v_z, R, z, med_z, params=[np.log10(30.),np.log10(30.),0.1]):
    sigmaR, sigmaz, contfrac = params
    sigmaR, sigmaz = 10**sigmaR, 10**sigmaz
    vo = 0.
    contA = (contfrac)*(1./(100*np.sqrt(2*np.pi)))
    contE_R = (-(v_R-vo)**2/(2*100**2))
    contE_z = (-(v_z-vo)**2/(2*100**2))
    A_R = (1-contfrac)*(1./(sigmaR*np.sqrt(2*np.pi)))
    A_z = (1-contfrac)*(1./(sigmaz*np.sqrt(2*np.pi)))
    E_R = (-(v_R-vo)**2)/(2*sigmaR**2)
    E_z = (-(v_z-vo)**2)/(2*sigmaz**2)
    Es = np.dstack([E_R, E_z, contE_R, contE_z])[0]
    As = np.dstack([A_R, A_z, contA, contA])[0]
    logp = logsumexp(Es, b=As, axis=1)
    logp = np.sum(logp)
    return logp
    
def gaussian_expR_quadz_fixedv0(v_R, v_z, R, z, 
                                cov_vRvTvz, med_z, 
                                params=[1/8.,np.log10(50.),1.,1.,1/8.,np.log10(50.),1.,1.,0.01], return_each=False):
    vo = 0.
    sigmacont = 100.
    h_sigmaR, sigmaR, a_R, b_R, h_sigmaz, sigmaz, a_z, b_z, contfrac = params
    h_sigmaR, h_sigmaz = 1/h_sigmaR, 1/h_sigmaz
    sigmaR, sigmaz = 10**sigmaR, 10**sigmaz
    sigmacontR, sigmacontz = np.sqrt(sigmacont**2+cov_vRvTvz[:,0,0]), np.sqrt(sigmacont**2+cov_vRvTvz[:,2,2])
    z = np.fabs(z)
    sigma_fRz_R = np.sqrt(((a_R*(z-med_z)**2+b_R*(z-med_z)+sigmaR)*(np.exp(-1*(R-_R0)/h_sigmaR)))**2+cov_vRvTvz[:,0,0])
    A_R = (1-contfrac)*(1./(sigma_fRz_R*np.sqrt(2*np.pi)))
    E_R =  (-(v_R-vo)**2)/(2*sigma_fRz_R**2)
    sigma_fRz_z = np.sqrt(((a_z*(z-med_z)**2+b_z*(z-med_z)+sigmaz)*(np.exp(-1*(R-_R0)/h_sigmaz)))**2+cov_vRvTvz[:,2,2])
    A_z = (1-contfrac)*(1./(sigma_fRz_z*np.sqrt(2*np.pi)))
    E_z =  (-(v_z-vo)**2)/(2*sigma_fRz_z**2)
    contA_R = (contfrac)*(1./(sigmacontR*np.sqrt(2*np.pi)))
    contA_z = (contfrac)*(1./(sigmacontz*np.sqrt(2*np.pi)))
    contE_R = (-(v_R-vo)**2/(2*sigmacontR**2))
    contE_z = (-(v_z-vo)**2/(2*sigmacontz**2))
    As = np.dstack([A_R, A_z, contA_R, contA_z])[0]
    Es = np.dstack([E_R, E_z, contE_R, contE_z])[0]
    logp = logsumexp(Es, b=As, axis=1)
    if return_each:
        return logp
    logp = np.sum(logp)
    return logp

def gaussian_expR_quadz(v_R, v_z, R, z, 
                        cov_vRvTvz, med_z, 
                        params=[1/8.,np.log10(50.),1.,1.,1/8.,np.log10(50.),1.,1.,0.,0.,0.01]):
    vo = 0.
    sigmacont=100.
    h_sigmaR, sigmaR, a_R, b_R, h_sigmaz, sigmaz, a_z, b_z, v_Ro, v_zo, contfrac = params
    h_sigmaR, h_sigmaz = 1/h_sigmaR, 1/h_sigmaz
    sigmaR, sigmaz = 10**sigmaR, 10**sigmaz
    sigmacontR, sigmacontz = np.sqrt(sigmacont**2+cov_vRvTvz[:,0,0]), np.sqrt(sigmacont**2+cov_vRvTvz[:,2,2])
    z = np.fabs(z)
    sigma_fRz_R = np.sqrt(((a_R*(z-med_z)**2+b_R*(z-med_z)+sigmaR)*(np.exp(-1*(R-_R0)/h_sigmaR)))**2+cov_vRvTvz[:,0,0])
    A_R = (1-contfrac)*(1./(sigma_fRz_R*np.sqrt(2*np.pi)))
    E_R =  (-(v_R-v_Ro)**2)/(2*sigma_fRz_R**2)
    sigma_fRz_z = np.sqrt(((a_z*(z-med_z)**2+b_z*(z-med_z)+sigmaz)*(np.exp(-1*(R-_R0)/h_sigmaz)))**2+cov_vRvTvz[:,2,2])
    A_z = (1-contfrac)*(1./(sigma_fRz_z*np.sqrt(2*np.pi)))
    E_z =  (-(v_z-v_zo)**2)/(2*sigma_fRz_z**2)
    contA_R = (contfrac)*(1./(sigmacontR*np.sqrt(2*np.pi)))
    contA_z = (contfrac)*(1./(sigmacontz*np.sqrt(2*np.pi)))
    contE_R = (-(v_R-vo)**2/(2*sigmacontR**2))
    contE_z = (-(v_z-vo)**2/(2*sigmacontz**2))
    As = np.dstack([A_R, A_z, contA_R, contA_z])[0]
    Es = np.dstack([E_R, E_z, contE_R, contE_z])[0]
    logp = logsumexp(Es, b=As, axis=1)
    logp = np.sum(logp)
    return logp

def gaussian_expR_expz_fixedv0(v_R, v_z, R, z, med_z, params=[1/8.,1/8.,np.log10(50.),1/8.,1/8.,np.log10(50.),0.01]):
    vo = 0.
    hRsigmaR, hzsigmaR, sigmaR, hRsigmaz, hzsigmaz, sigmaz, contfrac = params
    sigmaR, sigmaz = 10**sigmaR, 10**sigmaz
    sigmacont = 200.
    z = np.fabs(z)
    sigma_fRz_R = sigmaR*np.exp(-hRsigmaR*(R-_R0)-hzsigmaR*(z-med_z))
    sigma_fRz_z = sigmaz*np.exp(-hRsigmaz*(R-_R0)-hzsigmaz*(z-med_z))
    A_R = (1-contfrac)*(1./(sigma_fRz_R*np.sqrt(2*np.pi)))
    E_R =  (-(v_R-vo)**2)/(2*sigma_fRz_R**2)
    A_z = (1-contfrac)*(1./(sigma_fRz_z*np.sqrt(2*np.pi)))
    E_z =  (-(v_z-vo)**2)/(2*sigma_fRz_z**2)
    contA = (contfrac)*(1./(sigmacont*np.sqrt(2*np.pi)))
    contE_R = (-(v_R-vo)**2/(2*sigmacont**2))
    contE_z = (-(v_z-vo)**2/(2*sigmacont**2))
    As = np.dstack([A_R, A_z, np.ones(len(v_R))*contA, np.ones(len(v_R))*contA])[0]
    Es = np.dstack([E_R, E_z, contE_R, contE_z])[0]
    logp = logsumexp(Es, b=As, axis=1)
    logp = np.sum(logp)
    return logp

def gaussian_expR_expz(v_R, v_z, R, z, med_z, params=[1/8.,1/8.,np.log10(50.),1/8.,1/8.,np.log10(50.),0.,0.,0.01]):
    vo = 0.
    hRsigmaR, hzsigmaR, sigmaR, hRsigmaz, hzsigmaz, sigmaz, v_Ro, v_zo, contfrac = params
    sigmaR, sigmaz = 10**sigmaR, 10**sigmaz
    z = np.fabs(z)
    sigma_fRz_R = sigmaR*np.exp(-hRsigmaR*(R-_R0)-hzsigmaR*(z-med_z))
    sigma_fRz_z = sigmaz*np.exp(-hRsigmaz*(R-_R0)-hzsigmaz*(z-med_z))
    A_R = (1-contfrac)*(1./(sigma_fRz_R*np.sqrt(2*np.pi)))
    E_R =  (-(v_R-v_Ro)**2)/(2*sigma_fRz_R**2)
    A_z = (1-contfrac)*(1./(sigma_fRz_z*np.sqrt(2*np.pi)))
    E_z =  (-(v_z-v_zo)**2)/(2*sigma_fRz_z**2)
    contA = (contfrac)*(1./(100*np.sqrt(2*np.pi)))
    contE_R = (-(v_R-vo)**2/(2*100**2))
    contE_z = (-(v_z-vo)**2/(2*100**2))
    As = np.dstack([A_R, A_z, np.ones(len(v_R))*contA, np.ones(len(v_R))*contA])[0]
    Es = np.dstack([E_R, E_z, contE_R, contE_z])[0]
    logp = logsumexp(Es, b=As, axis=1)
    logp = np.sum(logp)
    return logp

def ellipsoid(v_R, v_z, R, z, 
              cov_vRvTvz, med_z, 
              params=[1/8.,np.log10(50.),1.,1.,1/8.,np.log10(50.),1.,1.,0.,0.,0.01]):
    vo = 0.
    sigmacont = 100.
    h_sigmaR, sigmaR, a_R, b_R, h_sigmaz, sigmaz, a_z, b_z, alpha_0,alpha_1, contfrac = params
    h_sigmaR, h_sigmaz = 1/h_sigmaR, 1/h_sigmaz
    sigmaR, sigmaz = 10**sigmaR, 10**sigmaz
    z = np.fabs(z)
    sigma_fRz_R = (a_R*(z-med_z)**2+b_R*(z-med_z)+sigmaR)*(np.exp(-1*(R-_R0)/h_sigmaR))
    sigma_fRz_z = (a_z*(z-med_z)**2+b_z*(z-med_z)+sigmaz)*(np.exp(-1*(R-_R0)/h_sigmaz))
    tana= alpha_0+alpha_1*z/R #+params[11]*(z/R)**2.
    sig2rz= (sigma_fRz_R**2.-sigma_fRz_z**2.)*tana/(1.-tana**2.)
    #Do likelihood
    out= 0.
    for ii in range(len(v_R)):
        vv= np.array([v_R[ii],v_z[ii]])
        VV= np.array([[sigma_fRz_R[ii]**2.+cov_vRvTvz[ii,0,0],
                          sig2rz[ii]+cov_vRvTvz[ii,0,2]],
                         [sig2rz[ii]+cov_vRvTvz[ii,0,2],
                          sigma_fRz_z[ii]**2.+cov_vRvTvz[ii,2,2]]])
        outVV= np.array([[sigmacont**2.+cov_vRvTvz[ii,0,0],
                             cov_vRvTvz[ii,0,2]],
                            [cov_vRvTvz[ii,0,2],
                            sigmacont**2.+cov_vRvTvz[ii,2,2]]])
        #print VV, outVV, numpy.linalg.det(VV), numpy.linalg.det(outVV)
        detVV= np.linalg.det(VV)
        if detVV < 0.: return -np.finfo(np.dtype(np.float64)).max
        out += np.log(contfrac/np.sqrt(np.linalg.det(outVV))\
                            *np.exp(-0.5*np.dot(vv,
                                                np.dot(np.linalg.inv(outVV),vv)))
                        +(1.-contfrac)/np.sqrt(detVV)
                            *np.exp(-0.5*np.dot(vv,
                                                np.dot(np.linalg.inv(VV),vv))))
    return out    
    
    