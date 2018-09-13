from scipy.optimize import fmin
from scipy.special import logsumexp
import numpy as np
import emcee
from emcee.utils import MPIPool
from schwimmbad import MPIPool
import dispmodels 
from tqdm import tqdm
import sys

_R0 = 8.

def fitDisp(R, z, v_R, v_z, cov_vRvTvz, med_z, mtype, init,
            mcmc=False,
            nofit=False,
            nit=300, 
            nwalk=200, 
            ncut=50, 
            threads=16,
            debug=False):
    if not nofit:
        dispmodel = setup_dispmodel(mtype)
        sys.stdout.write('inital parameters: '+str(init)+' \n')
        out = fmin(lambda x: mloglike(x, R, z, v_R, v_z, cov_vRvTvz, med_z, dispmodel, mtype, emcee=False), init, disp=True)
        sys.stdout.write('maximum likelihood params: '+str(out)+' \n')
        sys.stdout.flush()
    if mcmc:
        if nofit:
            out = init
        ndim = len(out)
        pos = np.fabs([out+1e-4*np.random.randn(ndim) for i in range(nwalk)])
        sampler = emcee.EnsembleSampler(nwalk, ndim, loglike, 
                                    args=(R, z, v_R, v_z, cov_vRvTvz, med_z, dispmodel, mtype), 
                                    threads=threads)
        for i, result in tqdm(enumerate(sampler.sample(pos, iterations=nit))):
            continue
        sampler.pool.terminate()
        sampler.pool.join()
        samples = sampler.chain[:,ncut:,:].reshape((-1,ndim))
        return out, samples
    return out

def fitDisp_qdf(vxvvsample, init,
            mcmc=False,
            nofit=False,
            nit=300, 
            nwalk=200, 
            ncut=50, 
            threads=16,
            debug=False):
    mtype='qdf'
    if not nofit:
        dispmodel = setup_dispmodel(mtype)
        sys.stdout.write('inital parameters: '+str(init)+' \n')
        out = fmin(lambda x: mloglike_qdf(x, vxvvsample, dispmodel, mtype, emcee=False), init, disp=True)
        sys.stdout.write('maximum likelihood params: '+str(out)+' \n')
        sys.stdout.flush()
    if mcmc:
        if nofit:
            out = init
        ndim = len(out)
        pos = np.fabs([out+1e-4*np.random.randn(ndim) for i in range(nwalk)])
        sampler = emcee.EnsembleSampler(nwalk, ndim, loglike_qdf, 
                                    args=(vxvvsample, dispmodel, mtype), 
                                    threads=threads)
        for i, result in tqdm(enumerate(sampler.sample(pos, iterations=nit))):
            continue
        sampler.pool.terminate()
        sampler.pool.join()
        samples = sampler.chain[:,ncut:,:].reshape((-1,ndim))
        return out, samples
    return out

def loglike(params, dataR, dataZ, datav_R, datav_z, datacov_vRvTvz, datamed_z, dispmodel, mtype, emcee=True):
    if not check_prior(params, mtype, datamed_z):
        if emcee:
            return -np.inf
        return -np.finfo(np.dtype(np.float64)).max
    mod = lambda v_R, v_z, R, z, cov_vRvTvz, datamed_z: dispmodel(v_R, v_z, R, z, cov_vRvTvz, datamed_z, params=params)
    lp = mod(datav_R, datav_z, dataR, dataZ, datacov_vRvTvz, datamed_z)
    if not np.isfinite(lp):
        if emcee:
            return -np.inf
        return -np.finfo(np.dtype(np.float64)).max
    return lp

def loglike_qdf(params, vxvvsample, dispmodel, mtype, emcee=True):
    if not check_prior(params, mtype, None):
        if emcee:
            return -np.inf
        return -np.finfo(np.dtype(np.float64)).max
    mod = lambda samp: dispmodel(samp, params=params)
    lp = mod(vxvvsample)
    if not np.isfinite(lp):
        if emcee:
            return -np.inf
        return -np.finfo(np.dtype(np.float64)).max
    return lp

def mloglike(*args,**kwargs):
    return -loglike(*args,**kwargs)

def mloglike_qdf(*args,**kwargs):
    return -loglike_qdf(*args,**kwargs)

def setup_dispmodel(model):
    if model.lower() == 'gaussian1d':
        return dispmodels.gaussian_1d
    elif model.lower() == 'gaussian':
        return dispmodels.gaussian_fixedv0
    elif model.lower() == 'gaussianexprquadz':
        return dispmodels.gaussian_expR_quadz_fixedv0
    elif model.lower() == 'gaussianexprquadzvaryvo':
        return dispmodels.gaussian_expR_quadz
    elif model.lower() == 'gaussianexprexpz':
        return dispmodels.gaussian_expR_expz_fixedv0
    elif model.lower() == 'gaussianexprexpzvaryvo':
        return dispmodels.gaussian_expR_expz
    elif model.lower() == 'ellipsoid':
        return dispmodels.ellipsoid
    elif model.lower() == 'ellipsoidvaryvo':
        return dispmodels.ellipsoid_varying_v0
    elif model.lower() == 'ellipsoidvo_fr':
        return dispmodels.ellipsoid_v0_R
    elif model.lower() == 'qdf':
        return dispmodels.quasiisothermal
    else:
        raise NameError('Bad model type string!')
    

def check_prior(params, model, med_z):
    if model.lower() == 'gaussian1d':
        if params[0] > 200.: return False
        if params[0] < 1.: return False
        return True
    if model.lower() == 'gaussian':
        if params[0] > 200.: return False
        if params[1] > 200.: return False
        if params[0] < 1.: return False
        if params[1] < 1.: return False
        if params[2] > 1. : return False
        if params[2] < 0. : return False
        return True
    if model.lower() == 'gaussianexprquadz':
        sigmaR_z0 = (params[2]*(-med_z)**2+params[3]*(-med_z)+10**params[1])
        sigmaz_z0 = (params[6]*(-med_z)**2+params[7]*(-med_z)+10**params[5])
        if sigmaR_z0 > 200.:return False
        if sigmaR_z0 < 1.:return False
        if sigmaz_z0 > 200.:return False
        if sigmaz_z0 < 1.:return False
        if params[8] > 1.:return False
        if params[8] < 0.:return False
        return True
    if model.lower() == 'gaussianexprquadzvaryvo':
        sigmaR_z0 = (params[2]*(-med_z)**2+params[3]*(-med_z)+10**params[1])
        sigmaz_z0 = (params[6]*(-med_z)**2+params[7]*(-med_z)+10**params[5])
        if sigmaR_z0 > 200.:return False
        if sigmaR_z0 < 1.:return False
        if sigmaz_z0 > 200.:return False
        if sigmaz_z0 < 1.:return False
        if params[10] > 1.:return False
        if params[10] < 0.:return False
        return True
    if model.lower() == 'gaussianexprexpz':
        sigmaR_z0 = 10**params[2]*np.exp(-params[1]*(-med_z))
        sigmaz_z0 = 10**params[5]*np.exp(-params[4]*(-med_z))
        if sigmaR_z0 > 200.:return False
        if sigmaR_z0 < 1.:return False
        if sigmaz_z0 > 200.:return False
        if sigmaz_z0 < 1.:return False
        if params[6] > 1.:return False
        if params[6] < 0.:return False
        return True
    if model.lower() == 'gaussianexprexpzvaryvo':
        sigmaR_z0 = 10**params[2]*np.exp(-params[1]*(-med_z))
        sigmaz_z0 = 10**params[5]*np.exp(-params[4]*(-med_z))
        if sigmaR_z0 > 200.:return False
        if sigmaR_z0 < 1.:return False
        if sigmaz_z0 > 200.:return False
        if sigmaz_z0 < 1.:return False
        if params[8] > 1.:return False
        if params[8] < 0.:return False
        return True
    if model.lower() == 'ellipsoid':
        sigmaR_z0 = (params[2]*(-med_z)**2+params[3]*(-med_z)+10**params[1])
        sigmaz_z0 = (params[6]*(-med_z)**2+params[7]*(-med_z)+10**params[5])
        if sigmaR_z0 > 200.:return False
        if sigmaR_z0 < 1.:return False
        if sigmaz_z0 > 200.:return False
        if sigmaz_z0 < 1.:return False
        if 10**params[1] > 200.: return False
        if 10**params[1] < 1.: return False
        if 10**params[5] > 200.: return False
        if 10**params[5] < 1.: return False 
        if params[8] > 10: return False
        if params[8] < -10: return False
        if params[9] > 10: return False
        if params[9] < -10: return False
        if params[10] > 1.:return False
        if params[10] < 0.:return False
        return True
    if model.lower() == 'ellipsoidvaryvo':
        sigmaR_z0 = (params[2]*(-med_z)**2+params[3]*(-med_z)+10**params[1])
        sigmaz_z0 = (params[6]*(-med_z)**2+params[7]*(-med_z)+10**params[5])
        if sigmaR_z0 > 200.:return False
        if sigmaR_z0 < 1.:return False
        if sigmaz_z0 > 200.:return False
        if sigmaz_z0 < 1.:return False
        if 10**params[1] > 200.: return False
        if 10**params[1] < 1.: return False
        if 10**params[5] > 200.: return False
        if 10**params[5] < 1.: return False 
        if params[8] > 10: return False
        if params[8] < -10: return False
        if params[9] > 10: return False
        if params[9] < -10: return False
        if params[10] > 100: return False
        if params[10] < -100: return False
        if params[11] > 100: return False
        if params[11] < -100: return False
        if params[12] > 1.:return False
        if params[12] < 0.:return False
        return True
    if model.lower() == 'ellipsoidvo_fr':
        sigmaR_z0 = (params[2]*(-med_z)**2+params[3]*(-med_z)+10**params[1])
        sigmaz_z0 = (params[6]*(-med_z)**2+params[7]*(-med_z)+10**params[5])
        if sigmaR_z0 > 200.:return False
        if sigmaR_z0 < 1.:return False
        if sigmaz_z0 > 200.:return False
        if sigmaz_z0 < 1.:return False
        if 10**params[1] > 200.: return False
        if 10**params[1] < 1.: return False
        if 10**params[5] > 200.: return False
        if 10**params[5] < 1.: return False 
        if params[8] > 10: return False
        if params[8] < -10: return False
        if params[9] > 10: return False
        if params[9] < -10: return False
        if params[11] > 100: return False
        if params[11] < -100: return False
        if params[13] > 100: return False
        if params[13] < -100: return False
        if params[14] > 1.:return False
        if params[14] < 0.:return False
        return True
    if model.lower() == 'qdf':
        if params[0] < 0.: return False
        if params[1] < 0.: return False
        if params[2] < 0.: return False
        return True

def lnprior(theta, med_z):
    if len(theta) == 9:
        hsigmaR, sigmaR, a_R, b_R, hsigmaz, sigmaz, a_z, b_z, c = theta
        sigmaR_z0 = (a_R*(-med_z)**2+b_R*(-med_z)+10**sigmaR)
        sigmaz_z0 = (a_z*(-med_z)**2+b_z*(-med_z)+10**sigmaz)
        if 1. < sigmaR_z0 < 200. and \
            1. < sigmaz_z0 < 200. and \
            0. < c < 1.:
            return 0.0
        return -np.inf
    else:
        sigmaR, sigmaz, c = theta
        if 1. < sigmaR < 200. and \
            1. < sigmaz < 200. and \
            0. < c < 1.:
            return 0.0
        return -np.inf

def lnprob(params, v_R, v_z, R, z, med_z):
    logprior = lnprior(params, med_z)
    if not np.isfinite(logprior):
        return -np.inf
    p = gaussian_expR_quadz_fixedv0(v_R, v_z, R, z, med_z, params=params)
    if not np.isfinite(p):
        return -np.inf
    return logprior + p

        
        