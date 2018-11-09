import define_rgbsample
import dispmodels
import fitDisp
import numpy as np
import emcee
from galpy.util import bovy_coords
from schwimmbad import MPIPool
import pickle
import corner
from tqdm import tqdm
import os
import sys

def cov_vxyz_to_galcenrect_single(cov_vxyz,Xsun=1.,Zsun=0.):
    dgc= np.sqrt(Xsun**2.+Zsun**2.)
    costheta, sintheta= Xsun/dgc, Zsun/dgc
    R = np.array([[costheta,0.,-sintheta],
                  [0.,1.,0.],
                  [sintheta,0.,costheta]])
    return np.dot(R.T,np.dot(cov_vxyz,R))   

def cov_galcenrect_to_galcencyl_single(cov_galcenrect, phi):
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    R = np.array([[cosphi, sinphi, 0.],
                 [-sinphi, cosphi, 0.],
                 [0., 0., 1.]])
    return np.dot(R, np.dot(cov_galcenrect, R.T))

def cov_vxyz_to_galcencyl(cov_vxyz, phi, Xsun=1., Zsun=0.):
    if len(np.shape(cov_vxyz)) == 3:
        cov_galcencyl = np.empty(np.shape(cov_vxyz))
        for i in tqdm(range(len(cov_vxyz))):
            cov_galcenrect = cov_vxyz_to_galcenrect_single(cov_vxyz[i], Xsun=Xsun, Zsun=Zsun)
            cov_galcencyl[i] = cov_galcenrect_to_galcencyl_single(cov_galcenrect, phi[i])
        return cov_galcencyl
    else:
        cov_galcenrect = cov_vxyz_to_galcenrect_single(cov_vxyz, Xsun=Xsun, Zsun=Zsun)
        cov_galcencyl = cov_galcenrect_to_galcencyl_single(cov_galcenrect, phi)
        return cov_galcencyl



agebins = np.arange(1.,14.,1.5)
fehbins = np.arange(-0.6,0.6,0.1)

tparams = []
tparams.extend([1/8.,])
tparams.extend(np.ones((len(agebins)*len(fehbins)))*np.log10(50.))
tparams.extend(np.ones((len(agebins)*len(fehbins)))*1)
tparams.extend(np.ones((len(agebins)*len(fehbins)))*1)
tparams.extend([1/8.,])
tparams.extend(np.ones((len(agebins)*len(fehbins)))*np.log10(50.))
tparams.extend(np.ones((len(agebins)*len(fehbins)))*1)
tparams.extend(np.ones((len(agebins)*len(fehbins)))*1)
tparams.extend(np.ones((len(agebins)*len(fehbins)))*0.001)

model_type = 'ellipsoid_all'
nparams = len(tparams)

nwalk=nparams*4
nit=500
ncut=300
threads = 32

if os.path.isfile('../sav/DR14_Gaia_astroNNAges_dists_galcencyl.npy'):
    with open('../sav/DR14_Gaia_astroNNAges_dists_galcencyl.npy', 'rb') as f:
        dat = np.load(f)
        Rpz = np.load(f)
        vRvTvz = np.load(f)
        cov_galcencyl = np.load(f)
else:
    dat = define_rgbsample.get_rgbsample(cuts = True, 
                                         add_dist=True, 
                                         astronn_dist = True,
                                         add_ages=True,
                                         rm_bad_dist=True,
                                         alternate_ages=True,
                                         distkey='pc',
                                         rmdups=True)
    dat['pc'] *= 1e-3
    dat['pc_error'] *= 1e-3
    XYZ, vxyz, cov_vxyz, Rpz, vRvTvz, cov_galcencyl = define_rgbsample.dat_to_galcen(dat, 
                                                                                      return_cov=True,
                                                                                      return_rphiz =True,
                                                                                      verbose =True,
                                                                                      ro = 8.,
                                                                                      vo = 220.,
                                                                                      zo = 0.025,
                                                                                      keys = ['ra', 
                                                                                              'dec', 
                                                                                              'pc', 
                                                                                              'pmra', 
                                                                                              'pmdec', 
                                                                                              'VHELIO_AVG'],
                                                                                      cov_keys =['pmra_error',
                                                                                                 'pmdec_error',
                                                                                                 'pmra_pmdec_corr',
                                                                                                 'pc_error',
                                                                                                 'VERR'],
                                                                                      parallax = False)
    
    with open('../sav/DR14_Gaia_astroNNAges_dists_galcencyl.npy', 'wb') as f:
        np.save(f, dat)
        np.save(f, Rpz)
        np.save(f, vRvTvz)
        np.save(f, cov_galcencyl)
        
mask = np.all(np.isfinite(vRvTvz), axis=1)

dat = dat[mask]
vRvTvz = vRvTvz[mask]
Rpz = Rpz[mask]
cov_galcencyl = cov_galcencyl[mask]
        
lo_samples = np.empty((nwalk*nit-ncut*nwalk,nparams))
hi_samples = np.empty((nwalk*nit-ncut*nwalk,nparams))
co_samples = np.empty((nwalk*nit-ncut*nwalk,nparams))
lo_med_z = np.empty([len(agebins)-1,len(fehbins)-1,5])
hi_med_z = np.empty([len(agebins)-1,len(fehbins)-1,5])
co_med_z = np.empty([len(agebins)-1,len(fehbins)-1,5])

lo_mask = (dat['AVG_ALPHAFE'] < define_rgbsample.alphaedge(dat['FE_H']))
hi_mask = (dat['AVG_ALPHAFE'] > define_rgbsample.alphaedge(dat['FE_H'])+0.04)
co_mask = np.ones(len(dat), dtype=bool)
for i in range(len(agebins)-1):
    for j in range(len(fehbins)-1):
        mask = [(dat['Age'] > agebins[i]) & 
                (dat['Age'] < agebins[i+1]) & 
                (dat['FE_H'] > fehbins[j]) & 
                (dat['FE_H'] < fehbins[j+1])]
        co_med_z[i,j] = np.nanpercentile(Rpz[:,2][mask]*8., [5,16,50,84,95])
        mask = [(dat['Age'] > agebins[i]) & 
                (dat['Age'] < agebins[i+1]) & 
                (dat['FE_H'] > fehbins[j]) & 
                (dat['FE_H'] < fehbins[j+1]) &
                (dat['AVG_ALPHAFE'] < define_rgbsample.alphaedge(dat['FE_H']))]
        lo_med_z[i,j] = np.nanpercentile(Rpz[:,2][mask]*8., [5,16,50,84,95])
        mask = [(dat['Age'] > agebins[i]) & 
                (dat['Age'] < agebins[i+1]) & 
                (dat['FE_H'] > fehbins[j]) & 
                (dat['FE_H'] < fehbins[j+1]) &
                (dat['AVG_ALPHAFE'] > define_rgbsample.alphaedge(dat['FE_H'])+0.04)]
        hi_med_z[i,j] = np.nanpercentile(Rpz[:,2][mask]*8., [5,16,50,84,95])
        

                   
med_z_arrs = [lo_med_z, hi_med_z, co_med_z]
sample_arrs = [lo_samples, hi_samples, co_samples]
labs = ['low a/Fe', 'high a/Fe', 'all stars']
        
masks = [lo_mask, hi_mask, co_mask]
for ii,m in enumerate(masks):
    print('fitting '+model_type.lower()+' for '+labs[ii])
    v_R = vRvTvz[:,0][m]*220.
    v_z = vRvTvz[:,2][m]*220.
    R = Rpz[:,0][m]*8.
    z = Rpz[:,2][m]*8.
    age = dat['Age'][m]
    feh = dat['FE_H'][m]
    cov_vRvTvz = cov_galcencyl[m]
    med_z = med_z_arrs[ii]
    init = tparams
    out = fitDisp.fitDisp_all(R, z, v_R, v_z, cov_vRvTvz, med_z, age, feh, agebins, fehbins, model_type, init,
                          mcmc=True, threads=threads, ncut=ncut, nit=nit, nwalk=nwalk)
    sys.stdout.flush()
    sys.stdout.write(str(np.median(out[1],axis=0))+' \n')
    sample_arrs[ii] = out[1]

            
with open('../sav/apdr14gaiadr2_astroNN_ages_distances_ellipsoid_fixedv0_monoage_samples.npy', 'wb') as f:
    np.save(f, lo_samples)
    np.save(f, hi_samples)
    np.save(f, co_samples)
    np.save(f, lo_med_z)
    np.save(f, hi_med_z)
    np.save(f, co_med_z)