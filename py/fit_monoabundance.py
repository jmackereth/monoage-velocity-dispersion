import define_rgbtgassample
import dispmodels
import fitDisp
import numpy as np
import emcee
from schwimmbad import MPIPool
import pickle
import corner
import os
import sys

nwalk=100
nit=500
ncut=300
threads = 32

agebins = np.arange(-0.1,0.35,0.05)
fehbins = np.arange(-0.6,0.6,0.1)

model_type = 'ellipsoid'

if os.path.isfile('../sav/DR14_TGAS_Ages_galcencyl.npy'):
    with open('../sav/DR14_TGAS_Ages_galcencyl.npy', 'rb') as f:
        dat = np.load(f)
        Rpz = np.load(f)
        vRvTvz = np.load(f)
        cov_galcencyl = np.load(f)
else:
    dat = define_rgbtgassample.get_rgbtgassample(cuts = True, 
                                                 add_dist=True, 
                                                 add_ages=True,
                                                 rm_bad_dist=True,
                                                 distkey='BPG_meandist', 
                                                 disterrkey='BPG_diststd')
    XYZ, vxvyvz, cov_vxyz = define_rgbtgassample.dat_to_rectgal(dat, 
                                                                return_cov=True,
                                                                keys = ['ra', 
                                                                        'dec', 
                                                                        'BPG_meandist', 
                                                                        'pmra', 
                                                                        'pmdec', 
                                                                        'VHELIO_AVG'],
                                                                cov_keys =['pmra_error',
                                                                           'pmdec_error',
                                                                           'pmra_pmdec_corr',
                                                                           'BPG_diststd',
                                                                           'VERR'])
    ro, vo, zo = 8., 220., 0.025
    vsolar= np.array([-10.1,4.0,6.7])
    vsun= np.array([0.,1.,0.,])+vsolar/vo
    Rpz = bovy_coords.XYZ_to_galcencyl(XYZ[:,0],XYZ[:,1],XYZ[:,2],Zsun=0.025/8.)
    vRvTvz = bovy_coords.vxvyvz_to_galcencyl(vxvyvz[:,0], vxvyvz[:,1], vxvyvz[:,2], Rpz[:,0], Rpz[:,1], Rpz[:,2],
                                             vsun=vsun,
                                             Xsun=1.,
                                             Zsun=zo/ro,
                                             galcen=True)
    cov_galcencyl = cov_vxyz_to_galcencyl(cov_vxyz, Rpz[:,1], Xsun=1., Zsun=0.025/8.)
    with open('../sav/DR14_TGAS_Ages_galcencyl.npy', 'wb') as f:
        np.save(f, dat)
        np.save(f, Rpz)
        np.save(f, vRvTvz)
        np.save(f, cov_galcencyl)
        
samples = np.empty([len(agebins)-1,len(fehbins)-1,nwalk*nit-ncut*nwalk,11])
med_zs = np.empty([len(agebins)-1,len(fehbins)-1,5])


for i in range(len(agebins)-1):
    for j in range(len(fehbins)-1):
        m = [(dat['AVG_ALPHAFE'] > agebins[i]) & \
                   (dat['AVG_ALPHAFE'] < agebins[i+1]) & \
                   (dat['FE_H'] > fehbins[j]) & \
                   (dat['FE_H'] < fehbins[j+1])]
        if len(vRvTvz[:,0][m]) < 100:
            samples[i,j] = np.ones([nwalk*nit-ncut*nwalk,11])*np.nan
            med_zs[i,j] = np.ones(5)*np.nan
            continue
        v_R = vRvTvz[:,0][m]*220.
        v_z = vRvTvz[:,2][m]*220.
        R = Rpz[:,0][m]*8.
        z = Rpz[:,2][m]*8.
        cov_vRvTvz = cov_galcencyl[m]
        med_z = np.nanmedian(np.fabs(z))
        med_zs[i,j] = np.nanpercentile(np.fabs(z), [10,25,50,75,90])
        init = [1/8.,np.log10(30.),0.,0.,1/2.5,np.log10(20.),0.,0.,0.,0.,0.01]
        sys.stdout.write(str(agebins[i])+' < [a/Fe] < '+str(agebins[i+1])+' '+str(round(fehbins[j],2))+' < [Fe/H] < '+str(round(fehbins[j+1],2))+' \n')
        sys.stdout.write(str(len(vRvTvz[:,0][m]))+' \n')
        out = fitDisp.fitDisp(R, z, v_R, v_z, cov_vRvTvz, med_z, model_type, init,
                              mcmc=True, threads=threads, ncut=ncut, nit=nit, nwalk=nwalk)
        sys.stdout.flush()
        samples[i,j] = out[1]

            
with open('../sav/samples_monoabundance_ellipsoid.npy', 'wb') as f:
    np.save(f, samples)
    np.save(f, med_zs)