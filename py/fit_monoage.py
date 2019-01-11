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
from optparse import OptionParser

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

def fit_monoage(args, options):
    nwalk=100
    nit=500
    ncut=300
    threads = options.nthreads

    agebins = np.arange(1.,14.,1.5)
    fehbins = np.arange(-0.6,0.6,0.1)

    model_type = options.model_type

    print('fitting '+model_type.lower())
    
    if model_type == 'ellipsoid':
        nparams = 11
    elif model_type == 'ellipsoidvaryvo':
        nparams = 13
    else: 
        raise NameError('Bad model type string (new models require editing of the code!)')



    if os.path.isfile('../sav/DR14_Gaia_astroNNAges_dists_galcencyl_sgrvsun.npy'):
        with open('../sav/DR14_Gaia_astroNNAges_dists_galcencyl_sgrvsun.npy', 'rb') as f:
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

        with open('../sav/DR14_Gaia_astroNNAges_dists_galcencyl_sgrvsun.npy', 'wb') as f:
            np.save(f, dat)
            np.save(f, Rpz)
            np.save(f, vRvTvz)
            np.save(f, cov_galcencyl)

    mask = np.all(np.isfinite(vRvTvz), axis=1)

    dat = dat[mask]
    vRvTvz = vRvTvz[mask]
    Rpz = Rpz[mask]
    cov_galcencyl = cov_galcencyl[mask]

    lo_samples = np.empty([len(agebins)-1,len(fehbins)-1,nwalk*nit-ncut*nwalk,nparams])
    hi_samples = np.empty([len(agebins)-1,len(fehbins)-1,nwalk*nit-ncut*nwalk,nparams])
    co_samples = np.empty([len(agebins)-1,len(fehbins)-1,nwalk*nit-ncut*nwalk,nparams])
    lo_med_z = np.empty([len(agebins)-1,len(fehbins)-1,5])
    hi_med_z = np.empty([len(agebins)-1,len(fehbins)-1,5])
    co_med_z = np.empty([len(agebins)-1,len(fehbins)-1,5])
    med_z_arrs = [lo_med_z, hi_med_z, co_med_z]
    sample_arrs = [lo_samples, hi_samples, co_samples]


    for i in range(len(agebins)-1):
        for j in range(len(fehbins)-1):
            lo_mask = [(dat['Age'] > agebins[i]) & \
                       (dat['Age'] < agebins[i+1]) & \
                       (dat['FE_H'] > fehbins[j]) & \
                       (dat['FE_H'] < fehbins[j+1]) & \
                       (dat['AVG_ALPHAFE'] < define_rgbsample.alphaedge(dat['FE_H']))]
            hi_mask = [(dat['Age'] > agebins[i]) & \
                       (dat['Age'] < agebins[i+1]) & \
                       (dat['FE_H'] > fehbins[j]) & \
                       (dat['FE_H'] < fehbins[j+1]) & \
                       (dat['AVG_ALPHAFE'] > define_rgbsample.alphaedge(dat['FE_H'])+0.04)]
            co_mask = [(dat['Age'] > agebins[i]) & \
                       (dat['Age'] < agebins[i+1]) & \
                       (dat['FE_H'] > fehbins[j]) & \
                       (dat['FE_H'] < fehbins[j+1])]
            print('%i stars' % len(dat[co_mask]))
            masks = [lo_mask, hi_mask, co_mask]
            labs = ['low [a/Fe]', 'high [a/Fe]', 'all [a/Fe]']
            for ii,m in enumerate(masks):
                if len(vRvTvz[:,0][m]) < 80:
                    sample_arrs[ii][i,j] = np.ones([nwalk*nit-ncut*nwalk,nparams])*np.nan
                    med_z_arrs[ii][i,j] = np.ones(5)*np.nan
                    continue
                v_R = vRvTvz[:,0][m]*220.
                v_z = vRvTvz[:,2][m]*220.
                R = Rpz[:,0][m]*8.
                z = Rpz[:,2][m]*8.
                cov_vRvTvz = cov_galcencyl[m]
                med_z = np.nanmedian(np.fabs(z))
                med_z_arrs[ii][i,j] = np.nanpercentile(np.fabs(z), [10,25,50,75,90])
                if model_type == 'ellipsoid':
                    init = [1/8.,np.log10(30.),0.,0.,1/2.5,np.log10(20.),0.,0.,0.,0.,0.01]
                elif model_type == 'ellipsoidvaryvo':
                    init = [1/8.,np.log10(30.),1.,1.,1/2.5,np.log10(20.),1.,1.,0.,0.,0.,0.,0.01]
                else: 
                    raise NameError('Bad model type string (new models require editing of the code!)')
                sys.stdout.write(str(agebins[i])+' < Age < '+str(agebins[i+1])+' '+str(round(fehbins[j],2))+' < [Fe/H] < '+str(round(fehbins[j+1],2))+' \n')
                sys.stdout.write(labs[ii]+' \n')
                sys.stdout.write(str(len(vRvTvz[:,0][m]))+' \n')
                out = fitDisp.fitDisp(R, z, v_R, v_z, cov_vRvTvz, med_z, model_type, init,
                                      mcmc=True, threads=threads, ncut=ncut, nit=nit, nwalk=nwalk)
                sys.stdout.flush()
                sys.stdout.write(str(np.median(out[1],axis=0))+' \n')
                sample_arrs[ii][i,j] = out[1]


    with open('../sav/'+str(args[0]), 'wb') as f:
        np.save(f, lo_samples)
        np.save(f, hi_samples)
        np.save(f, co_samples)
        np.save(f, lo_med_z)
        np.save(f, hi_med_z)
        np.save(f, co_med_z)

def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the output will be saved to"
    parser = OptionParser(usage=usage)
    # Distances at which to calculate the effective selection function
    parser.add_option("--model-type",dest='model_type',default='ellipsoid',type='str',
                      help="Which model to fit")
    parser.add_option("--nthreads",dest='nthreads',default=32,type='float',
                      help="number of threads to use with emcee")
    
    return parser

if __name__ == '__main__':
    parser= get_options()
    options, args= parser.parse_args()
    fit_monoage(args,options)