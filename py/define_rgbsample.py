import math
import numpy
import statsmodels.api as sm
lowess= sm.nonparametric.lowess
import esutil
from galpy.util import bovy_coords, bovy_plot
import apogee.tools.read as apread
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import apogee.tools.read as apread
import gaia_tools.load as gload
from gaia_tools import xmatch
from apogee.select import apogeeSelect
from astropy.io import fits
from astropy.table import Table, join
from numpy.lib.recfunctions import merge_arrays
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm
from apogee.tools.read import remove_duplicates
_R0= 8. # kpc
_Z0= 0.025 # kpc
_FEHTAG= 'FE_H'
_AFETAG= 'AVG_ALPHAFE'
_AFELABEL= r'$[\left([\mathrm{O+Mg+Si+S+Ca}]/5\right)/\mathrm{Fe}]$'
catpath = '/data5/astjmack/apogee/catalogues/'
agecat = '../sav/ages_dr14.txt'
corr_agecat = '../sav/astroNNBayes_ages_goodDR14.npy'
astronn_dists = '../sav/apogee_dr14_nn_dist_new.fits'

def get_rgbsample(cuts = True, 
                      add_dist=False,
                      astronn_dist = True,
                      add_ages=False,
                      rm_bad_dist=True,
                      no_gaia = False,
                      distkey = 'BPG_meandist',
                      verbose = True,
                      alternate_ages = False,
                      rmdups =True):
    """
    Get a clean sample of dr14 APOGEE data with Gaia (!) parallaxes and PMs
    ---
    INPUT:
        None
    OUTPUT:
        Clean rgb sample with added parallaxes and PMs
    HISTORY:
        Started - Mackereth 24/04/18
    """
    if cuts:
        allStar = apread.allStar(rmcommissioning=True, 
                                 exclude_star_bad=True, 
                                 exclude_star_warn=True,
                                 main=False,
                                 ak=True, 
                                 rmdups = True,
                                 adddist=False)
        allStar = allStar[(allStar['LOGG'] > 1.8)&(allStar['LOGG'] < 3.0)]
    else:
        allStar = apread.allStar(rmcommissioning=True,
                                 main=False,
                                 ak=True, 
                                 rmdups = True,
                                 adddist=False)
    if verbose:
        print('%i Objects meeting quality cuts in APOGEE DR14' % len(allStar))
    if not no_gaia:
        gaia_xmatch = fits.open('../sav/allStar_l31c2_GaiaDR2_crossmatch_withpms.fits')
        gaia_xmatch = gaia_xmatch[1].data
        gaia_xmatch = Table(data=gaia_xmatch)
        allStar_tab = Table(data=allStar)
        tab = join(allStar_tab, gaia_xmatch, keys='APOGEE_ID', uniq_col_name='{col_name}{table_name}', 
                   table_names=['','_xmatch'])
        dat = tab.as_array()
        if verbose:
            print('%i Matched Objects in Gaia DR2' % len(dat))
    else:
        dat = allStar
    dat = esutil.numpy_util.add_fields(dat, [('AVG_ALPHAFE', float)])
    dat['AVG_ALPHAFE'] = avg_alphafe(dat)
    if add_dist:
        if astronn_dist:
            dists= fits.open(astronn_dists)[1].data
            distkey = 'pc'
        else:
            dists = fits.open('/gal/astjmack/apogee/catalogues/apogee_distances-DR14.fits')[1].data
        allStar_tab = Table(data=dat)
        dists_tab = Table(data=dists)
        #join table
        tab = join(allStar_tab, dists_tab, 
                   keys='APOGEE_ID', 
                   uniq_col_name='{col_name}{table_name}', 
                   table_names=['','_dist_table'])
        dat = tab.as_array()
        if rm_bad_dist:
            mask = np.isfinite(dat[distkey])
            dat=dat[mask]
        if verbose:
            print('%i Matched Objects in APOGEE distance VAC' % len(dat))
    if add_ages:
        allStar_tab = Table(data=dat)
        if alternate_ages:
            ages = np.load(corr_agecat)
            ages_tab = Table(data=ages)
            ages_tab.rename_column('astroNN_age', 'Age')
        else:
            ages = np.genfromtxt(agecat, names=True, dtype=None)
            ages_tab = Table(data=ages)
            ages_tab.rename_column('2MASS_ID', 'APOGEE_ID')
        tab = join(allStar_tab, ages_tab, 
                   keys='APOGEE_ID', 
                   uniq_col_name='{col_name}{table_name}', 
                   table_names=['','_ages'])
        dat = tab.as_array()
        if rmdups:
            print('removing duplicates...')
            dat = remove_duplicates(dat)
        if verbose:
            print('%i Matched Objects in Age Catalogue' % len(dat))
            
    return dat

def dat_to_galcen(dat, 
                  return_cov=True,
                  return_rphiz =True,
                  verbose =False,
                  ro = 8.,
                  vo = 220.,
                  zo = 0.025,
                  keys = ['ra', 'dec', 'BPG_meandist', 'pmra', 'pmdec', 'VHELIO_AVG'],
                  cov_keys = ['pmra_error','pmdec_error','pmra_pmdec_corr','BPG_diststd','VERR'],
                  parallax = False):
    vxvv = np.dstack([dat[keys[i]] for i in range(len(keys))])[0]
    ra, dec= vxvv[:,0], vxvv[:,1]
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True)
    pmra, pmdec= vxvv[:,3], vxvv[:,4]
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec,degree=True)
    d, vlos= vxvv[:,2], vxvv[:,5]
    if parallax:
        d = 1./d
    rectgal= bovy_coords.sphergal_to_rectgal(lb[:,0],lb[:,1],d,vlos,pmllpmbb[:,0], pmllpmbb[:,1],degree=True)
    vsolar= np.array([-11.1,12.24,7.25])
    vsun= np.array([0.,1.,0.,])+vsolar/vo
    X = rectgal[:,0]/ro
    Y = rectgal[:,1]/ro
    Z = rectgal[:,2]/ro
    vx = rectgal[:,3]/vo
    vy = rectgal[:,4]/vo
    vz = rectgal[:,5]/vo
    XYZ = np.dstack([X, Y, Z])[0]
    vxyz = np.dstack([vx,vy,vz])[0]
    if return_rphiz:
        Rpz = bovy_coords.XYZ_to_galcencyl(XYZ[:,0],XYZ[:,1],XYZ[:,2],Zsun=zo/ro)
        vRvTvz = bovy_coords.vxvyvz_to_galcencyl(vxyz[:,0], vxyz[:,1], vxyz[:,2], Rpz[:,0], Rpz[:,1], Rpz[:,2],
                                                                    vsun=vsun,
                                                                    Xsun=1.,
                                                                    Zsun=zo/ro,
                                                                    galcen=True)
    if return_cov == True:
        cov_pmradec = np.array([[[dat[cov_keys[0]][i]**2, 
                                  dat[cov_keys[2]][i]*dat[cov_keys[0]][i]*dat[cov_keys[1]][i]],
                                 [dat[cov_keys[2]][i]*dat[cov_keys[0]][i]*dat[cov_keys[1]][i], 
                                  dat[cov_keys[1]][i]**2]] for i in range(len(dat))])
        if verbose:
            print('propagating covariance in pmra pmdec -> pmll pmbb')
        cov_pmllbb =  bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmradec, vxvv[:,0], vxvv[:,1],
                                                            degree=True,
                                                            epoch='J2015')
        if verbose:
            print('propagating covariance in pmll pmbb -> vx vy vz')
        cov_vxyz = bovy_coords.cov_dvrpmllbb_to_vxyz(vxvv[:,2], 
                                                     dat[cov_keys[3]], 
                                                     dat[cov_keys[4]], 
                                                     pmllpmbb[:,0], 
                                                     pmllpmbb[:,1], 
                                                     cov_pmllbb, 
                                                     lb[:,0], 
                                                     lb[:,1])
        if not return_rphiz:
            return XYZ, vxyz, cov_vxyz
        
        if verbose:
            print('propagating covariance in vx vy vz -> vR vT vz')
        cov_galcencyl = bovy_coords.cov_vxyz_to_galcencyl(cov_vxyz, Rpz[:,1], Xsun=1., Zsun=zo/ro)
        return XYZ, vxyz, cov_vxyz, Rpz, vRvTvz, cov_galcencyl
    if not return_rphiz:
        return XYZ, vxyz
    return XYZ, vxyz, Rpz, vRvTvz

def obs_to_galcen(ra, dec, dist, pmra, pmdec, rv, pmra_err, pmdec_err, pmra_pmdec_corr, dist_err, rv_err,
                  return_cov = True,
                  verbose = True,
                  return_rphiz = True,
                  ro = 8.,
                  vo = 220.,
                  zo = 0.025,
                  parallax=False):
    vxvv = np.dstack([ra,dec,dist,pmra,pmdec,rv])[0]
    ra, dec= vxvv[:,0], vxvv[:,1]
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True)
    pmra, pmdec= vxvv[:,3], vxvv[:,4]
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec,degree=True)
    d, vlos= vxvv[:,2], vxvv[:,5]
    if parallax:
        d = 1./d
    rectgal= bovy_coords.sphergal_to_rectgal(lb[:,0],lb[:,1],d,vlos,pmllpmbb[:,0], pmllpmbb[:,1],degree=True)
    vsolar= np.array([-10.1,4.0,6.7])
    vsun= np.array([0.,1.,0.,])+vsolar/vo
    X = rectgal[:,0]/ro
    Y = rectgal[:,1]/ro
    Z = rectgal[:,2]/ro
    vx = rectgal[:,3]/vo
    vy = rectgal[:,4]/vo
    vz = rectgal[:,5]/vo
    XYZ = np.dstack([X, Y, Z])[0]
    vxyz = np.dstack([vx,vy,vz])[0]
    if return_rphiz:
        Rpz = bovy_coords.XYZ_to_galcencyl(XYZ[:,0],XYZ[:,1],XYZ[:,2],Zsun=zo/ro)
        vRvTvz = bovy_coords.vxvyvz_to_galcencyl(vxyz[:,0], vxyz[:,1], vxyz[:,2], Rpz[:,0], Rpz[:,1], Rpz[:,2],
                                                                    vsun=vsun,
                                                                    Xsun=1.,
                                                                    Zsun=zo/ro,
                                                                    galcen=True)
    if return_cov == True:
        cov_pmradec = np.empty([len(pmra_err), 2,2])
        cov_pmradec[:,0,0] = pmra_err**2
        cov_pmradec[:,1,1] = pmdec_err**2
        cov_pmradec[:,0,1] = pmra_pmdec_corr*pmra_err*pmdec_err
        cov_pmradec[:,1,0] = pmra_pmdec_corr*pmra_err*pmdec_err
        if verbose:
            print('propagating covariance in pmra pmdec -> pmll pmbb')
        cov_pmllbb =  bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmradec, vxvv[:,0], vxvv[:,1],
                                                            degree=True,
                                                            epoch='J2015')
        if verbose:
            print('propagating covariance in pmll pmbb -> vx vy vz')
        cov_vxyz = bovy_coords.cov_dvrpmllbb_to_vxyz(vxvv[:,2], 
                                                     dist_err, 
                                                     rv_err, 
                                                     pmllpmbb[:,0], 
                                                     pmllpmbb[:,1], 
                                                     cov_pmllbb, 
                                                     lb[:,0], 
                                                     lb[:,1])
        if not return_rphiz:
            return XYZ, vxyz, cov_vxyz
        
        if verbose:
            print('propagating covariance in vx vy vz -> vR vT vz')
        cov_galcencyl = bovy_coords.cov_vxyz_to_galcencyl(cov_vxyz, Rpz[:,1], Xsun=1., Zsun=zo/ro)
        return XYZ, vxyz, cov_vxyz, Rpz, vRvTvz, cov_galcencyl
    if not return_rphiz:
        return XYZ, vxyz
    return XYZ, vxyz, Rpz, vRvTvz

def avg_alphafe(data):    
    weight_o= np.ones(len(data))
    weight_s= np.ones(len(data))
    weight_si= np.ones(len(data))
    weight_ca= np.ones(len(data))
    weight_mg= np.ones(len(data))
    weight_o[data['O_FE'] == -9999.0]= 0.
    weight_s[data['S_FE'] == -9999.0]= 0.
    weight_si[data['SI_FE'] == -9999.0]= 0.
    weight_ca[data['CA_FE'] == -9999.0]= 0.
    weight_mg[data['MG_FE'] == -9999.0]= 0.
    return (weight_o*data['O_FE']+weight_s*data['S_FE']
            +weight_si*data['SI_FE']+weight_ca*data['CA_FE']
            +weight_mg*data['MG_FE'])/(weight_o+weight_s
                                      +weight_si+weight_ca
                                      +weight_mg)


def alphaedge(fehs):
    if not hasattr(fehs, '__iter__'):
        if fehs < 0.1:
            return (0.12/-0.6)*fehs+0.03
        elif fehs >= 0.1:
            return 0.03
    edge = np.zeros(len(fehs))
    edge[fehs < 0.2] = (0.12/-0.6)*fehs[fehs < 0.2]+0.05
    edge[fehs >= 0.2] = (0.12/-0.6)*0.2+0.05
    return edge


