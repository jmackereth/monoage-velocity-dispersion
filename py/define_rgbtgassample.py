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
#import fitDens 
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
_R0= 8. # kpc
_Z0= 0.025 # kpc
_FEHTAG= 'FE_H'
_AFETAG= 'AVG_ALPHAFE'
_AFELABEL= r'$[\left([\mathrm{O+Mg+Si+S+Ca}]/5\right)/\mathrm{Fe}]$'
#catpath = '/Users/Ted/Documents/Work/apogee/catalogues/'
catpath = '/data5/astjmack/apogee/catalogues/'
selectFile= 'savs/selfunc-nospdata.sav'
agecat = '../sav/ages_dr14.txt'
corr_agecat = '../sav/corrected_ages_dr14.npy'
if os.path.exists(selectFile):
    with open(selectFile,'rb') as savefile:
        apo= pickle.load(savefile)


def get_rgbtgassample(cuts = True, 
                      add_dist=False, 
                      add_ages=False,
                      rm_bad_dist=True,
                      notgas = False,
                      alternate_ages = True,
                      distkey='BPG_meandist', 
                      disterrkey='BPG_diststd'):
    """
    Get a clean sample of dr13 APOGEE data with TGAS parallaxes and PMs
    ---
    INPUT:
        None
    OUTPUT:
        Clean rgb sample with added parallaxes
    HISTORY:
        Started - Mackereth 17/07/17 
    """
    if cuts:
        allStar = apread.allStar(rmcommissioning=False, 
                                 exclude_star_bad=True, 
                                 exclude_star_warn=True, 
                                 main=False,
                                 ak=False, 
                                 adddist=False)
        allStar = allStar[(allStar['LOGG'] > 1.8)&(allStar['LOGG'] < 3.0)]
    else:
        allStar = apread.allStar()
    dists = fits.open('/gal/astjmack/apogee/catalogues/apogee_distances-DR14.fits')[1].data
    if not notgas:
        tgas_dat = gload.tgas()
        m1,m2,sep= xmatch.xmatch(allStar,tgas_dat,
                                            colRA1='RA',
                                            colDec1='DEC', 
                                            colRA2='ra', 
                                            colDec2='dec', 
                                            epoch1 =2000, 
                                            epoch2=2015, swap=True)
        dat = merge_arrays([allStar[m1], tgas_dat[m2]], asrecarray=True, flatten=True)
    else:
        dat = allStar
    dat = esutil.numpy_util.add_fields(dat, [('AVG_ALPHAFE', float)])
    dat['AVG_ALPHAFE'] = avg_alphafe(dat)
    if add_dist:
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
    if add_ages:
        allStar_tab = Table(data=dat)
        if alternate_ages:
            ages = np.load(corr_agecat)
            ages_tab = Table(data=ages)
        else:
            ages = np.genfromtxt(agecat, names=True, dtype=None)
            ages_tab = Table(data=ages)
            ages_tab.rename_column('2MASS_ID', 'APOGEE_ID')
        tab = join(allStar_tab, ages_tab, 
                   keys='APOGEE_ID', 
                   uniq_col_name='{col_name}{table_name}', 
                   table_names=['','_marie_ages'])
        dat = tab.as_array()
    return dat

def dat_to_rectgal(dat, 
                  return_cov=True,
                  return_rphiz =True,
                  keys = ['ra', 'dec', 'BPG_meandist', 'pmra', 'pmdec', 'VHELIO_AVG'],
                  cov_keys = ['pmra_error','pmdec_error','pmra_pmdec_corr','BPG_diststd','VERR']):
    vxvv = np.dstack([dat[keys[i]] for i in range(len(keys))])[0]
    ro, vo, zo = 8., 220., 0.025
    ra, dec= vxvv[:,0], vxvv[:,1]
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True)
    pmra, pmdec= vxvv[:,3], vxvv[:,4]
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec,degree=True)
    d, vlos= vxvv[:,2], vxvv[:,5]
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
    if return_cov == True:
        cov_pmradec = np.array([[[dat[cov_keys[0]][i]**2, 
                                  dat[cov_keys[2]][i]*dat[cov_keys[0]][i]*dat[cov_keys[1]][i]],
                                 [dat[cov_keys[2]][i]*dat[cov_keys[0]][i]*dat[cov_keys[1]][i], 
                                  dat[cov_keys[1]][i]**2]] for i in range(len(dat))])
        cov_pmllbb =  bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmradec, vxvv[:,0], vxvv[:,1],
                                                            degree=True,
                                                            epoch='J2015')
        cov_vxyz = bovy_coords.cov_dvrpmllbb_to_vxyz(vxvv[:,2], 
                                                     dat[cov_keys[3]], 
                                                     dat[cov_keys[4]], 
                                                     pmllpmbb[:,0], 
                                                     pmllpmbb[:,1], 
                                                     cov_pmllbb, 
                                                     lb[:,0], 
                                                     lb[:,1])
        return XYZ, vxyz, cov_vxyz
    return XYZ, vxyz

def sample_pos_vel(dat, cov_vxyz,
                   nsamp = 100,
                   keys = ['ra', 'dec', 'BPG_meandist', 'pmra', 'pmdec', 'VHELIO_AVG'],
                   cov_keys = ['pmra_error','pmdec_error','pmra_pmdec_corr','BPG_diststd','VERR']):

    ro, vo, zo = 8., 220., 0.025
    vxvv = np.dstack([dat[keys[i]] for i in range(len(keys))])[0]
    ra, dec= vxvv[:,0], vxvv[:,1]
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True)
    vsolar= np.array([-10.1,4.0,6.7])
    vsun= np.array([0.,1.,0.,])+vsolar/vo
    pmra, pmdec= vxvv[:,3], vxvv[:,4]
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec,degree=True)
    d, vlos= vxvv[:,2], vxvv[:,5]
    rectgal= bovy_coords.sphergal_to_rectgal(lb[:,0],lb[:,1],d,vlos,pmllpmbb[:,0], pmllpmbb[:,1],degree=True)
    X = rectgal[:,0]/ro
    Y = rectgal[:,1]/ro
    Z = rectgal[:,2]/ro
    vx = rectgal[:,3]/vo
    vy = rectgal[:,4]/vo
    vz = rectgal[:,5]/vo
    dataRphiz = np.empty([len(dat),nsamp, 3])
    datavxyz = np.empty([len(dat),nsamp, 3])
    datavRvTvz = np.empty([len(dat),nsamp, 3])
    for i in tqdm(range(len(dat))):
        if not np.isfinite(cov_vxyz[i]).all():
            dataRphiZ[i] = np.ones([nsamp,3])*np.nan
            datavxyz[i] = np.ones([nsamp,3])*np.nan
            datavRphiZ[i] = np.ones([nsamp,3])*np.nan
            continue
        d = norm(loc=vxvv[:,2][i], scale=dat['BPG_diststd'][i])
        ds = d.rvs(nsamp)
        ls = np.ones(nsamp)*lb[:,0][i]
        bs = np.ones(nsamp)*lb[:,1][i]
        txyz = bovy_coords.lbd_to_XYZ(ls,bs,ds, degree=True)
        t_X = txyz[:,0]/ro
        t_Y = txyz[:,1]/ro
        t_Z = txyz[:,2]/ro
        rpz = bovy_coords.XYZ_to_galcencyl(t_X,t_Y,t_Z,Zsun=zo/ro)
        dataRphiz[i] = rpz
        t = multivariate_normal(mean=[vx[i], vy[i], vz[i]], cov = cov_vxyz[i]/220.)
        ts = t.rvs(nsamp)
        datavxyz[i] = ts
        ndat = np.ones(nsamp)
        t_vx = ts[:,0]
        t_vy = ts[:,1]
        t_vz = ts[:,2]
        vsun= np.array([0.,1.,0.,])+vsolar/vo
        datavRvTvz[i] = bovy_coords.vxvyvz_to_galcencyl(t_vx,t_vy,t_vz,rpz[:,0],rpz[:,1],rpz[:,2],
                                                        vsun=vsun,
                                                        Xsun=1.,
                                                        Zsun=zo/ro,
                                                        galcen=True)
    return dataRphiz, datavRvTvz



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
        if fehs < 0:
            return (0.12/-0.6)*fehs+0.03
        elif fehs >= 0:
            return 0.03
    edge = np.zeros(len(fehs))
    edge[fehs < 0] = (0.12/-0.6)*fehs[fehs < 0]+0.03
    edge[fehs >= 0] = 0.03
    return edge

