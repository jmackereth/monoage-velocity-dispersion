import math
import numpy
import statsmodels.api as sm
lowess= sm.nonparametric.lowess
import esutil
from galpy.util import bovy_coords, bovy_plot
import apogee.tools.read as apread
import isodist
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
#import fitDens 
import apogee.tools.read as apread
import gaia_tools
import gaia_tools.load as gload
from apogee.select import apogeeSelect
from astropy.io import fits
from astropy.table import Table, join
from numpy.lib.recfunctions import merge_arrays
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
from tqdm import tqdm
_R0= 8. # kpc
_Z0= 0.025 # kpc
_FEHTAG= 'FE_H'
_AFETAG= 'AVG_ALPHAFE'
_AFELABEL= r'$[\left([\mathrm{O+Mg+Si+S+Ca}]/5\right)/\mathrm{Fe}]$'
#catpath = '/Users/Ted/Documents/Work/apogee/catalogues/'
catpath = '/data5/astjmack/apogee/catalogues/'
selectFile= 'savs/selfunc-nospdata.sav'
if os.path.exists(selectFile):
    with open(selectFile,'rb') as savefile:
        apo= pickle.load(savefile)

savfile = open('/gal/astjmack/apogee/apogee-maps/py/paramsRGB_brokenexpflare_01dex2gyrbins_monoabundance_massgrid.dat', 'rb')
obj = pickle.load(savfile)
afebins, lfehbins, numbins, paramt, samples, massgrid, m_samplegrid = obj

def get_rgbtgassample(cuts = True, errorsamples=100, add_dist=False, distkey='BPG_meandist', disterrkey='BPG_diststd'):
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
    tgas_dat = gaia_tools.load.tgas()
    m1,m2,sep= gaia_tools.xmatch.xmatch(allStar,tgas_dat,colRA1='RA',colDec1='DEC', colRA2='ra', colDec2='dec', epoch1 =2000, epoch2=2015)
    dat = merge_arrays([allStar[m1], tgas_dat[m2]], asrecarray=True, flatten=True)
    dat = esutil.numpy_util.add_fields(dat, [('AVG_ALPHAFE', float),('MAPs_dist', float), ('MAPs_dist_err', float), ('err_samples', float, (errorsamples,7))])
    err_samp = [sample_error_ellipse(dat[i], size=errorsamples) for i in tqdm(range(len(dat)))]
    vhelio_samp = [sample_vrad_gauss(dat[i], size=errorsamples) for i in tqdm(range(len(dat)))]
    dat['err_samples'][:,:,6] = vhelio_samp
    dat['err_samples'][:,:,:5] = err_samp
    dat['AVG_ALPHAFE'] = avg_alphafe_dr13(dat)
    distarr = np.array([bayes_parallax_invert(dat[i], ageisafe=True, nsamp=errorsamples) for i in tqdm(range(len(dat)))])
    #dat['err_samples'][:,:,5] = distarr
    dat['MAPs_dist'], lo, hi = distarr[:,0], distarr[:,1], distarr[:,2]
    errors = np.dstack([dat['MAPs_dist']-lo, hi-dat['MAPs_dist']])
    dat['MAPs_dist_err'] = np.mean(errors, axis=2)
    if add_dist:
        allStar_tab = Table(data=dat)
        dists_tab = Table(data=dists)
        #join table
        tab = join(allStar_tab, dists_tab, keys='APOGEE_ID', uniq_col_name='{col_name}{table_name}', table_names=['','2'])
        dat = tab.as_array()
    dist_samp = [sample_dist_gauss(dat[i], size=errorsamples, distkey=distkey, disterrkey=disterrkey) for i in tqdm(range(len(dat)))]
    dat['err_samples'][:,:,5] = dist_samp
    return dat



def avg_alphafe_dr13(data):    
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
    if fehs.dtype == np.float32:
        if fehs < 0:
            return (0.12/-0.6)*fehs+0.03
        elif fehs >= 0:
            return 0.03
    edge = np.zeros(len(fehs))
    edge[fehs < 0] = (0.12/-0.6)*fehs[fehs < 0]+0.03
    edge[fehs >= 0] = 0.03
    return edge

def gauss(x, mu, sig):
    return np.exp(-1*((x-mu)**2.)/(2.*sig**2))

def density_model(age,feh,alpha='High', denstype='brokenexpflare', ageisafe=False ):
    if ageisafe:
        agebins = np.array(afebins)
    elif not ageisafe:
    	agebins = np.array(lagebins)
    fehbins = np.array(lfehbins)
    cagebins = (agebins[:-1]+agebins[1:])/2.
    cfehbins = (fehbins[:-1]+fehbins[1:])/2.
    i = np.argmin(np.fabs(cagebins-age))
    j = np.argmin(np.fabs(cfehbins-feh))
    if ageisafe == True:
        samp = samples
    else:
        if alpha== 'High':
            samp = hsamples
        elif alpha== 'Low':
            samp = lsamples
    params = np.median(samp[j,i], axis=1)
    tdensfunc = fitDens._setup_densfunc(denstype)
    densfunc = lambda x,y,z: tdensfunc(x,y,z,params=params)
    return densfunc
                                          
def bayes_parallax_invert(dat, ageisafe=False, forsampling=False, nsamp = 1000 ):
	point = dat
	if point['AVG_ALPHAFE'] > alphaedge(point['FE_H']):
		afe = 'High'
	else:
		afe = 'Low'
	if ageisafe:
		dmodel = density_model(point['AVG_ALPHAFE'], point['FE_H'], ageisafe=True)
	else:
		dmodel = density_model(point['Age'], point['FE_H'], alpha=afe)
	pis = np.linspace(0.00001,10.,1000)
	dists = 1./pis
	l = np.ones(len(dists))*point['l']
	b = np.ones(len(dists))*point['b']
	xyz = bovy_coords.lbd_to_XYZ(l, b, dists)
	rphiz = bovy_coords.XYZ_to_galcencyl(xyz[:,0], xyz[:,1], xyz[:,2], Xsun=8., Zsun=0.025)
	los_density = dmodel(rphiz[:,0], rphiz[:,1], rphiz[:,2])
	p = point['parallax']
	p_err = point['parallax_error']
	pdf = los_density*gauss(pis, p, p_err)
	interp = interp1d(np.cumsum(pdf)/np.sum(pdf),pis)
	sample = interp(np.random.rand(nsamp))
	p_mean, p_sig = np.mean(sample), np.std(sample)
	p_low = p_mean-p_sig
	p_hi = p_mean+p_sig
	return sample 
	
def cov_mat(data):
	ra_err_rad = data['ra_error']*(np.pi/(180.*3600.*1000.))
	dec_err_rad = data['dec_error']*(np.pi/(180.*3600.*1000.))
	A = ra_err_rad *           np.array([ra_err_rad,                          data['ra_dec_corr']*dec_err_rad,       data['ra_pmra_corr']*data['pmra_error'],       data['ra_pmdec_corr']*data['pmdec_error'],       data['ra_parallax_corr']*data['parallax_error']])
	B = dec_err_rad *          np.array([data['ra_dec_corr']*ra_err_rad,      dec_err_rad,                           data['dec_pmra_corr']*data['pmra_error'],      data['dec_pmdec_corr']*data['pmdec_error'],      data['dec_parallax_corr']*data['parallax_error']])
	C = data['pmra_error']*    np.array([data['ra_pmra_corr']*ra_err_rad,     data['dec_pmra_corr']*dec_err_rad,     data['pmra_error'],                            data['pmra_pmdec_corr']*data['pmdec_error'],     data['parallax_pmra_corr']*data['parallax_error']])
	D = data['pmdec_error']*   np.array([data['ra_pmdec_corr']*ra_err_rad,    data['dec_pmdec_corr']*dec_err_rad,    data['pmra_pmdec_corr']*data['pmra_error'],    data['pmdec_error'],                             data['parallax_pmdec_corr']*data['parallax_error']])
	E = data['parallax_error']*np.array([data['ra_parallax_corr']*ra_err_rad, data['dec_parallax_corr']*dec_err_rad, data['parallax_pmra_corr']*data['pmra_error'], data['parallax_pmdec_corr']*data['pmdec_error'], data['parallax_error']])
	cov =np.matrix([A,
					B,
					C,
					D,
					E])
	cov[0,1] = cov[1,0]
	cov[0,2] = cov[2,0]
	cov[0,3] = cov[3,0]
	cov[0,4] = cov[4,0]
	
	cov[1,2] = cov[2,1]
	cov[1,3] = cov[3,1]
	cov[1,4] = cov[4,1]
	
	cov[2,3] = cov[3,2]
	cov[2,4] = cov[4,2]
	
	cov[3,4] = cov[4,3]	
	return cov
    

def sample_error_ellipse(data, size=1000):
    c = cov_mat(data)
    means = np.array([data['ra']*(np.pi/180.), data['dec']*(np.pi/180.), data['pmra'], data['pmdec'], data['parallax']])
    multivar = multivariate_normal(mean=means, cov=np.array(c), allow_singular=True)
    rvs = multivar.rvs(size)
    return rvs  
    	
def sample_vrad_gauss(data, size=1000):
	pdf = gauss(np.linspace(-400,400,10000), data['VHELIO_AVG'], data['VERR'])
	interp = interp1d(np.cumsum(pdf)/np.sum(pdf),np.linspace(-400,400,10000))
	sample = interp(np.random.rand(size))
	return sample

def sample_dist_gauss(data, size=1000, distkey='BPG_meandist', disterrkey='BPG_diststd'):
	pdf = gauss(np.linspace(-400,400,10000), data[distkey], data[disterrkey])
	interp = interp1d(np.cumsum(pdf)/np.sum(pdf),np.linspace(-400,400,10000))
	sample = interp(np.random.rand(size))
	return sample