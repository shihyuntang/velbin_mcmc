import numpy as np
import sys, os, argparse, ast, itertools

from astropy.table import Table
from scipy import optimize as opti
from multiprocessing import Pool

import corner, emcee

from  function.binaries import *
from  function.mcmc_func import *

def mcmc_setting(args):
    """you can change the value here for MCMC walkers

    Returns:
        nwalkers int: number of walkers to use
        nstep int: number of steps for each walker to walk
        nburn int: number of burn in steps to drop off while plotting 
                        the corner plot 
                        (so your final sampled number are nstep-nburn)
        
        In general, 100 walkers are pretty good for this work.
        If you still see burn in in your corner plot, try to increase 
        the nstep and the nburn
    """

    # nwalkers, nstep, nburn = 100, 25000, 20000  # for RV only...
    # nwalkers, nstep, nburn = 50, 30000, 20000  # for RV only...
    nwalkers, nstep, nburn = args.walker, args.steps, args.burnin  # for RV only...

    # mcmc results save file name
    fnrv      = "rv_mcmcsave.h5"
    fnpmra    = "pmra_mcmcsave.h5"
    fnpmdec   = "pmdec_mcmcsave.h5"
    
    return nwalkers, nstep, nburn, fnrv, fnpmra, fnpmdec

def lnprior_rv(x, max_vmean, max_vmean_f):
    """log prior for the RV data.
    Change the values here to narrow down the range your walkers walk, 
    and hope to speed up the MCMC.
    But, do not give a too tight constrain for getting a biased distribution

    Args:
        x (list): stored par.
        max_vmean (float): mean rv you inputted
        max_vmean_f (float): mean field rv you inputted

    Returns:
        float: log prior probability
    """
    # vmean, vdisp, fbin, vmean_f, vdisp_f = x
    vmean, vdisp, vmean_f, vdisp_f = x

    # uniform priors
    # only change: vmean_range_xxxx
    vmean_range_cluster = 50
    vmean_range_field = 50
    uni_rv_mean   = (vmean > (max_vmean-vmean_range_cluster)) & (vmean < (max_vmean+vmean_range_cluster)) 
    uni_rv_mean_f = (vmean_f > (max_vmean_f-vmean_range_field)) & (vmean_f < (max_vmean_f+vmean_range_field))
    
    # only change: vdisp_range_xxxx
    vdisp_range_cluster = 50.0
    vdisp_range_field = 50.0
    uni_rv_disp   = (vdisp   > 0.0) & (vdisp   < vdisp_range_cluster)
    uni_rv_disp_f = (vdisp_f > 0.0) & (vdisp_f < vdisp_range_field)
    
    # you might not need to change this, binary fraction...
    # uni_rv_fbin   = (fbin    > 0.0)              & (fbin < 1.)

    uni_all = uni_rv_mean & uni_rv_disp & uni_rv_mean_f & uni_rv_disp_f # & uni_rv_fbin
    if not uni_all:
        return -np.inf
    else:
        return 0

def lnprior_pm(x, max_vmean, max_vmean_f):
    """log prior for the PM data.
    Change the values here to narrow down the range your walkers walk, 
    and hope to speed up the MCMC.
    But, do not give a too tight constrain for getting a biased distrebution

    Args:
        x (list): stored par.
        max_vmean (float): mean pm(ra or dec) you inputted
        max_vmean_f (float): mean field pm(ra or dec) you inputted

    Returns:
        float: log prior probability
    """
    pm_mean, pm_disp, pm_mean_f, pm_disp_f = x

    # uniform priors
    # only change: vmean_range
    vmean_range_cluster = 50
    vmean_range_field = 50
    uni_pm_mean   = (pm_mean > (max_vmean-vmean_range_cluster) ) & (pm_mean < (max_vmean+vmean_range_cluster)) 
    uni_pm_mean_f = (pm_mean_f > (max_vmean_f-vmean_range_field)) & (pm_mean_f < (max_vmean_f+vmean_range_field))
    
    # only change: vdisp_range 
    vdisp_range_cluster = 50
    vdisp_range_field = 50
    uni_pm_disp   = (pm_disp   > 0.0) & (pm_disp   < vdisp_range_cluster)
    uni_pm_disp_f = (pm_disp_f > 0.0) & (pm_disp_f < vdisp_range_field)

    uni_all = uni_pm_mean & uni_pm_disp & uni_pm_mean_f & uni_pm_disp_f

    if not uni_all:
        return -np.inf
    else:
        return 0


def read_input(args):
    """read in your observation data under the ./Input dir

    Returns:
        velocity float: rv (km/s)
        e_velocity float: rv error (km/s)
        mass float: stellar mass (M_sun)
        pmra float: proper motion ra (mas/yr)
        pmdec float: proper motion dec (mas/yr)
        epmra float: proper motion ra err (mas/yr)
        epmdec float: proper motion dec err (mas/yr)
    """
    
    tb_org = Table.read('./Input/{}'.format(args.filename))
    
    # column name to be read
    rvuseN = 'dr2_radial_velocity'
    rvuseNerr = 'dr2_radial_velocity_error'
    # rvuseN = 'RV_Jackson'
    # rvuseNerr = 'e_RV_Jackson'
    
    # if RV use -9999 as nan value, clean our -9999
    tb_with_rv = tb_org[ (tb_org[rvuseN]>-9000) ]
    
    rv = np.array(tb_with_rv[rvuseN])
    erv   = np.array(tb_with_rv[rvuseNerr])
    mass  = np.array(tb_with_rv['Mass'])
    
    pmra     = np.array(tb_org['pmra'])
    epmra    = np.array(tb_org['pmra_error'])
    pmdec    = np.array(tb_org['pmdec'])
    epmdec   = np.array(tb_org['pmdec_error'])
    
    return rv, erv, mass, pmra, pmdec, epmra, epmdec


# ==============================================================================
# ==============================================================================
# you probably don't need to touch stuff below... 
# if you don't know what you are doing...
def lnprob_rv(x, lnlike, max_vmean, max_vmean_f, fbin):
    """ln probability for rv 

    Args:
        x (list): stored par.
        lnlike (func): ln likelihood function
        max_vmean (float): mean rv you inputted
        max_vmean_f (float): mean field rv you inputted

    Returns:
        float: posterior probability
    """
    lp      = lnprior_rv(x, max_vmean, max_vmean_f)

    if not np.isfinite(lp):
        return -np.inf
    
    # lnlike(x) will call BinaryFit-->individual_log_likelihood, and use fbin
    return lp + lnlike(x, fbin)


def lnprob_pm(x, pm, sig_pm, max_vmean, max_vmean_f):
    """ln probability for pm

    Args:
        x (list): stored par.
        lnlike (func): ln likelihood function
        max_vmean (float): mean pm(ra or dec) you inputted
        max_vmean_f (float): mean field pm(ra or dec) you inputted

    Returns:
        float: posterior probability
    """
    lp      = lnprior_pm(x, max_vmean, max_vmean_f)

    if not np.isfinite(lp):
        return -np.inf

    return lp + ln_pm(x, pm, sig_pm)
    
# def ln_rv(velocity, sigvel, mass, F_yn):   
#     global args 
#     nbinaries = int(1e6)
#     all_binaries = solar(args, nbinaries=nbinaries )
#     lnlike = all_binaries.single_epoch(velocity, sigvel, mass, F_yn, 
#                                        log_minv=-3, log_maxv=None, log_stepv=0.02)
#     return lnlike


def solar(args, nbinaries):
    """Returns a randomly generated dataset of `nbinaries` solar-type binaries.

    These can be used to fit an observed radial velocity distribution or to 
    generate random radial velocity datasets for Monte Carlo simulations.

    These are defined by:
    - The log-normal period distribution from Raghavan et al. 
        (2010, ApJS, 190, 1)
    - The slightly sloped mass ratio distribution between 0.1 and 1 from 
        Reggiani & Meyer (2013, A&A, 553, 124).
    - A flat eccentricity distribution with a maximum eccentricity as observed 
        in Raghavan et al. (2010) and characterized by Parker et al. (...)

    Arguments:
    - `nbinaries`: number of orbital parameters to draw.
    """
    properties = OrbitalParameters(nbinaries)
    properties.draw_period(args)
    properties.draw_mass_ratio(args)
    properties.draw_eccentricities(args)
    return properties

# def rv_max_like(nll, initial):
#     """get maximum likelihood answer
#     not useful, you shouldn't use it
#     """
#     print('---------------------------------------------')
#     print('Now finding the maximum likelihood for RV...')
    
#     soln = opti.minimize(nll, initial)
#     max_vmean, max_vdisp, max_fbin, max_vmean_f, max_vdisp_f = soln.x
#     print(f'RV maximum likelihood values: vmean={max_vmean:1.2f}, vdisp={max_vdisp:1.2f}, fbin={max_fbin:1.2f}, vmean_f={max_vmean_f:1.2f}, vdisp_f={max_vdisp_f:1.2f}\n')
#     run_max = 1
    
#     initial = np.array([max_vmean, max_vdisp, max_fbin, max_vmean_f, 
#                         max_vdisp_f]).T # initial samples
    
#     return initial, run_max, max_vmean, max_vmean_f

# def pm_max_like(nll, initial, rxORdec='RA'):
#     """get maximum likelihood answer
#     not useful, you shouldn't use it
#     """
#     print('---------------------------------------------')
#     print('Now finding the maximum likelihood for pm{rxORdec}...')
    
#     soln = opti.minimize(nll, initial, args=(pmra, epmra))
#     max_vmean, max_vdisp, max_vmean_f, max_vdisp_f = soln.x
#     print(f'pm{rxORdec} maximum likelihood values: vmean={max_vmean:1.2f}, vdisp={max_vdisp:1.2f}, vmean_f={max_vmean_f:1.2f}, vdisp_f={max_vdisp_f:1.2f}\n')
#     run_max = 1
    
#     initial = np.array([max_vmean, max_vdisp, max_vmean_f, 
#                         max_vdisp_f]).T # initial samples
    
#     return initial, run_max, max_vmean, max_vmean_f

def clean_h5(dd):
    # remove previous .h5 file to save space
    # dir_list = os.listdir()
    # for dd in dir_list:
    #     if '.h' in dd:
    #         os.remove(dd)
    #         print(f'Clean {dd}')
    os.remove(dd)
    print(f'Clean {dd}')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                                     prog        = 'modified the velbin code',
                                     description = '''
                                     modified velbin code.
                                     ''',
                                     epilog = "Contact author: sytang@lowell.edu")
    parser.add_argument("filename",                          action="store",
                        help="Enter your filename you wish you use under ./Input/", type=str)

    parser.add_argument('-rv',       dest="rv",         action="store",
                        help="RV initial parameters [vmean,vdisp,vmean_f,vdisp_f]",
                        type=str,   default='' )
    parser.add_argument('-fbin',       dest="fbin",         action="store",
                        help="finary fraction, from 0-->1, default = 0.5",
                        type=float,   default=0.5)

    parser.add_argument('-pmra',       dest="pmra",         action="store",
                        help="pmRA initial parameters [mean,dispertion,mean_f,dispertion_f]",
                        type=str,   default='' )

    parser.add_argument('-pmdec',       dest="pmdec",         action="store",
                        help="pmDEC initial parameters [mean,dispertion,mean_f,dispertion_f]",
                         type=str,   default='' )

    # MCMC setting     
    parser.add_argument('-walker',       dest="walker",         action="store",
                        help="Number of walkers to use for MCMC. Default = 50",
                         type=int,   default=50 )
    
    parser.add_argument('-steps',       dest="steps",         action="store",
                        help="Number of steps for each walkers to run in MCMC. Default = 20000",
                         type=int,   default=20000 )

    parser.add_argument('-burnin',       dest="burnin",         action="store",
                        help="Number of burn in steps. Default = 15000",
                         type=int,   default=15000 )

    #---- optional inputs ----
    parser.add_argument('-mode',       dest="mode",         action="store",
                        help="'solar' OR 'ob_stars'. Default='solar'",
                        type=str,   default='solar' )
    parser.add_argument('-period',    dest="period",    action="store",
                        help="pre-loaded: Raghavan10, DM91, Sana12, Sana13, Kiminki12. Default = Raghavan10",
                        type=str,   default='Raghavan10' )
    parser.add_argument('-massratio', dest="mass_ratio", action="store",
                        help="pre-loaded: flat, Reggiani13, Sana12, Sana13, Kiminki12. Default = flat",
                        type=str,   default='flat' )

    global args
    args = parser.parse_args()
    clean_name = args.filename.replace(' ', '')
    clean_name = clean_name.replace('_', '')
    png_save_name = clean_name

    os.environ["OMP_NUM_THREADS"] = "1"
    
    if len(args.rv) != 0:
        doRV = 1
        unpackRV = np.array(ast.literal_eval(args.rv),  dtype=str)
        if len(unpackRV)==4:
            vmean, vdisp, vmean_f, vdisp_f = np.array(
                                        ast.literal_eval(args.rv), dtype=str)
            doFieldStars = 1
            print('\nDo the field star fit...')
        else:
            vmean, vdisp = np.array(ast.literal_eval(args.rv),  dtype=str)
            doFieldStars = 0
            print('\nSkip the field star fit...')
    else:
        doRV = 0
        
    
    if len(args.pmra) != 0:
        doPMra = 1
        pmra_mean, pmra_disp, pmra_mean_f, pmra_disp_f  = np.array(
                                        ast.literal_eval(args.pmra),  dtype=str)
    else:
        doPMra = 0
        
        
    if len(args.pmdec) != 0:
        doPMdec = 1
        pmdec_mean, pmdec_disp, pmdec_mean_f, pmdec_disp_f = np.array(
                                    ast.literal_eval(args.pmdec), dtype=str)
    else:
        doPMdec = 0

        
    # read in data
    velocity, sigvel, mass, pmra, pmdec, epmra, epmdec = read_input(args)
    
    # mcmc setting 
    nwalkers, nstep, nburn, fnrv, fnpmra, fnpmdec = mcmc_setting(args)

    # number of binaries to draw in MC
    nbinaries = int(1e6)
    print(f'Draw {nbinaries} binaries for MC...')

    if args.mode.lower() == 'solar':
        all_binaries = solar(args, nbinaries=nbinaries )
        print('Using the "single_epoch" mode to fit... \n')

        if doRV : 
            try:
                clean_h5(fnrv)
            except FileNotFoundError:
                pass
            # get ln likelihood for observed single-epoch radial velocity distribution
            single_epoch_lnlike = all_binaries.single_epoch(
                velocity, sigvel, mass, doFieldStars, 
                log_minv=-3, log_maxv=None, log_stepv=0.02)

            nll = lambda *argsss: -single_epoch_lnlike(*argsss)
            
            if doFieldStars :
                # initial guess
                initial = np.array([float(vmean),
                                    float(vdisp),
                                    # float(fbin),
                                    float(vmean_f),
                                    float(vdisp_f)  ]).T # initial guess

                # prevent negative vales
                initial = np.where(initial > 1E-5, initial, 1E-1)

                # run opti.minimize to get max likelihood
                # but found not that useful...
                
                # initial, run_max = rv_max_like(nll, initial)

                print(f'now run rv MCMC')
                
                ndim = len(initial) # number of parameters/dimensions
                pos = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
                # print(pos[0])

                # define backup
                backend = emcee.backends.HDFBackend(fnrv)
                backend.reset(nwalkers, ndim)

                with Pool() as pool:
                    # if not run_max:
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_rv,
                                                    args=(single_epoch_lnlike, 
                                                          initial[0],
                                                          initial[2],
                                                          args.fbin),
                                                    backend=backend, pool=pool)
                    
                    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_rv,
                    #                                     args=(lnlike, max_vmean, max_vmean_f),
                    #                                     backend=backend, pool=pool)
                    sampler.run_mcmc(pos, nstep, progress=True)

                # plotting
                cornerplot(fnrv, nburn, ndim, 20, png_save_name, 'rv')

            else:
                # initial guess
                initial = np.array([float(vmean),
                                    float(vdisp),
                                    # float(fbin)  
                                    ]).T # initial samples

                soln = opti.minimize(nll, initial)
                # max_vmean, max_vdisp, max_fbin = soln.x
                # print(f'RV maximum likelihood values: vmean={max_vmean:1.2f}, vdisp={max_vdisp:1.2f}, fbin={max_fbin:1.2f}\n')
                max_vmean, max_vdisp = soln.x
                print(f'RV maximum likelihood values: vmean={max_vmean:1.2f}, vdisp={max_vdisp:1.2f}\n')

        #-----------------------------------------------------------------------------------
        if doPMra :
            try:
                clean_h5(fnpmra)  
            except FileNotFoundError:
                pass
        # nwalkers, nstep, nburn = 200, 50000, 45000 # for pmRA and pmDEC
        # nwalkers, nstep, nburn = 50, 30000, 25000
        
            nll = lambda *argsss: -ln_pm(*argsss)

            # initial guess --------
            initial = np.array([float(pmra_mean),
                                float(pmra_disp),
                                float(pmra_mean_f),
                                float(pmra_disp_f)]).T # initial samples
            initial = np.where(initial > 1E-5, initial, 1E-10)
            
            # run opti.minimize to get max likelihood
            # but found not that useful...
                
            # initial, run_max = pm_max_like(nll, initial)
            
            print(f'now run PMra MCMC')

            ndim = len(initial) # number of parameters/dimensions
            pos = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
            # print(pos[0])
            
            # define backup
            backend = emcee.backends.HDFBackend(fnpmra)
            backend.reset(nwalkers, ndim)
            
            with Pool() as pool:
                # if not run_max:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pm,
                                                args=(pmra, epmra, initial[0],
                                                      initial[2]),
                                                backend=backend, pool=pool)
                # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pmra,
                                                    # args=(pmra, epmra, max_vmean, max_vmean_f),
                                                    # backend=backend, pool=pool)
                sampler.run_mcmc(pos, nstep, progress=True)
            
            ### plotting
            cornerplot(fnpmra, nburn, ndim, 20, png_save_name, 'pmra')
    

        #-----------------------------------------------------------------------------------
        if doPMdec:
            try:
                clean_h5(fnpmdec)  
            except FileNotFoundError:
                pass
            
            nll = lambda *argsss: -ln_pm(*argsss)

            # initial guess --------
            initial = np.array([float(pmdec_mean),
                                float(pmdec_disp),
                                float(pmdec_mean_f),
                                float(pmdec_disp_f)]).T # initial samples
            initial = np.where(initial > 1E-5, initial, 1E-10)


            # run opti.minimize to get max likelihood
            # but found not that useful...
                
            # initial, run_max = pm_max_like(nll, initial, 'DEC)
            
            print(f'now run PMdec MCMC')
    
            ndim = len(initial) # number of parameters/dimensions
            pos = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
            # print(pos[0])
            
            # define backup
            backend = emcee.backends.HDFBackend(fnpmdec)
            backend.reset(nwalkers, ndim)
            
            with Pool() as pool:
                # if not run_max:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pm,
                                                args=(pmdec, epmdec, 
                                                    initial[0], initial[2]),
                                                backend=backend, pool=pool)
                # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pmdec,
                #                                     args=(pmdec, epmdec, max_vmean, max_vmean_f),
                #                                     backend=backend, pool=pool)
                sampler.run_mcmc(pos, nstep, progress=True)
            
            ### plotting
            cornerplot(fnpmdec, nburn, ndim, 20, png_save_name, 'pmdec')


        print('Program finished \n')

        # print(BinaryObj(velocity, sigvel, mass))
        # print(all_binaries.arr['mass_ratio'])

    elif args.mode.lower() == 'ob_stars':
        sys.exit(f"Sorry, MCMC code is currently unable to perform 'ob_stars'")
        # all_binaries = ob_stars(args, source, pmax=None, nbinaries=nbinaries)

    else:
        sys.exit(f"Wrong input with: {args.mode}, only take 'solar' OR 'ob_stars'")
    
    os.environ["OMP_NUM_THREADS"] = ""
