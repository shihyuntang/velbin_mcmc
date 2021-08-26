import numpy as np
import sys, os, argparse, ast
import itertools
from  function.binaries import *
# from  function.fitter import *

from astropy.table import Table
from scipy import optimize as opti
from multiprocessing import Pool

import corner, emcee
import matplotlib.pyplot as plt
fig_dpi      = 300
fig_typeface = 'Helvetica'
# fig_family   = 'sans-serif'
fig_family   = 'monospace'
fig_style    = 'normal'


def solar(args, nbinaries):
    """Returns a randomly generated dataset of `nbinaries` solar-type binaries.

    These can be used to fit an observed radial velocity distribution or to generate random radial velocity datasets for Monte Carlo simulations.

    These are defined by:
    - The log-normal period distribution from Raghavan et al. (2010, ApJS, 190, 1)
    - The slightly sloped mass ratio distribution between 0.1 and 1 from Reggiani & Meyer (2013, A&A, 553, 124).
    - A flat eccentricity distribution with a maximum eccentricity as observed in Raghavan et al. (2010) and characterized by Parker et al. (...)

    Arguments:
    - `nbinaries`: number of orbital parameters to draw.
    """
    properties = OrbitalParameters(nbinaries)
    properties.draw_period(args, 'Raghavan10')
    properties.draw_mass_ratio(args, 'Reggiani13')
    properties.draw_eccentricities(args)
    return properties


#----------------------------------
#--- log likelihood ---------------
def ln_pmra(x, pmra, sig_pmra):
    f_c  = 0.95
    pmra_mean, pmra_disp, pmra_mean_f, pmra_disp_f = x
    likelihood_single = f_c    * np.exp(-(pmra - pmra_mean) ** 2 / (2 * (sig_pmra ** 2. + pmra_disp ** 2.))) / np.sqrt(2 * np.pi * (sig_pmra ** 2. + pmra_disp ** 2.)) + \
                        (1-f_c)* np.exp(-(pmra - pmra_mean_f) ** 2 / (2 * (sig_pmra ** 2. + pmra_disp_f ** 2.))) / np.sqrt(2 * np.pi * (sig_pmra** 2. + pmra_disp_f ** 2.))

    result = np.where(likelihood_single > 1E-5, likelihood_single, 1E-10)
    return np.sum(np.log(result))


def ln_pmdec(x, pmdec, sig_pmdec):
    f_c  = 0.95
    pmdec_mean, pmdec_disp, pmdec_mean_f, pmdec_disp_f = x
    likelihood_single = f_c    * np.exp(-(pmdec - pmdec_mean) ** 2 / (2 * (sig_pmdec ** 2. + pmdec_disp ** 2.))) / np.sqrt(2 * np.pi * (sig_pmdec ** 2. + pmdec_disp ** 2.)) + \
                        (1-f_c)* np.exp(-(pmdec - pmdec_mean_f) ** 2 / (2 * (sig_pmdec ** 2. + pmdec_disp_f ** 2.))) / np.sqrt(2 * np.pi * (sig_pmdec** 2. + pmdec_disp_f ** 2.))

    result = np.where(likelihood_single > 1E-5, likelihood_single, 1E-10)
    return np.sum(np.log(result))


def ln_rv(velocity, sigvel, mass, F_yn):
    global args
    nbinaries = np.int(1e6)
    all_binaries = solar(args, nbinaries=nbinaries )
    lnlike = all_binaries.single_epoch(velocity, sigvel, mass, F_yn, log_minv=-3, log_maxv=None, log_stepv=0.02)
    return lnlike

#----------------------------------
#--- Radial velocity --------------
def lnprior_rv(x, velocity, sigvel, mass, F_yn, max_vmean, max_vmean_f):
    vmean, vdisp, fbin, vmean_f, vdisp_f = x

    # Uniform prior
    uni_rv_mean   = (vmean   > (max_vmean-5) )  & (vmean   < (max_vmean+5)) #\pm5
    uni_rv_disp   = (vdisp   > 0.0)             & (vdisp   < 1.5) #
    uni_rv_mean_f = (vmean_f > (max_vmean_f-10)) & (vmean_f < (max_vmean_f+10))
    uni_rv_disp_f = (vdisp_f > 0.0)             & (vdisp_f < 1.5)
    uni_rv_fbin   = (fbin    > 0.0)             & (fbin < 1.)

    uni_all = uni_rv_mean & uni_rv_disp & uni_rv_mean_f & uni_rv_disp_f & uni_rv_fbin

    if not uni_all:
        return -np.inf
    else:
        return 0

def lnprob_rv(x, lnlike, max_vmean, max_vmean_f):
    lp      = lnprior_rv(x, velocity, sigvel, mass, F_yn, max_vmean, max_vmean_f)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(x)

#--------------------------------
#--- proper motion ra -----------
def lnprior_pmra(x, pmra, sig_pmra, max_vmean, max_vmean_f):
    pmra_mean, pmra_disp, pmra_mean_f, pmra_disp_f = x

    # Uniform prior
    uni_pmra_mean   = (pmra_mean   > (max_vmean-5) )   & (pmra_mean   < (max_vmean+5)) #\pm5
    uni_pmra_disp   = (pmra_disp   > 0.0)              & (pmra_disp   < 1.5) #
    uni_pmra_mean_f = (pmra_mean_f > (max_vmean_f-5))  & (pmra_mean_f < (max_vmean_f+5))
    uni_pmra_disp_f = (pmra_disp_f > 0.0)              & (pmra_disp_f < 1.5)

    uni_all = uni_pmra_mean & uni_pmra_disp & uni_pmra_mean_f & uni_pmra_disp_f

    if not uni_all:
        return -np.inf

    else:
        return 0

def lnprob_pmra(x, pmra, sig_pmra, max_vmean, max_vmean_f):
    lp      = lnprior_pmra(x, pmra, sig_pmra, max_vmean, max_vmean_f)

    if not np.isfinite(lp):
        return -np.inf

    return lp + ln_pmra(x, pmra, sig_pmra)

#--------------------------------
#--- proper motion dec ----------
def lnprior_pmdec(x, pmdec, sig_pmdec, max_vmean, max_vmean_f):
    pmdec_mean, pmdec_disp, pmdec_mean_f, pmdec_disp_f = x

    # Uniform prior
    uni_pmdec_mean   = (pmdec_mean   > (max_vmean-5) )  & (pmdec_mean   < (max_vmean+5) ) #\pm5
    uni_pmdec_disp   = (pmdec_disp   > 0.0)             & (pmdec_disp   < 1.5) #
    uni_pmdec_mean_f = (pmdec_mean_f > (max_vmean_f-5)) & (pmdec_mean_f < (max_vmean_f+5))
    uni_pmdec_disp_f = (pmdec_disp_f > 0.0)             & (pmdec_disp_f < 1.5)

    uni_all = uni_pmdec_mean & uni_pmdec_disp & uni_pmdec_mean_f & uni_pmdec_disp_f

    if not uni_all:
        return -np.inf

    else:
        return 0

def lnprob_pmdec(x, pmdec, sig_pmdec, max_vmean, max_vmean_f):
    lp      = lnprior_pmdec(x, pmdec, sig_pmdec, max_vmean, max_vmean_f)

    if not np.isfinite(lp):
        return -np.inf

    return lp + ln_pmdec(x, pmdec, sig_pmdec)

# --------------------------------
#--- corner plotting -------------
def cornerplot(fn, nburn, ndim, nthin, pngsave_name, partype):
    reader = emcee.backends.HDFBackend(fn)
    samples = reader.get_chain(discard=nburn, thin=nthin, flat=True)

    f, ax = plt.subplots(ndim, ndim, figsize=(ndim+1, ndim+1), facecolor='white', dpi=300, gridspec_kw={'hspace': .05, 'wspace': 0.05})

    if ndim == 4:
        labb = ["vmean", r"vdisp", r"vmean_f", "vdisp_f"]
    else:
        labb = ["vmean", r"vdisp", "fbin", r"vmean_f", "vdisp_f"]

    fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, color='xkcd:olive',
                        labels=labb, title_kwargs={'size':7}, title_fmt='1.3f',
                        fig=f)

    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.tick_params(axis='both', which ='both', labelsize=6, right=True, top=True, direction='in', width=.4)

            ax.yaxis.get_label().set_fontsize(8)
            ax.yaxis.get_label().set_fontstyle(fig_style)
            ax.yaxis.get_label().set_fontfamily(fig_family)

            ax.xaxis.get_label().set_fontsize(8)
            ax.xaxis.get_label().set_fontstyle(fig_style)
            ax.xaxis.get_label().set_fontfamily(fig_family)

            ax.xaxis.get_offset_text().set_fontsize(6)
            ax.yaxis.get_offset_text().set_fontsize(6)


    for i in range(ndim):
        ax = axes[i, i]
        ax.tick_params(axis='both', which ='both', labelsize=6, right=True, top=True, direction='in', width=.4)

        ax.xaxis.get_label().set_fontsize(8)
        ax.xaxis.get_label().set_fontstyle(fig_style)
        ax.xaxis.get_label().set_fontfamily(fig_family)

    f.savefig(f'./figs/{pngsave_name}_{partype}_mcmc.png', format='png', bbox_inches='tight')


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                                     prog        = 'modified the velbin code',
                                     description = '''
                                     modified velbin code.
                                     ''',
                                     epilog = "Contact author: sytang@lowell.edu")
    parser.add_argument("filename",                          action="store",
                        help="Enter your filename you wish you use under ./Input/", type=str)
    parser.add_argument('-headerused',       dest="mode",         action="store",
                        help="if not specified, will use the default names",
                        type=str,   default='' )

    #---- must inputs ----
    parser.add_argument('-rv',       dest="rv",         action="store",
                        help="RV initial parameters [vmean,vdisp,vmean_f,vdisp_f,fbin]",
                        type=str,   default='' )

    parser.add_argument('-pmra',       dest="pmra",         action="store",
                        help="pmRA initial parameters [mean,dispertion,mean_f,dispertion_f]",
                        type=str,   default='' )

    parser.add_argument('-pmdec',       dest="pmdec",         action="store",
                        help="pmDEC initial parameters [mean,dispertion,mean_f,dispertion_f]",
                         type=str,   default='' )

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

    # parser.add_argument('-c',       dest="Nthreads",         action="store",
    #                     help="Numbers of run_bash gnerate, i.e., numbers of cpu (threads) to use, default is 1",
    #                     type=int,   default=int(1) )
    global args
    args = parser.parse_args()


    #---- check must inputs ----
    # if args.vmean == '':
    #     sys.exit('missing values for "-vmean", you must give a initial guess')
    # if args.vdisp == '':
    #     sys.exit('missing values for "-vdisp", you must give a initial guess')
    # if args.fbin == '':
    #     sys.exit('missing values for "-fbin", you must give a initial guess')

    unpackRV = np.array(ast.literal_eval(args.rv),  dtype=str)
    if len(unpackRV)==5:
        vmean, vdisp, vmean_f, vdisp_f, fbin = np.array(ast.literal_eval(args.rv),  dtype=str)
        F_yn = 1
        print('Do the field star fit...')
    else:
        vmean, vdisp, fbin = np.array(ast.literal_eval(args.rv),  dtype=str)
        F_yn = 0
        print('Skip the field star fit...')

    pngsave_name = f'{args.filename.split("_")[0]}{args.filename.split("_")[1]}'
    #---- inpur data ----
    dataorg = Table.read('./Input/{}.fits'.format(args.filename))
    print('data must have "RV_Jackson" RV, cleaning...')

    if (args.filename == 'Coma_Berenices_rv_dr3_tr') or (args.filename == 'NGC_6774_rv_dr3_tr'):
        rvuseN    = 'dr2_radial_velocity'
        rvuseNerr = 'dr2_radial_velocity_error'

    elif args.filename == 'NGC_2422_rv_dr3_tr':
        vuseN    = 'RV_Bailey'
        rvuseNerr = 'e_RV_Bailey'

    else:
        rvuseN    = 'RV_Jackson'
        rvuseNerr = 'e_RV_Jackson'

    dataclean = dataorg[dataorg[rvuseN]>-9000]

    velocity = np.array(dataclean[rvuseN])
    sigvel   = np.array(dataclean[rvuseNerr])
    mass     = dataclean['Mass']

    pmra     = np.array(dataclean['pmra'])
    epmra    = np.array(dataclean['pmra_error'])
    pmdec    = np.array(dataclean['pmdec'])
    epmdec   = np.array(dataclean['pmdec_error'])

    #-- MCMC setting ---
    nwalkers, nstep, nburn = 100, 5000, 1000  # for RV only...
    fnrv      = "rv_mcmcsave.h5"
    fnpmra    = "pmra_mcmcsave.h5"
    fnpmdec   = "pmdec_mcmcsave.h5"
    #--------------
    #---- mode ----
    nbinaries = np.int(1e6)
    if args.mode.lower() == 'solar':
        all_binaries = solar(args, nbinaries=nbinaries )

        print('Using the "single_epoch" mode to fit... \n')

        lnlike = all_binaries.single_epoch(velocity, sigvel, mass, F_yn, log_minv=-3, log_maxv=None, log_stepv=0.02)

        #-----------
        print('---------------------------------------------')
        print('Now finding the maximum likelihood for RV...')
        nll = lambda *argsss: -lnlike(*argsss)

        if F_yn ==1:
            # initial guess --------
            initial = np.array([np.float(vmean),
                                np.float(vdisp),
                                np.float(fbin),
                                np.float(vmean_f),
                                np.float(vdisp_f)  ]).T # initial samples

            soln = opti.minimize(nll, initial)
            max_vmean, max_vdisp, max_fbin, max_vmean_f, max_vdisp_f = soln.x
            print(f'RV maximum likelihood values: vmean={max_vmean:1.2f}, vdisp={max_vdisp:1.2f}, fbin={max_fbin:1.2f}, vmean_f={max_vmean_f:1.2f}, vdisp_f={max_vdisp_f:1.2f}\n')

            if max_vdisp_f <= 1E-5:
                max_vdisp_f = 1E-1
                print(f'Getting minus or zero for vdisp_f, setting it to {max_vdisp_f} to avoid error...')
            #----------------
            print(f'now run MCMC')
            #----------------
            # initial = np.array([max_vmean, max_vdisp, max_fbin, max_vmean_f, max_vdisp_f]).T # initial samples
            ndim = len(initial) # number of parameters/dimensions
            pos = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
            # print(pos[0])

            # define backup
            backend = emcee.backends.HDFBackend(fnrv)
            backend.reset(nwalkers, ndim)

            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_rv,
                                                    args=(lnlike, max_vmean, max_vmean_f),
                                                    backend=backend, pool=pool)
                sampler.run_mcmc(pos, nstep, progress=True)

            ### plotting
            cornerplot(fnrv, nburn, ndim, 5, pngsave_name, 'rv')
            #----------------


        elif F_yn ==0:
            # initial guess --------
            initial = np.array([np.float(vmean),
                                np.float(vdisp),
                                np.float(fbin)  ]).T # initial samples

            soln = opti.minimize(nll, initial)
            max_vmean, max_vdisp, max_fbin = soln.x
            print(f'RV maximum likelihood values: vmean={max_vmean:1.2f}, vdisp={max_vdisp:1.2f}, fbin={max_fbin:1.2f}\n')

        #-----------------------------------------------------------------------------------
        nwalkers, nstep, nburn = 100, 5000, 1000 # for pmRA and pmDEC
        if args.pmra !='':
            print('---------------------------------------------')
            print('Now finding the maximum likelihood for pmRA...')
            pmra_mean, pmra_disp, pmra_mean_f, pmra_disp_f     = np.array(ast.literal_eval(args.pmra),  dtype=str)

            nll = lambda *argsss: -ln_pmra(*argsss)

            # initial guess --------
            initial = np.array([np.float(pmra_mean),
                                np.float(pmra_disp),
                                np.float(pmra_mean_f),
                                np.float(pmra_disp_f)]).T # initial samples

            soln = opti.minimize(nll, initial, args=(pmra, epmra))
            max_vmean, max_vdisp, max_vmean_f, max_vdisp_f = soln.x
            print(f'pmRA maximum likelihood values: vmean={max_vmean:1.2f}, vdisp={max_vdisp:1.2f}, vmean_f={max_vmean_f:1.2f}, vdisp_f={max_vdisp_f:1.2f}\n')

            #----------------
            print(f'now run MCMC')
            #----------------
            # initial = np.array([max_vmean, max_vdisp, max_vmean_f, max_vdisp_f]).T # initial samples
            ndim = len(initial) # number of parameters/dimensions
            pos = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
            # print(pos[0])

            # define backup
            backend = emcee.backends.HDFBackend(fnpmra)
            backend.reset(nwalkers, ndim)

            with Pool() as pool:

                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pmra,
                                                    args=(pmra, epmra, max_vmean, max_vmean_f),
                                                    backend=backend, pool=pool)
                sampler.run_mcmc(pos, nstep, progress=True)

            ### plotting
            cornerplot(fnpmra, nburn, ndim, 5, pngsave_name, 'pmra')
            #----------------

        #-----------------------------------------------------------------------------------
        if args.pmdec !='':
            print('---------------------------------------------')
            print('Now finding the maximum likelihood for pmDEC...')
            pmdec_mean, pmdec_disp, pmdec_mean_f, pmdec_disp_f = np.array(ast.literal_eval(args.pmdec), dtype=str)

            nll = lambda *argsss: -ln_pmdec(*argsss)

            # initial guess --------
            initial = np.array([np.float(pmdec_mean),
                                np.float(pmdec_disp),
                                np.float(pmdec_mean_f),
                                np.float(pmdec_disp_f)]).T # initial samples

            soln = opti.minimize(nll, initial, args=(pmdec, epmdec))
            max_vmean, max_vdisp, max_vmean_f, max_vdisp_f = soln.x
            print(f'pmDEC maximum likelihood values: vmean={max_vmean:1.2f}, vdisp={max_vdisp:1.2f}, vmean_f={max_vmean_f:1.2f}, vdisp_f={max_vdisp_f:1.2f}')
            #----------------
            #----------------
            print(f'now run MCMC')
            #----------------
            # initial = np.array([max_vmean, max_vdisp, max_vmean_f, max_vdisp_f]).T # initial samples
            ndim = len(initial) # number of parameters/dimensions
            pos = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
            # print(pos[0])

            # define backup
            backend = emcee.backends.HDFBackend(fnpmdec)
            backend.reset(nwalkers, ndim)

            with Pool() as pool:

                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pmdec,
                                                    args=(pmdec, epmdec, max_vmean, max_vmean_f),
                                                    backend=backend, pool=pool)
                sampler.run_mcmc(pos, nstep, progress=True)

            ### plotting
            cornerplot(fnpmdec, nburn, ndim, 5, pngsave_name, 'pmdec')
            #----------------

        print('Program finished \n')

        # print(BinaryObj(velocity, sigvel, mass))
        # print(all_binaries.arr['mass_ratio'])

    elif args.mode.lower() == 'ob_stars':
        all_binaries = ob_stars(args, source, pmax=None, nbinaries=nbinaries)

    else:
        sys.exit(f"Wrong input with: {args.mode}, only take 'solar' OR 'ob_stars'")
