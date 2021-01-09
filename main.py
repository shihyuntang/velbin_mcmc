import numpy as np
import sys, os, argparse
import itertools
from  function.binaries import *
# from  function.fitter import *

from astropy.table import Table
from scipy import optimize as opti



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

# def ob_stars(args, source, pmax=None, nbinaries):
#     """Returns a randomly generated dataset of `nbinaries` OB spectroscopic binaries.
#
#     These can be used to fit an observed radial velocity distribution or to generate random radial velocity datasets for Monte Carlo simulations.
#
#     The `source` parameter is used to select a reference setting the orbital parameter distribution. It should be one of:
#     - 'Kiminki12': The binary properties found for 114 B3-O5 stars in Cyg OB2 (Kimkinki & Kobulnicky, 2012, ApJ, 751, 4)
#     - 'Sana12': The binary properties found for the O-type star population of six nearby Galactic open clusters (Sana et al., 2012, Science, 337, 444).
#     - 'Sana13': The binary properties found for 360 O-type stars in 30 Doradus as part of the Tarantula survey (Sana et al, 2013, A&A, 550, 107).
#
#     The maximum period in these distribution can be overwritten by setting `pmax`.
#     In all cases a flat eccentricity distribution is used.
#
#     Arguments:
#     - `source`: A reference setting the orbital parameter distribution (one from ['Kiminki12', 'Sana12', 'Sana13'], see above)
#     - `pmax`: maximum period to include in the distribution (default set by `source`)
#     - `nbinaries`: number of orbital parameters to draw.
#     """
#     properties = OrbitalParameters(nbinaries)
#     properties.draw_period(source, pmax=pmax)
#     properties.draw_mass_ratio(source)
#     properties.draw_eccentricities(emax=0.99)
#     return properties


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
    parser.add_argument('-vmean',       dest="vmean",         action="store",
                        help="mean velocity of the cluster in km/s",
                        type=str,   default='' )
    parser.add_argument('-vdisp',       dest="vdisp",         action="store",
                        help="velocity dispersion of the cluster in km/s",
                        type=str,   default='' )
    parser.add_argument('-fbin',       dest="fbin",         action="store",
                        help="binary fraction of the stars in the cluster.",
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

    args = parser.parse_args()


    #---- check must inputs ----
    if args.vmean == '':
        sys.exit('missing values for "-vmean", you must give a initial guess')
    if args.vdisp == '':
        sys.exit('missing values for "-vdisp", you must give a initial guess')
    if args.fbin == '':
        sys.exit('missing values for "-fbin", you must give a initial guess')

    #---- inpur data ----
    dataorg = Table.read('./Input/{}.fits'.format(args.filename))
    print('data must have "RV_tb2" RV, cleaning...')
    dataclean = dataorg[dataorg['RV_tb2']>-9000]

    velocity = np.array(dataclean['RV_tb2'])
    sigvel   = np.array(dataclean['e_RV_tb2'])
    mass     = np.ones(len(dataclean))
    # mass     = dataclean['mass']

    #---- mode ---
    nbinaries = np.int(1e6)
    if args.mode.lower() == 'solar':
        all_binaries = solar(args, nbinaries=nbinaries )

        print('Using the "single_epoch" mode to fit... \n')
        lnlike = all_binaries.single_epoch(velocity, sigvel, mass, log_minv=-3, log_maxv=None, log_stepv=0.02)

        #-----------
        print('Now finding the maximum likelihood...')
        np.random.seed(42)
        nll = lambda *argsss: -lnlike(*argsss)

        # initial guess --------
        initial = np.array([np.float(args.vmean),
                            np.float(args.vdisp),
                            np.float(args.fbin)  ]).T # initial samples

        soln = opti.minimize(nll, initial)
        max_vmean, max_vdisp, max_fbin = soln.x
        print(f'maximum likelihood values: vmean={max_vmean:1.2f}, vdisp={max_vdisp:1.2f}, fbin={max_fbin:1.2f}')
        # print(BinaryObj(velocity, sigvel, mass))
        # print(all_binaries.arr['mass_ratio'])

    elif args.mode.lower() == 'ob_stars':
        all_binaries = ob_stars(args, source, pmax=None, nbinaries=nbinaries)

    else:
        sys.exit(f"Wrong input with: {args.mode}, only take 'solar' OR 'ob_stars'")
