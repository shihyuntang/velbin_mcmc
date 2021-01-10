"""Contains the BinaryFit class, which can be used to compute the log-likelihood of reproducing a set of observed radial velocities.

This class is created by OrbitalParameters.single_epoch or OrbitalParameters.multi_epoch, so this module does not have to be imported.
"""
import scipy as sp
import numpy as np
import scipy.special

class BinaryFit(object):
    """When called computes the log-likelihood of reproducing a set of observed radial velocities.

    The call takes 3 parameters, the mean velocity, velocity dispersion, and binary fraction of the cluster. These parameters can be varied to maximize the log-likelihood using any optmization routines (see scipy.optimize) or can be used in Markov Chain Monte Carlo Simulations (e.g. pymc, emcee).

    Properties (all of these are correctly set during initialization):
    - `velocity`: observed radial velocities in km/s (in the multi-epoch case contains the weighted means of the seemingly single stars).
    - `sigvel`: measurement uncertainties of the observed radial velocities in km/s (in the multi-epoch case this will be set to one, as the measurement uncertainties has already been taken into account in pbin).
    - `mass`: best mass estimate of the observed stars in solar masses (in the multi-epoch case this will be set to one, as the mass has already been taken into account in pbin).
    - `vbin`: velocity edges of the bins (in km/s) over which `pbin` was computed.
    - `pbin`: the probability distribution of the radial velocity offset due to binary orbital motions. The probability (defined per km/s) to have a velocity between vbin[i] and vbin[i + 1] is given by pbin[i]. In the single-epoch case this will be a single 1-dimensional array for a generic mass of 1 solar masses. In the multi-epoch case this will be a 2-dimensional array containing the probability distribution for every seemingly single star (given the observations of the star).
    - `pdet_single`: For all seemingly single stars: the probability that a binary would have been detected if the star is a binary (0 for the single-epoch case).
    - `pdet_rvvar`: For all stars with detected radial velocity variability: the probability that a binary would have been detected if the star is a binary (empy array for the single-epoch case).
    - `is_single`: Boolean array marking which stars are single.
    """
    def __init__(self, velocity, sigvel, mass, vbin, pbin, pdet_single=0., pdet_rvvar=np.array([]), is_single=None):
        """Do not initialize BinaryFit directly.
        Use OrbitalParameters.single_epoch or OrbitalParameters.multi_epoch instead.

        Parameters:
        - `velocity`: observed velocities for seemingly single stars in km/s.
        - `sigvel`: measurement uncertainties for seemingly single stars in km/s.
        - `mass`: mass of observed stars in solar masses.
        - `vbin`: array-like; velocities of the bins over which the `pbin` is computed.
        - `pbin`: array-like; the probability per km/s to have a velocity offset due to binary orbital motions between vbin[i] and vbin[i + 1] is stored in pbin[i]
        - `pdet_single`: for all seemingly single stars: the probability that a binary would have been detected if the star is a binary.
        - `pdet_rvvar`: for all stars with detected RV variability: the probability that a bianary would have been detected if the star is a binary.
        """
        self.velocity = velocity
        self.sigvel = sigvel
        self.mass = mass
        self.vbin = vbin
        if len(pbin.shape) == 1:
            pbin = pbin[:, None]
        self.pbin = pbin
        self.pdet_single = pdet_single
        self.pdet_rvvar = pdet_rvvar
        if is_single is None:
            self.is_single = np.ones(self.velocity.shape, dtype='bool')
        else:
            self.is_single = is_single

    def individual_log_likelihood(self, vmean, vdisp, fbin):
        """Computes array with the individual log-likelihoods for observing the (average) radial velocity of the observed (seemingly single) stars.

        Call the object for the total log-likelihood (which includes a contribution from RV-variables).

        Arguments:
        - `vmean`: mean velocity of the cluster in km/s.
        - `vdisp`: velocity dispersion of the cluster in km/s.
        - `fbin`: binary fraction of the stars in the cluster.
        """
        vmean_all, vdisp_all, fbin_all = [np.asarray(variable) for variable in [vmean, vdisp, fbin]]
        vmean, vdisp, fbin = [variable[self.is_single] if variable.size > 1 else variable for variable in [vmean_all, vdisp_all, fbin_all]]

        nvel = len(self.velocity)
        fbin_new = fbin * (1 - self.pdet_single) / (1 - fbin * self.pdet_single)
        likelihood_single = np.exp(-(self.velocity - vmean) ** 2 / (2 * (self.sigvel ** 2. + vdisp ** 2.))) / np.sqrt(2 * np.pi * (self.sigvel ** 2. + vdisp ** 2.))

        p_bound = sp.special.erf((vmean + self.vbin[1:-1, None] * self.mass ** (1. / 3.) - self.velocity) / np.sqrt(2. * (self.sigvel ** 2. + vdisp ** 2.))) * .5 + .5
        p_long = np.append(np.append(np.zeros((1, nvel)), p_bound, 0), np.ones((1, nvel)), 0)
        weight = p_long[1:, :] - p_long[:-1, :]
        likelihood_binary = np.sum(self.pbin * weight / self.mass ** (1. / 3.), 0)

        # print(np.log(fbin_new * likelihood_binary + (1 - fbin_new) * likelihood_single))
        return np.log(fbin_new * likelihood_binary + (1 - fbin_new) * likelihood_single)

    def log_likelihood_detection(self, fbin):
        """Calculates the log-likelihood of the RV-variables actually being detected and the seemingly single stars not.

        Computes the binomial probability using the individual detection probability of every star

        Arguments:
        - `fbin`: binary fraction of the stars in the cluster.
        """
        fbin_all = np.array(fbin)
        fbin_sin = fbin_all[self.is_single] if fbin_all.size > 1 else fbin_all
        fbin_mult = fbin_all[~self.is_single] if fbin_all.size > 1 else fbin_all
        return np.sum(np.log(1 - fbin_sin * self.pdet_single)) + np.sum(np.log(fbin_mult * self.pdet_rvvar))

    def __call__(self, x):
        vmean, vdisp, fbin = x
        """Returns the log-likelihood that the observed radial velocity data is reproduced for a cluster with given parameters.

        Maximizing the returned value gives the best-fit parameters for vmean, vdisp, and fbin (assuming flat priors). These parameters can be single floats or arrays of the length of the original input velocity array.

        Arguments:
        - `vmean`: mean velocity of the cluster in km/s.
        - `vdisp`: velocity dispersion of the cluster in km/s.
        - `fbin`: binary fraction of the stars in the cluster.
        """
        return np.sum(self.individual_log_likelihood(vmean, vdisp, fbin)) + self.log_likelihood_detection(fbin)
