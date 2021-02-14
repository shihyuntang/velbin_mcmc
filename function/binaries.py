import scipy as sp
import numpy as np
import itertools
from function.fitter import *



class OrbitalParameters:
    """Record array containing a large amount of randomly generated binaries.

    For every binary defines the following field:
    - 'period': binary orbital period in years.
    - 'mass_ratio': ratio of the mass of the secondary to primary star. The primary star is the observed star (for which the velocity has been measured), the secondary star is its companion.
    - 'eccentricity': eccentricity of the binary orbit.
    - 'phase': phase of the orbit.
    - 'theta': angle between line connecting center of mass and periastron and the projection of the line of sight on the orbital plane (ignored when fitting single-epoch datasets).
    - 'inclination': angle between line of sight and the angular momentum of the binary (which is perpindicular to the orbital plane) (ignored when fitting single-epoch datasets).

    In addition some methods have been added:
    - `semi_major`: Returns an array containing the semi-major axis in AU for every binary.
    - `velocities`: Returns the radial velocity and proper motion offsets due to binary orbital motions for every binary.
    - `single_epoch`: Creates a callable object, which can be used to compute the log-likelihood of reproducing a single-epoch radial velocity dataset.
    - `multi_epoch`: Creates a callable object, which can be used to compute the log-likelihood of reproducing a multi-epoch radial velocity dataset.
    - `fake_dataset`: Creates a single- or multi-epoch fake radial velocity dataset for Monte Carlo simulations.
    """
    def __init__(self, nbinaries):
        arr = np.ones(nbinaries, dtype=np.dtype([(name, np.float) for name in ['period', 'mass_ratio', 'eccentricity', 'phase', 'theta', 'inclination']]))
        arr['eccentricity'] = 0.
        arr['phase'] = np.random.rand(nbinaries)
        arr['theta'] = np.random.rand(nbinaries) * 2 * np.pi
        arr['inclination'] = np.arccos(np.random.rand(nbinaries) * 2. - 1.)

        self.arr = arr

#        return arr

    def draw_period(self, args, period='Raghavan10', pmax=None):
        """Draw a new period distribution in unit of years.

        Either provide an array (with the same length as the number of binaries) or choose from:
        For solar-type stars:
          - 'Raghavan10': The log-normal distribution from Raghavan et al. (2010, ApJS, 190, 1).
          - 'DM91': The log-normal distribution from Duquennoy & Mayor (1991, A&A, 248, 485).
        For OB stars:
          - 'Kiminki12': The log-powerlaw distribution from Kiminki & Kobulnicky (2012, ApJ, 751, 4).
          - 'Sana12': The log-powerlaw distribution from Sana et al. (2012, Science, 337, 444).
          - 'Sana13': The log-powerlaw distribution from Sana et al. (2013, A&A, 550, 107).

        For the log-normal distributions `pmax` can be set (in years) to override the default maximum period.

        If the eccentricity distribution depends on the period, the eccentricity might have to be redrawn afterwards.

        Arguments:
        - `period`: New period distribution (array or name of new distribution).
        - `pmax`: Maximum period in years of log-powerlaw distribution (ignored for log-normal distributions).
        """
        nbinaries = self.arr.size

        # print(eee)
        if args.period in ('Raghavan10', 'DM91'):
            if args.period == 'Raghavan10':
                log_p_mean, sigma_log_p = 5.03, 2.28
            elif args.period == 'DM91':
                log_p_mean, sigma_log_p = 4.8, 2.3
            self.arr['period'] = 10. ** (np.random.randn(nbinaries) * sigma_log_p + log_p_mean) / 365.25

        elif args.period in ('Sana12', 'Sana13', 'Kiminki12'):
            if args.period == 'Sana12':
                slope, pmin, pmax_def = -0.55, 0.15, 3.5
            elif args.period == 'Sana13':
                slope, pmin, pmax_def = -0.45, 0.15, 3.5
            elif args.period == 'Kiminki12':
                slope, pmin, pmax_def = 0.1, 0., 3.
            if pmax is None:
                pmax = pmax_def
            else:
                pmax = np.log10(pmax * 365.25)
            var = np.random.rand(nbinaries)
            if slope == -1:
                self.arr['period'] = 10. ** (pmin ** (1 - var) * pmax ** var) / 365.25
            else:
                self.arr['period'] =  10. ** ((var * (pmax ** (slope + 1) - pmin ** (slope + 1)) + pmin ** (slope + 1)) ** (1. / (slope + 1))) / 365.25
        elif isinstance(period, basestring):
            raise ValueError("Period distribution '%s' not found." % period)
        else:
            self.arr['period'] = period

    def draw_mass_ratio(self, args, mass_ratio='flat', qmin=None, qmax=None):
        """Draw a new mass ratio distribution.

        The mass ratio is the ratio of the mass of the secondary star to the primary star, where the observed star is defined as the primary and its companion as the secondary.

        Either provide an array (with the same length as the number of binaries) or choose from:
        - 'flat': Flat distribution between 0 and 1.
        - 'Reggiani13': Powerlaw distribution between 0.1 and 1 from Reggiani & Meyer (2013, A&A, 553, 124).
        - 'Kiminki12': Powerlaw distribution from Kiminki & Kobulnicky (2012, ApJ, 751, 4).
        - 'Sana12': Powerlaw distribution between 0.1 and 1 from Sana et al. (2012, Science, 337, 444).
        - 'Sana13': Powerlaw distribution from Sana et al. (2013, A&A, 550, 107).

        Arguments:
        - `mass_ratio`: New mass ratio distribution (array or name of distribution).
        - `qmin`: Overrides the default minimum mass ratio from the adopted distribution.
        - `qmax`: Overrides the default maximum mass ratio from the adopted distribution.
        """
        nbinaries = self.arr.size
        if args.mass_ratio in ('flat', 'Reggiani13', 'Sana12', 'Sana13', 'Kiminki12'):
            if args.mass_ratio == 'flat':
                slope, qmin_def, qmax_def = 0., 0., 1.
            elif args.mass_ratio == 'Reggiani13':
                slope, qmin_def, qmax_def = 0.25, 0.1, 1.
            elif args.mass_ratio == 'Sana12':
                slope, qmin_def, qmax_def = -0.1, 0.1, 1.
            elif args.mass_ratio == 'Sana13':
                slope, qmin_def, qmax_def = -1., 0.1, 1.
            elif args.mass_ratio == 'Kiminki12':
                slope, qmin_def, qmax_def = 0.1, 0.005, 1.
            if qmin is None:
                qmin = qmin_def
            if qmax is None:
                qmax = qmax_def
            var = np.random.rand(nbinaries)
            if slope == -1:
                self.arr['mass_ratio'] = qmin ** (1 - var) * qmax ** var
            else:
                self.arr['mass_ratio'] =  (var * (qmax ** (slope + 1) - qmin ** (slope + 1)) + qmin ** (slope + 1)) ** (1. / (slope + 1))
        elif isinstance(mass_ratio, basestring):
            raise ValueError("Mass ratio distribution '%s' not found" % mass_ratio)
        else:
            self.arr['mass_ratio'] = mass_ratio

    def draw_eccentricities(self, args, eccentricity='flat', emin=0., emax='tidal'):
        """Draw a new eccentricity distribution.

        Either provide an array (with the same length as the number of binaries) or choose from:
        - 'flat': Flat distribution between `emin` and `emax`.
        - 'thermal`: f(e) ~ e^2 between `emin` and `emax`.

        If `emax` is set to 'tidal' it will depend on the period of the binary to mimic tidal circulirization through:
        emax = max(emin, 0.5 * (0.95 + tanh(0.6 * log_10(period in days) - 1.7)))
        If this setting is chosen the eccentricities will have to be redrawn if the periods are redrawn.

        Arguments:
        - `eccentricity`: New eccentricity distribution (array or name of distriution).
        - `emin`: Minimum eccentricity (default: 0.)
        - `emax`: Maximum eccentricity (default: set by tidal circularization)
        """
        nbinaries = self.arr.size
        if emax == 'tidal':
            emax =  .5 * (0.95 + np.tanh(0.6 * np.log10(self.arr['period'] * 365.25) - 1.7))
            emax[emax < emin] = emin
        if eccentricity == 'flat':
            self.arr['eccentricity'] = np.random.rand(nbinaries) * (emax - emin) + emin
        elif eccentricity == 'thermal':
            self.arr['eccentricity'] = np.sqrt(np.random.rand(nbinaries) * (emax ** 2. - emin ** 2.) + emin ** 2.)
        elif isinstance(eccentricity, basestring):
            raise ValueError("Eccentricity distribution '%s' not found" % eccentricity)
        else:
            self.arr['eccentricity'] = eccentricity

    def semi_major(self, mass=1.):
        """Returns the semi-major axis of the binaries in AU.

        Arguments:
        - `mass`: primary mass in solar masses."""
        return (self.arr['period'] ** 2 * mass * (1 + self.arr['mass_ratio'])) ** (1. / 3.)

    def velocity(self, mass, time=0., anomaly_offset=1e-3):
        """Returns the radial velocities and proper motions in km/s.

        Returns an (N, 2) array with the radial velocities and the proper motions due to the binary orbital motions of the N binaries.

        Arguments:
        - `mass`: primary mass of the star in solar masses.
        - `time`:
        """
        nbinaries = self.arr.size
        mean_anomaly = (self.arr['phase'] + time / self.arr['period']) * 2. * np.pi
        ecc_anomaly = mean_anomaly
        old = np.zeros(nbinaries) - 1.
        count_iterations = 0
        while (np.abs(ecc_anomaly - old) > anomaly_offset).any() and count_iterations < 20:
            old = ecc_anomaly
            ecc_anomaly = ecc_anomaly - (ecc_anomaly - self.arr['eccentricity'] * np.sin(ecc_anomaly) - mean_anomaly) / (1. - self.arr['eccentricity'] * np.cos(ecc_anomaly))
            count_iterations += 1

        theta_orb = 2. * sp.arctan(np.sqrt((1. + self.arr['eccentricity']) / (1. - self.arr['eccentricity'])) * np.tan(ecc_anomaly / 2.))
        seperation = (1 - self.arr['eccentricity'] ** 2) / (1 + self.arr['eccentricity'] * np.cos(theta_orb))
        thdot = 2 * np.pi * np.sqrt(1 - self.arr['eccentricity'] ** 2) / seperation ** 2
        rdot = seperation * self.arr['eccentricity'] * thdot * np.sin(theta_orb) / (1 + self.arr['eccentricity'] * np.cos(theta_orb))

        vtotsq = (thdot * seperation) ** 2 + rdot ** 2
        vlos = (thdot * seperation * np.sin(self.arr['theta'] - theta_orb) + rdot * np.cos(self.arr['theta'] - theta_orb)) * np.sin(self.arr['inclination'])
        vperp = np.sqrt(vtotsq - vlos ** 2)
        velocity = np.array([vlos, vperp]) * self.semi_major(mass) / (self.arr['period'] * (1 + 1 / self.arr['mass_ratio'])) * 4.74057581
        return velocity

    def single_epoch(self, velocity, sigvel, mass, F_yn, log_minv=-3, log_maxv=None, log_stepv=0.02):
        """Returns a callable Basefitter which computes the log-likelihood to reproduce the observed single-epoch radial velocity distribution.

        Uses the current settings of the binary properties to calculate the distribution of radial velocity offsets due to binary orbital motions.

        Arguments:
        - `velocity`: 1D array-like giving velocities in km/s.
        - `sigvel`: 1D array-like (or single number) giving measurement uncertainties in km/s.
        - `mass`: 1D array-like (or single number) giving best estimate for mass of the observed stars in solar masses.
        - `log_minv`: 10_log of the lowest velocity bin in km/s (should be significantly smaller than the velocity dispersion).
        - `log_maxv`: 10_log maximum of the largest velocity bin (default: logarithm of maximum velocity)
        - `log_stepv`: step size in 10_log(velocity) space.
        """
        vel = np.sort(np.sum(self.velocity(1.) ** 2., 0) ** .5)
        cum_weight = np.cumsum(1. / vel[::-1])[::-1]

        if log_maxv == None:
            log_maxv = np.log10(vel[-1])

        vbord = 10 ** np.arange(log_minv, log_maxv, log_stepv)
        ixbound = np.searchsorted(vel, vbord)

        pdist = []
        vtot = np.append(0, vbord)
        for ix in range(len(vbord)):
            lower = vtot[ix]
            upper = vtot[ix + 1]
            if ix == 0:
                vuse = vel[:ixbound[ix]]
            else:
                vuse = vel[ixbound[ix - 1]: ixbound[ix]]
            if ixbound[ix] == len(vel):
                est = 0.
            else:
                est = cum_weight[ixbound[ix]]
            pdist.append(est + np.sum((vuse - lower) / vuse) / (upper - lower))

        vbound = np.append(-vbord[::-1], np.append(0, vbord))
        prob = np.append(pdist[::-1], pdist) / 2.  / len(vel)

        if F_yn == 1:
            return BinaryFit(velocity, sigvel, mass, vbound, prob)
        else:
            return BinaryFit_noF(velocity, sigvel, mass, vbound, prob)
        # return BinaryFit(velocity, sigvel, mass, vbound, prob)

    def multi_epoch(self, velocity, sigvel, mass, dates, pfalse=1e-4, log_minv=-3, log_maxv=4, log_stepv=0.02):
        """Returns a callable Basefitter which computes the log-likelihood to reproduce the observed multi-epoch radial velocity distribution.

        Uses the current settings of the binary properties to calculate the distribution of radial velocity offsets due to binary orbital motions.

        Arguments:
        - `velocity`: list-like with for every star an array-like containing the observed radial velocities in km/s.
        - `sigvel`: list-like with for every star an array-like containing the measurement uncertainties in km/s.
        - `mass`: 1D array-like (or single number) giving best estimate for mass of the observed stars in solar masses.
        - `dates`: list-like with for every star an array-like containing the date of observations in years.
        - `pfalse`: probability of false detection (i.e. detecting a single star as a binary). Lowering this will decrease the number of detected binaries.
        - `log_minv`: 10_log of the lowest velocity bin in km/s (should be significantly smaller than the velocity dispersion).
        - `log_maxv`: 10_log maximum of the largest velocity bin.
        - `log_stepv`: step size in 10_log(velocity) space.
        """
        if sp.asarray(mass).ndim == 0:
            mass = [mass] * len(velocity)
        for test_length in ('sigvel', 'mass', 'dates'):
            if len(locals()[test_length]) != len(velocity):
                raise ValueError('%s does not have the same length as the velocity list' % test_length)
        unique_dates = sp.unique(reduce(sp.append, dates))
        vbin = {}
        for date in unique_dates:
            vbin.update({date: self.velocity(1., time=date)[0]})

        vmean = []
        sigmean = []
        single_mass = []
        pdet_single = []
        pdet_rvvar = []
        pbin = []
        is_single = []
        vbord = 10 ** sp.arange(log_minv, log_maxv, log_stepv)
        vbound = sp.append(-vbord[::-1], sp.append(0, vbord))

        for mult_vel, mult_sigvel, pmass, epochs in zip(velocity, sigvel, mass, dates):
            epochs, mult_vel, mult_sigvel = sp.broadcast_arrays(epochs, mult_vel, mult_sigvel)
            if epochs.size == 1:
                mean_rv = mult_vel[0]
                mean_sig = mult_sigvel[0]
                rv_binoffset = vbin[epochs[0]]# + sp.randn(self.size) * mult_sigvel[0]
                pdet = 0.
                rvvariable = False
            else:
                weight = mult_sigvel ** -2
                rv_offset_per_epoch = sp.zeros((self.size, len(epochs)))
                rv_binoffset_per_epoch = sp.zeros((self.size, len(epochs)))
                for ixepoch, (date, sv) in enumerate(zip(epochs, mult_sigvel)):
                    rv_binoffset_per_epoch[:, ixepoch] = vbin[date]
                    rv_offset_per_epoch[:, ixepoch] = rv_binoffset_per_epoch[:, ixepoch] * pmass ** (1. / 3.)  + sp.randn(self.size) * sv
                rv_offset_mean = sp.sum(rv_offset_per_epoch * weight[sp.newaxis, :], -1) / sp.sum(weight)
                chisq = sp.sum((rv_offset_per_epoch - rv_offset_mean[:, sp.newaxis]) ** 2. * weight[sp.newaxis, :], -1)
                isdetected = sp.stats.chisqprob(chisq, len(epochs) - 1) < pfalse
                pdet = np.float(sp.sum(isdetected)) / isdetected.size
                rv_binoffset = (sp.sum(rv_binoffset_per_epoch * weight[sp.newaxis, :], -1) / sp.sum(weight))[~isdetected]

                mean_rv = sp.sum(mult_vel * weight) / sp.sum(weight)
                mean_sig = sp.sum(weight) ** -.5
                rvvariable = sp.stats.chisqprob(sp.sum((mean_rv - mult_vel) ** 2 * weight), len(epochs) - 1) < pfalse
            if rvvariable:
                pdet_rvvar.append(pdet)
            else:
                vmean.append(mean_rv)
                sigmean.append(mean_sig)
                pdet_single.append(pdet)
                single_mass.append(pmass)
                prob_bin = sp.histogram(abs(rv_binoffset), bins=sp.append(0, vbord))[0] * 1. / rv_binoffset.size
                pbin.append(sp.append(prob_bin[::-1], prob_bin) / 2. / (vbound[1:] - vbound[:-1]))
            is_single.append(not rvvariable)
        pbin = sp.array(pbin).T
        vmean = sp.array(vmean)
        sigmean = sp.array(sigmean)
        single_mass = sp.array(single_mass)
        pdet_single = sp.array(pdet_single)
        pdet_rvvar = sp.array(pdet_rvvar)
        is_single = sp.array(is_single, dtype='bool')

        return fitter.BinaryFit(vmean, sigmean, single_mass, vbound, pbin, pdet_single, pdet_rvvar, is_single)

    def fake_dataset(self, nvel, vdisp, fbin, sigvel, mass=1., dates=(0., ), vmean=0.):
        """Creates a fake single-epoch radial velocity data for Monte Carlo simulations.

        Note that multiple calls will use the same binary properties. Redraw the binary properties to get a new set.

        Arguments:
        - `nvel`: number of velocities to draw.
        - `vdisp`: velocity dispersion of the cluster in km/s.
        - `fbin`: binary fraction among observed stars.
        - `sigvel`: array-like of size `nvel` or single number; measurement uncertainty in km/s (undistinguishable from velocity dispersion for single epoch case).
        - `mass`: array-like of size `nvel` or single number; mass of observed stars in solar masses.
        - `dates`: iterable with relative dates of observations in years. Creates a one-dimensional single-epoch dataset if the iterable has a length of one.
        - `vmean`: mean velocity in km/s.
        """
        v_systematic = sp.randn(nvel) * vdisp
        v_bin_offset = sp.array([self[:nvel].velocity(mass, time)[0, :] for time in dates])
        v_bin_offset[:, sp.rand(nvel) > fbin] = 0.
        v_meas_offset = sp.randn(v_bin_offset.size).reshape(v_bin_offset.shape) * sp.atleast_1d(sigvel)[:, sp.newaxis]
        return sp.squeeze(v_systematic[sp.newaxis, :] + v_bin_offset + v_meas_offset)
