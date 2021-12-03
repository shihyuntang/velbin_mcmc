"""
MCMC use functions

by Shih-Yun Tang Dec. 2, 2020
"""
import numpy as np
import corner, emcee
import matplotlib.pyplot as plt

fig_dpi      = 300
fig_typeface = 'Helvetica'
fig_family   = 'monospace'
fig_style    = 'normal'

# log likelihood functions
def ln_pm(x, pm, pm_sig):
    f_c  = 0.95
    pm_mean, pm_disp, pm_mean_f, pm_disp_f = x
    likelihood_single = f_c    * np.exp(-(pm - pm_mean) ** 2 / (2 * (pm_sig ** 2. + pm_disp ** 2.))) / np.sqrt(2 * np.pi * (pm_sig ** 2. + pm_disp ** 2.)) + \
                        (1-f_c)* np.exp(-(pm - pm_mean_f) ** 2 / (2 * (pm_sig ** 2. + pm_disp_f ** 2.))) / np.sqrt(2 * np.pi * (pm_sig** 2. + pm_disp_f ** 2.))

    result = np.where(likelihood_single > 1E-5, likelihood_single, 1E-10)
    return np.sum(np.log(result))


# def ln_pmdec(x, pmdec, sig_pmdec):
#     f_c  = 0.95
#     pmdec_mean, pmdec_disp, pmdec_mean_f, pmdec_disp_f = x
#     likelihood_single = f_c    * np.exp(-(pmdec - pmdec_mean) ** 2 / (2 * (sig_pmdec ** 2. + pmdec_disp ** 2.))) / np.sqrt(2 * np.pi * (sig_pmdec ** 2. + pmdec_disp ** 2.)) + \
#                         (1-f_c)* np.exp(-(pmdec - pmdec_mean_f) ** 2 / (2 * (sig_pmdec ** 2. + pmdec_disp_f ** 2.))) / np.sqrt(2 * np.pi * (sig_pmdec** 2. + pmdec_disp_f ** 2.))

#     result = np.where(likelihood_single > 1E-5, likelihood_single, 1E-10)
#     return np.sum(np.log(result))


def ln_rv(velocity, sigvel, mass, F_yn):
    global args
    nbinaries = np.int(1e6)
    all_binaries = solar(args, nbinaries=nbinaries )
    lnlike = all_binaries.single_epoch(velocity, sigvel, mass, F_yn, log_minv=-3, log_maxv=None, log_stepv=0.02)
    return lnlike




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

    f.savefig(f'./figs/{pngsave_name}_{partype}_mcmc_3.png', format='png', bbox_inches='tight')
 
    
    
    
    