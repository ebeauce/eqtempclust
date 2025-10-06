import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging, warnings

import scipy.special as scispec
import scipy.optimize

from scipy.optimize import curve_fit
from scipy.special import gammainc, gammaincc
from scipy.stats import linregress
from functools import partial


def build_time_intervals(time_range, tau_min, tau_max, base_log=2.0):
    """Build the vector of log-spaced time interval sizes.

    Parameters
    ----------
    time_range : float
        Time interval of the study, in the same units of time than
        `tau_min` and `tau_max`.
    tau_min : float
        Smallest time bin size to test.
    tau_max : float
        Largest time bin size to test.
    base_log : float, optional
        The time bin sizes are log-spaced using the log-`base_log` basis.
        Default is 2.

    Returns
    -------
    tau : numpy.ndarray
        Array with log-spaced time bin sizes between `tau_min` and `tau_max`.
    nbins : numpy.ndarray
        Array with the number of bins required to cover the entire `time_range`
        for each bin size.
    """
    n_max = np.ceil(np.log(time_range / tau_min) / np.log(base_log))
    tau = float(base_log) ** (
        np.log(time_range) / np.log(base_log) - np.arange(int(n_max) + 1)
    )
    if tau.max() > tau_max:
        tau = tau[tau < tau_max]
        tau = np.hstack(([tau_max], tau))
    tau = np.sort(tau)
    nbins = np.int64(time_range / tau + 0.001)
    tau = time_range / np.float64(nbins)
    return tau, nbins


def compute_occupation_probability(
    eq_timings,
    normalized_tau_min=0.001,
    normalized_tau_max=None,
    base_log=2,
    return_normalized_times=True,
    use_valid_only=1,
    min_num_events=5,
    shortest_resolved_interevent_time=5.0,
    num_resamplings=50,
    confidence_level_pct=95.,
    mini_batch=1000000,
):
    """Compute the occupation probability

    Parameters
    ----------
    eq_timings : numpy.ndarray
        Earthquake timings, in arbitrary units (but, in general, seconds).
    normalized_tau_min : float, optional
        Smallest bin size in units of average inter-event time. Default is 0.001.
    normalized_tau_max : float, optional
        Largest bin size in units of average inter-event time.
        Default is None. If `None`, `normalized_tau_max = max_waiting_time / avg_waiting_time`.
    base_log : float, optional
        The time bin sizes are log-spaced using the log-`base_log` basis.
        Default is 2.
    return_normalized_times : boolean, optional
        If True (default), returns the time bin sizes in units of the
        average inter-event time. If False, the same units of `eq_timings`
        are used.
    use_valid_only : int, optional
        Integer taking a value in [0, 1, 2]. Defaults to 1.
        - 0: Use all time bins between `normalized_tau_min` and
          `normalized_tau_max`
        - 1: Do not use time bins below which the maximum number of earthquakes
          in any given bin is exactly 1. Below that, the occupation probability
          trivially decreases proportionally to `tau`.
        - 2: Do not use time bins below which the maximum number of earthquakes
          in any given bin is exactly 1. In addition, do not use time bins that
          are shorter than `shortest_resolved_interevent_time`.
    min_num_events : integer, optional
        The minimum number of events in order to compute the occupation probability.
        Should be at least 2 to not produce an error.
    shortest_resolved_interevent_time : scalar, optional
        Shortest inter-event time resolved in the catalog, in units of
        `eq_timings`. This number is used to issue a warning if
        `normalized_tau_min` pushes the analysis below the resolution limit.
    num_resamplings : int, optional
        Number of bootstrap replica used to estimate the uncertainty on
            `Phi(tau)`
        Defaults to 50.
    confidence_level_pct : float, optional
        The returned arrays, `Phi_lower` and `Phi_upper`, are the lower
        and upper `(100-confidence_level_pct)/2` percentile of `Phi` from
        bootstrapping.
    mini_batch : int, optional
        Bootstrapping on very large arrays, which happens when computing `Phi`
        for a very small bin size, can be memory consuming. Bootstrap estimates
        of `Phi` are computed by sampling `mini_batch` elements at a time, thus
        reducing the memory print of bootstrapping.

    Returns
    -------
    Phi : numpy.ndarray
        Observed occupation probability at the time bin sizes defined by `tau`.
    Phi_lower : numpy.ndarray
        Lower bound of the `(100 - interval_uncertainty_pct)`% confidence
        interval on `Phi`.
    Phi_upper : numpy.ndarray
        Upper bound of the `(100 - interval_uncertainty_pct)`% confidence
        interval on `Phi`.
    tau : numpy.ndarray
        Time bin sizes at which the occupation probability is computed. These
        times are normalized if `return_normalized_times=True`, that is, given
        in units of the average inter-event time.
    """
    assert min_num_events >= 2, print("min_num_events should be at least 2!")
    if len(eq_timings) < min_num_events:
        print(f"Not enough events (only {len(eq_timings)})!")
        return

    interval_uncertainty_pct = 100. - confidence_level_pct

    # measure the average seismicty rate
    eq_timings = eq_timings - min(eq_timings)
    min_eq_timing, max_eq_timing = min(eq_timings), max(eq_timings)
    # average rate is taken as the inverse of the average waiting time
    waiting_times = eq_timings[1:] - eq_timings[:-1]
    if use_valid_only == 2:
        average_waiting_time = get_mean_interevent_time(
                waiting_times[waiting_times >= shortest_resolved_interevent_time]
                )
    else:
        average_waiting_time = get_mean_interevent_time(waiting_times)

    average_rate = 1. / average_waiting_time
    shortest_interval_size = normalized_tau_min * average_waiting_time
    if shortest_interval_size < shortest_resolved_interevent_time:
        if use_valid_only == 2:
            normalized_tau_min = (
                    shortest_resolved_interevent_time / average_waiting_time
                    )
            shortest_interval_size = normalized_tau_min * average_waiting_time
        else:
            suggested_increase = shortest_resolved_interevent_time / shortest_interval_size
            warnings.warn(
                "Warning! You are computing the occupation probability for"
                " time intervals shorter than your shortest resolved "
                "inter-event time. You should increase `normalized_tau_min`"
                f" by a factor {suggested_increase:.2f}."
            )
    normalized_eq_timings = eq_timings / average_waiting_time
    min_normalized_eq_timing = min(normalized_eq_timings)
    max_normalized_eq_timing = max(normalized_eq_timings)
    # ----------------------------------
    normalized_time_range = max_normalized_eq_timing - min_normalized_eq_timing
    if normalized_tau_max is None:
        normalized_tau_max = normalized_time_range
    normalized_tau, nbins = build_time_intervals(
        normalized_time_range, normalized_tau_min, normalized_tau_max, base_log=base_log
    )
    #print(f"Inside compute_occ: largest time interval is: {normalized_tau.max():.2f}")
    # -------------------------------
    rescaled_eq_timings = (normalized_eq_timings - min_normalized_eq_timing) / (
        max_normalized_eq_timing - min_normalized_eq_timing
    )
    # force the last earthquake to be part of the last bin rather than
    # being exactly at time=1
    max_time = rescaled_eq_timings.max() - normalized_tau_min / 100.0
    rescaled_eq_timings[rescaled_eq_timings > max_time] = max_time
    # -------------------------------

    Phi = np.zeros(len(normalized_tau), dtype=np.float32)
    Phi_lower = np.zeros(len(normalized_tau), dtype=np.float32)
    Phi_upper = np.zeros(len(normalized_tau), dtype=np.float32)
    valid = np.ones(len(normalized_tau), dtype=bool)

    rng = np.random.default_rng()
    for i in range(len(normalized_tau)):
        n_tau = np.bincount(np.int64(rescaled_eq_timings * nbins[i]))
        if n_tau.max() == 1:
            # we reached the bin size when there is at most one
            # event per bin, for smaller bins, the occupation
            # probability will trivially decrease a 1/tau
            # print(f"Minimum relevant bin size: {normalized_tau[i]:.2e}")
            valid[i] = False
        occupied_bins = n_tau != 0
        del n_tau
        num_bins = len(occupied_bins)
        Phi[i] = np.sum(occupied_bins) / np.float64(nbins[i])
        Phi_rsmpl = np.zeros(num_resamplings, dtype=np.float32)
        for j in range(num_resamplings):
            summed_occupied = 0.0
            num_mini_batches = num_bins // mini_batch
            remain = num_bins % mini_batch
            for k in range(num_mini_batches):
                summed_occupied += np.sum(
                    rng.choice(occupied_bins, size=mini_batch, replace=True)
                )
            summed_occupied += np.sum(
                rng.choice(occupied_bins, size=remain, replace=True)
            )
            Phi_rsmpl[j] = summed_occupied / float(num_bins)
        Phi_lower[i] = np.percentile(Phi_rsmpl, interval_uncertainty_pct / 2.0)
        Phi_upper[i] = np.percentile(Phi_rsmpl, 100.0 - interval_uncertainty_pct / 2.0)
        del occupied_bins
    if use_valid_only > 0:
        # only keep non-trivial interval sizes
        # keep the largest time bins for which num_max == 1 happens for the 1st time
        (Phi, Phi_lower, Phi_upper, normalized_tau) = (
            Phi[valid],
            Phi_lower[valid],
            Phi_upper[valid],
            normalized_tau[valid],
        )
    if return_normalized_times:
        return Phi, Phi_lower, Phi_upper, normalized_tau
    else:
        return Phi, Phi_lower, Phi_upper, normalized_tau * average_waiting_time


def compute_wt_pdf_from_occupation(tau, Phi, full_output=False):
    """
    Compute the waiting time probability density function (PDF) from an occupation function.

    Parameters
    ----------
    tau : numpy.ndarray
        Array of time values at which the occupation function is defined.
    Phi : numpy.ndarray
        Array of occupation function values corresponding to `tau`.
    full_output : bool, optional
        If True, return additional arrays tau_for_dPhi and dPhi, and tau_for_ddPhi and ddPhi.
        Defaults to False.
    """
    tau, indexes = np.unique(tau, return_index=True)
    Phi = Phi[indexes]

    order = np.argsort(tau)
    tau = tau[order]
    Phi = Phi[order]

    dtau = tau[1:] - tau[:-1]
    dPhi = (Phi[1:] - Phi[:-1]) / dtau
    tau_for_dPhi = tau[:-1] + dtau / 2.0

    ddtau = tau_for_dPhi[1:] - tau_for_dPhi[:-1]
    ddPhi = (dPhi[1:] - dPhi[:-1]) / ddtau
    tau_for_ddPhi = tau_for_dPhi[:-1] + ddtau / 2.0

    wt_pdf = -ddPhi

    if full_output:
        return tau_for_ddPhi, wt_pdf, tau_for_dPhi, dPhi
    else:
        return tau_for_ddPhi, wt_pdf


def fractal_analysis(*args, **kwargs):
    """Alias for `occupation_analysis`.

    Here for legacy reasons.
    """
    return occupation_analysis(*args, **kwargs)

def get_mean_interevent_time(interevent_times, method="bootstrap", **kwargs):
    """Estimate of mean inter-event time.

    Parameters
    ----------
    interevent_times : array-like
        List or 1-d array of inter-event times.
    method : str, optional
        Either of 'median', 'bootstrap' or 'naive'.
    """
    if method == "median":
        med = np.median(interevent_times)
        # then, approximate mean under Poissonian conditions
        mean = med / np.log(2.)
    elif method == "bootstrap":
        num_bootstraps = kwargs.get("num_bootstraps", 25)
        means = np.zeros(num_bootstraps, dtype=np.float64)
        for i in range(num_bootstraps):
            means[i] = np.random.choice(
                    interevent_times, size=len(interevent_times), replace=True
                    ).mean()
        # the mean is an unbiased estimator, so, theoretically, the average 
        # value of the mean operator over many replica should be the true 
        # mean value
        mean = np.mean(means)
    elif method == "naive":
        mean = interevent_times.mean()
    return mean

def occupation_analysis(
    eq_timings,
    normalized_tau_min=0.001,
    normalized_tau_max=None,
    normalized_tau_max_fit=None,
    base_log=2.0,
    return_normalized_times=True,
    min_num_events=5,
    use_valid_only=1,
    shortest_resolved_interevent_time=5.0,
    model="fractal",
    ax=None,
    verbose=1,
    return_figure=False,
    plot_above=1.0,
    confidence_level_pct=95.0,
    num_resamplings=50,
    pre_computed_Phi=None,
    **kwargs,
):
    """Compute the fractal dimension.

    Parameters
    -----------
    eq_timings : numpy.ndarray
        Earthquake timings, in arbitrary units (but, in general, seconds).
    normalized_tau_min : float, optional
        Smallest bin size in units of average inter-event time.
        Default is 0.001.
    normalized_tau_max : float, optional
        Largest bin size in units of average inter-event time.
        Default is None. If `None`, `normalized_tau_max = max_waiting_time / avg_waiting_time`.
    normalized_tau_max_fit : float, optional
        If not None (default), is the maximum bin size used for fitting the model.
    base_log : float, optional
        The time bin sizes are log-spaced using the log-`base_log` basis.
        Default is 2.
    return_normalized_times : boolean, optional
        If True (default), returns the time bin sizes in units of the
        average inter-event time. If False, the same units of `eq_timings`
        are used.
    min_num_events : integer, optional
        The minimum number of events in order to compute the occupation probability.
    shortest_resolved_interevent_time : scalar, optional
        Shortest inter-event time resolved in the catalog, in units of
        `eq_timings`. This number is used to issue a warning if
        `normalized_tau_min` pushes the analysis below the resolution limit.
        Defaults to 5.0.a
    use_valid_only : int, optional
        Integer taking a value in [0, 1, 2]. Defaults to 1.
        - 0: Use all time bins between `normalized_tau_min` and
          `normalized_tau_max`
        - 1: Do not use time bins below which the maximum number of earthquakes
          in any given bin is exactly 1. Below that, the occupation probability
          trivially decreases proportionally to `tau`.
        - 2: Do not use time bins below which the maximum number of earthquakes
          in any given bin is exactly 1. In addition, do not use time bins that
          are shorter than `shortest_resolved_interevent_time`.
    model : string, optional
        One of 'linear_spline', 'fractal' (default) or 'gamma'.
        See `fit_occupation_probability' for more details.
    confidence_level_pct : float, optional
        Confidence level computed for the occupation probability. Defaults to 95.
    num_resamplings: integer, optional
        Number of bootstrap resamplings used to estimate the confidence intervals.
        Defaults to 50.
    pre_computed_Phi : dict, optional
        Dictionary with pre-computed occupation probability. Must contain
        'Phi', 'Phi_lower', 'Phi_upper' and 'tau', as computed by
        `compute_occupation_probability`. Defaults to None.
    ax : matplotlib.pyplot.Axis, optional
        If not None (default), uses `ax` to plot the x(tau) curve for both the
        observed and synthetic poissonian seismicity.
    verbose : int, optional
        If 1 (default), print warning messages. If 0, does not print anything.
    plot_above : float, optional
        Plot the graph of occupied fraction vs time bin size if the
        fractal dimension is above `plot_above`. Default to 1 (no plotting).
    return_figure : boolean, optional
        If True, return the figure.

    Returns
    ---------
    tau : numpy.ndarray
        Time bin sizes used to divide the time axis.
    Phi : numpy.ndarray
        Occupation probability at the `tau` bin sizes.
    Phi_std : numpy.ndarray
        Standard deviation of the occupation probability at the `tau` bin sizes.
    occupation_parameters : dictionary
        Dictionary with model parameters fitted to the occupation probability.
    """
    # average rate is taken as the inverse of the average waiting time
    waiting_times = eq_timings[1:] - eq_timings[:-1]
    if use_valid_only == 2:
        average_waiting_time = get_mean_interevent_time(
                waiting_times[waiting_times >= shortest_resolved_interevent_time]
                )
    else:
        average_waiting_time = get_mean_interevent_time(waiting_times)
    earthquake_rate = 1. / average_waiting_time

    # measure occupation probability
    if pre_computed_Phi is not None:
        Phi = pre_computed_Phi["Phi"]
        Phi_lower = pre_computed_Phi["Phi_lower"]
        Phi_upper = pre_computed_Phi["Phi_upper"]
        normalized_tau = pre_computed_Phi["tau"]
    else:
        Phi, Phi_lower, Phi_upper, normalized_tau = compute_occupation_probability(
            eq_timings,
            normalized_tau_min=normalized_tau_min,
            normalized_tau_max=normalized_tau_max,
            base_log=base_log,
            return_normalized_times=True,
            use_valid_only=use_valid_only,
            shortest_resolved_interevent_time=shortest_resolved_interevent_time,
            num_resamplings=num_resamplings,
            confidence_level_pct=confidence_level_pct,
        )

    # model occupation probability
    occupation_parameters = fit_occupation_probability(
        Phi,
        normalized_tau,
        normalized_tau_min,
        tau_max=normalized_tau_max_fit,
        model=model,
        average_rate=1.0,
        **kwargs,
    )

    if not return_normalized_times:
        tau = normalized_tau * average_waiting_time
    else:
        tau = normalized_tau
        earthquake_rate = 1.0
    Phi_poisson = 1.0 - np.exp(-earthquake_rate * tau)

    # -----------------------------------------------
    # figure (should I remove this part?)
    if (
        ("fractal_dim" in occupation_parameters)
        and ((occupation_parameters["fractal_dim"] > plot_above) or (ax is not None))
    ) or (
        "gamma" in occupation_parameters
        and ((1.0 - occupation_parameters["gamma"] > plot_above) or (ax is not None))
    ):
        if ax is None:
            fig = plt.figure(
                f"occupation_analysis_{eq_timings[-1]}",
                figsize=kwargs.get("figsize", (8, 8)),
            )
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        datetime = pd.Timestamp(eq_timings[-1] * 1.0e9)
        ax.set_title(datetime)
        color = kwargs.get("color", "C0")
        marker = kwargs.get("marker", "o")
        color_poisson = kwargs.get("color_poisson", "dimgrey")
        marker_poisson = kwargs.get("marker_poisson", "d")
        if model == "linear_spline":
            print(f"{occupation_parameters['log_tau_c']:.2f}")
            ax.plot(
                tau,
                10.0
                ** linear_spline(
                    1 - occupation_parameters["fractal_dim"],
                    occupation_parameters["log_tau_c"],
                    np.log10(tau),
                ),
                ls="--",
                color=color,
                label=f'Fractal Dimension D={occupation_parameters["fractal_dim"]:.3f}'
                r"$\pm$"
                f'{occupation_parameters["fractal_dim_err"]:.3f}',
            )
        elif model == "inverse_function" or model == "fractal":
            ax.plot(
                tau,
                10.0
                ** inverse_function_log(
                    tau,
                    1 - occupation_parameters["fractal_dim"],
                    10.0 ** occupation_parameters["log_tau_c"],
                    occupation_parameters["gamma"],
                ),
                ls="--",
                color=color,
                label=f'Fractal Dimension D={occupation_parameters["fractal_dim"]:.3f}'
                r"$\pm$"
                f'{occupation_parameters["fractal_dim_err"]:.3f}',
            )
        elif model == "gamma":
            occ_gamma = occupation_parameters["gamma"]
            label = (
                r"Model $\Phi(\tau)$, $\gamma=$"
                f"{occ_gamma:.2f}"
                r" ($1/\gamma=$"
                f"{1/occ_gamma:.2f})"
            )
            ax.plot(
                tau,
                occupation_probability_gamma_model(
                    tau,
                    occupation_parameters["gamma"],
                    beta=occupation_parameters["beta"],
                    lamb=earthquake_rate,
                ),
                ls="--",
                color=color,
                label=label,
            )
        ax.plot(
            tau,
            Phi,
            marker=marker,
            color=color,
            ls="",
            label=kwargs.get("label", "Observed"),
        )
        ax.fill_between(tau, Phi_lower, Phi_upper, color=color, alpha=0.25)
        ax.plot(
            tau,
            Phi_poisson,
            marker=marker_poisson,
            color=color_poisson,
            ls="",
            alpha=0.66,
            label=kwargs.get("label_poisson", "Synthetic Poisson"),
        )
        ax.loglog()
        if return_normalized_times:
            ax.set_xlabel(r"Normalized time interval, $\lambda \tau$")
            ax.set_ylabel(r"Occupation probability, $\Phi(\lambda \tau)$")
        else:
            ax.set_xlabel(r"Time interval, $\tau$")
            ax.set_ylabel(r"Occupation probability, $\Phi(\tau)$")
        ax.legend(loc="lower right")
    # -----------------------------------------------

    occupation_parameters["distance_from_poisson"] = kullback_leibler_divergence(
        Phi_poisson, Phi
    )
    occupation_parameters["wt_mean"] = average_waiting_time
    if return_figure:
        return tau, Phi, Phi_lower, Phi_upper, occupation_parameters, fig
    else:
        return tau, Phi, Phi_lower, Phi_upper, occupation_parameters


def lacunarity(eq_timings, bin_size, starttime=None, endtime=None):
    """Compute the lacunarity of an earthquake sequence.

    WORK-IN-PROGRESS
    """
    warnings.warn("Not implemented yet!")
    if starttime is None:
        starttime = min(eq_timings)
    if endtime is None:
        endtime = max(eq_timings)
    bins = np.arange(starttime, endtime + 0.5 * bin_size, bin_size)
    hist, _ = np.histogram(eq_timings, bins=bins)
    binary_hist = np.int32(hist > 0.0)
    return


def fit_occupation_probability(
    Phi,
    tau,
    tau_min,
    tau_max=None,
    model="fractal",
    loss="l2_log",
    average_rate=None,
    fix_beta=False,
    **kwargs,
):
    """
    Fit the occupation probability Phi(tau) in the log-log space.

    Parameters
    ----------
    Phi : array_like
        The fraction of occupied bins.
    tau : array_like
        The size of the bins.
    tau_min : float
        The minimum size bin used to fit the model.
    tau_max : float, optional
        If not None (default), this is the maximum bin size used to fit the model.
    model: string, optional
        One of `fractal` (default), ``gamma` or `linear_spline`.
        - `fractal` (default): use the fractal model
                `Phi(tau) = 1 / (1 + (tau_c/tau)**(alpha * n))**(1/alpha)`
          where `n` is the small `tau` asymptotic power-law exponent and is
          related to the fractal dimension through `n = 1 - D_tau`, `alpha`
          controls the large time behavior and `tau_c` define the transition
          between the short and large time behaviors.
        - `linear_spline`: use the linear spline model in loglog domain
                `log Phi(log tau) = a * log tau + b` if `tau < tau_c`
                `log Phi(log tau)` = 1` if `tau >= tau_c`
        - `gamma`: use the gamma model
                `Phi(tau) = lambda * tau * Q(gamma, lambda*gamma*tau)
                            + P(gamma + 1, lambda*gamma*tau)`
          where `Q` and `P` are the regularized upper and lower incomplete
          gamma functions, respectively, `lambda` is the average earthquake rate
          and `gamma` is the power-law and rate parameter of the gamma function
          describing the inter-event time distribution.
    loss : str, optional
        Either of 'l2_log' or 'relative_entropy'.
        - 'l2_log': Minimize the l2-norm of the residuals between the
          logarithm of the model and the logarithm of the observations.
        - 'relative_entropy': Minimize the relative entropy between the modeled
          and the observed occupation probability. Defaults to 'l2_log'.
    average_rate : float, optional
        Used only if `model='gamma'`. The average seismicity rate used if times
        are not normalized by the average inter-event time. Defaults to None.
    fix_beta : boolean, optional
        Used only if `model='gamma'`. If True, beta is set to be 1/gamma.

    Returns
    -------
    dict
        A dictionary containing the model parameters. See `model` for more details
        on the parameters.
    """
    from functools import partial
    import scipy.stats

    # legacy:
    if "method" in kwargs:
        model = kwargs["model"]

    assert model in ["linear_spline", "fractal", "inverse_function", "gamma"], print(
        "`model` should be one of 'linear_spline', 'fractal',\
                    'inverse_function' or 'gamma'"
    )

    indexes = np.argsort(tau)
    tau = tau[indexes]
    Phi = Phi[indexes]
    log_Phi = np.log10(Phi)
    log_tau = np.log10(tau)
    log_tau_min = np.log10(tau_min)
    occupation_parameters = {}

    param_bounds = {
        "n_min": 0.0,
        "n_max": 1.0,
        "alpha_min": 0.25,
        "alpha_max": 25.00,
        "gamma_min": 0.0,
        "gamma_max": 1.0,
        "beta_min": 0.0,
        "beta_max": np.inf,
    }

    # -------------------------------------------------------------
    #                     deprecated model?
    if model == "linear_spline":
        log_tau_range = log_tau.max() - np.log10(tau_min).min()
        if tau_max is None:
            log_tau_c_max = log_tau.max() - 0.01 * log_tau_range
        else:
            log_tau_c_max = np.log10(tau_max)
        log_tau_c_min = log_tau[tau > tau_min][0] + 0.01 * log_tau_range
        best_sol_rms = np.inf
        best_sol_stderr = np.inf
        best_sol_tau_c = 0.0
        best_sol_D = 0.0
        n_in = np.sum((log_tau >= log_tau_c_min) & (log_tau <= log_tau_c_max))
        range_log_tau_c = np.linspace(log_tau_c_min, log_tau_c_max, 2 * n_in)
        for log_tau_c in range_log_tau_c:
            # print(f'Cutoff time bin: {log_tau_c:.2f}')
            selection = (tau >= tau_min) & (log_tau <= log_tau_c)
            weights = np.ones(np.sum(selection))
            # give less weight to region around cutoff
            weights[(log_tau[selection] > -0.5) & (log_tau[selection] < 0.5)] = 0.2
            alpha, a1, stderr = weighted_linear_regression(
                log_tau[selection], log_Phi[selection], W=weights
            )
            rms = np.mean((log_Phi - linear_spline(alpha, log_tau_c, log_tau)) ** 2)
            if rms < best_sol_rms:
                best_sol_rms = rms
                best_sol_stderr = stderr
                best_sol_tau_c = log_tau_c
                best_sol_D = 1.0 - alpha
        occupation_parameters["fractal_dim"] = best_sol_D
        occupation_parameters["fractal_dim_err"] = best_sol_stderr
        occupation_parameters["log_tau_c"] = best_sol_tau_c
        occupation_parameters["rms"] = best_sol_rms
    # -------------------------------------------------------------
    #                    fractal model
    elif model == "fractal":
        selection = tau >= tau_min
        # first guess parameters
        D_random = 0.1
        tau_c = 1.0
        sharpness = 2.0
        # model with data-dependent theta_min
        p0 = [1.0 - D_random, tau_c, sharpness]
        # bounds (bounds_min, bounds_max)
        if loss == "relative_entropy":
            # define model
            fractal_occupation = partial(
                occupation_probability_fractal_model, log=False
            )
            loss = lambda params: scipy.stats.entropy(
                Phi[selection],
                qk=fractal_occupation(tau[selection], *params),
            )
            bounds = [
                (param_bounds["n_min"], param_bounds["n_max"]),
                (min(tau), max(tau)),
                (param_bounds["alpha_min"], param_bounds["alpha_max"]),
            ]
            # call minimize
            results = scipy.optimize.minimize(
                loss,
                p0,
                bounds=bounds,
            )
            popt = results.x
            perr = np.sqrt(np.diag(results.hess_inv.todense()) * results.fun)
        elif loss == "l2_log":
            # define model
            fractal_occupation_log = partial(
                occupation_probability_fractal_model, log=True
            )
            bounds = (
                (param_bounds["n_min"], min(tau), param_bounds["alpha_min"]),
                (param_bounds["n_max"], max(tau), param_bounds["alpha_max"]),
            )
            # call curve_fit
            popt, pcov = curve_fit(
                fractal_occupation_log,
                tau[selection],
                log_Phi[selection],
                p0=p0,
                bounds=bounds,
                **kwargs,
            )
            perr = np.sqrt(np.diag(pcov))
        else:
            print("loss should be either of:")
            print("'relative_entropy', 'l2_log'")
            return

        squared_res = np.mean(
            (
                log_Phi[selection]
                - occupation_probability_fractal_model(tau[selection], *popt, log=True)
            )
            ** 2
        )
        rms = np.sqrt(squared_res)
        var0 = np.var(log_Phi[selection])
        var_reduction = 1.0 - squared_res / var0

        occupation_parameters["n"] = popt[0]
        occupation_parameters["n_err"] = perr[0]
        occupation_parameters["fractal_dim"] = 1.0 - popt[0]
        occupation_parameters["fractal_dim_err"] = perr[0]
        occupation_parameters["D_tau"] = occupation_parameters["fractal_dim"]
        occupation_parameters["D_tau_err"] = occupation_parameters["fractal_dim_err"]
        # model with data-dependent min waiting time:
        occupation_parameters["tau_c"] = popt[1]
        occupation_parameters["tau_c_err"] = perr[1]
        occupation_parameters["log_tau_c"] = np.log10(popt[1])
        occupation_parameters["log_tau_c_err"] = abs(1.0 / popt[1]) * perr[1]
        occupation_parameters["alpha"] = popt[2]
        occupation_parameters["alpha_err"] = perr[2]
        occupation_parameters["tau_min"] = tau_min_fractal(
            *popt,
        )

        occupation_parameters["rms"] = rms
        occupation_parameters["var_reduction"] = var_reduction
    # -------------------------------------------------------------
    #                gamma model
    elif model == "gamma":
        selection = tau >= tau_min
        gamma0 = 0.67
        beta0 = 1.0 / gamma0
        if fix_beta:
            p0 = [gamma0]
            var_names = ["gamma"]
        else:
            p0 = [gamma0, beta0]
            var_names = ["gamma", "beta"]
        # gamma cannot be smaller than 0 because, otherwise, the exponential
        # tail is increasing instead of decreasing
        if loss == "l2_log":
            # define model
            gamma_occupation_log = partial(
                occupation_probability_gamma_model, lamb=average_rate, log=True
            )
            if fix_beta:
                bounds = (param_bounds["gamma_min"], param_bounds["gamma_max"])
            else:
                bounds = (
                    (param_bounds["gamma_min"], param_bounds["beta_min"]),
                    (param_bounds["gamma_max"], param_bounds["beta_max"]),
                )
            popt, pcov = curve_fit(
                gamma_occupation_log,
                tau[selection],
                log_Phi[selection],
                p0=p0,
                bounds=bounds,
                **kwargs,
            )
            perr = np.sqrt(np.diag(pcov))
        elif loss == "relative_entropy":
            gamma_occupation = partial(
                occupation_probability_gamma_model, lamb=average_rate, log=False
            )
            loss = lambda params: scipy.stats.entropy(
                Phi[selection],
                qk=gamma_occupation(tau[selection], params[0]),
            )
            # call minimize
            if fix_beta:
                bounds = [(param_bounds["gamma_min"], param_bounds["gamma_max"])]
            else:
                bounds = [
                    (param_bounds["gamma_min"], param_bounds["gamma_max"]),
                    (param_bounds["beta_min"], param_bounds["beta_max"]),
                ]
            results = scipy.optimize.minimize(loss, p0, bounds=bounds)
            popt = results.x
            perr = np.sqrt(np.diag(results.hess_inv.todense()) * results.fun)
        else:
            print("loss should be either of:")
            print("'relative_entropy', 'l2_log'")
            return

        squared_res = np.mean(
            (log_Phi[selection] - occupation_probability_gamma_model(
                tau[selection], *popt, lamb=average_rate, log=True
                )) ** 2
        )
        rms = np.sqrt(squared_res)
        var0 = np.var(log_Phi[selection])
        var_reduction = 1.0 - squared_res / var0

        for i, attr in enumerate(var_names):
            occupation_parameters[attr] = popt[i]
            occupation_parameters[f"{attr}_err"] = perr[i]
        occupation_parameters["rms"] = rms
        occupation_parameters["var_reduction"] = var_reduction
    return occupation_parameters


# ===============================================================
#               UTILITY FUNCTIONS
# ===============================================================


def linear_spline(b1, log_tau_c, log_tau):
    """Linear spline model of occupation probability."""
    spline1 = log_tau <= log_tau_c
    spline2 = ~spline1
    log_x = np.zeros(len(log_tau), dtype=np.float32)
    log_x[spline1] = b1 * (log_tau[spline1] - log_tau_c)
    return log_x


def linear_regression(x, y):
    """
    cf. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html

    Returns
    -------
    a: slope
    b: intercept
    r_val: correlation coefficient, usually
           people use the coefficient of determination
           R**2 = r_val**2 to measure the quality of
           the fit
    p_val: two-sided p-value for a hypothesis test whose null
           hypothesis is that the slope is zero
    std_err: standard error of the estimated slope
    """
    from scipy.stats import linregress

    a, b, r_val, p_val, std_err = linregress(x, y)
    return a, b, r_val, p_val, std_err


def weighted_linear_regression(X, Y, W=None):
    """
    Parameters
    -----------
    X: (n,) numpy array or list
    Y: (n,) numpy array or list
    W: default to None, (n,) numpy array or list

    Returns
    --------
    best_slope: scalar float,
        Best slope from the least square formula
    best_intercept: scalar float,
        Best intercept from the least square formula
    std_err: scalar float,
        Error on the slope
    """
    X = np.asarray(X)
    if W is None:
        W = np.ones(X.size)
    W_sum = W.sum()
    x_mean = np.sum(W * X) / W_sum
    y_mean = np.sum(W * Y) / W_sum
    x_var = np.sum(W * (X - x_mean) ** 2)
    xy_cov = np.sum(W * (X - x_mean) * (Y - y_mean))
    best_slope = xy_cov / x_var
    best_intercept = y_mean - best_slope * x_mean
    # errors in best_slope and best_intercept
    estimate = best_intercept + best_slope * X
    s2 = sum(estimate - Y) ** 2 / (Y.size - 2)
    s2_intercept = s2 * (1.0 / X.size + x_mean**2 / ((X.size - 1) * x_var))
    s2_slope = s2 * (1.0 / ((X.size - 1) * x_var))
    return best_slope, best_intercept, np.sqrt(s2_slope)


def kullback_leibler_divergence(ref_density_function, test_density_function):
    """
    Computes the Kullback-Leibler divergence, also called relative entropy.
    """
    valid = (ref_density_function != 0.0) & (test_density_function != 0.0)
    norm = test_density_function.sum()
    unnormalized_kl_div = np.sum(
        ref_density_function * np.log(ref_density_function / test_density_function)
    )
    return unnormalized_kl_div / norm


def lower_incomplete_gamma(a, x):
    return scispec.gamma(a) * scispec.gammainc(a, x)


def upper_incomplete_gamma(a, x):
    return scispec.gamma(a) * scispec.gammaincc(a, x)


def _dlogPhi_dlogtau(tau, gamma):
    gtau = gamma * tau
    loglog_slope = 1.0 + (
        gtau ** (gamma + 1.0) / lower_incomplete_gamma(gamma + 1.0, gtau)
        - gtau**gamma / upper_incomplete_gamma(gamma, gtau)
    ) * np.exp(-gtau)
    return loglog_slope

def interevent_pdf(
    ie_times,
    nbins=10,
    bins=None,
    min_events_per_bin=0,
    return_midbins=True,
    num_resamplings=50,
    confidence_level_pct=95.,
):
    """
    Compute the interevent time probability density function (PDF).

    Parameters
    ----------
    ie_times : array_like
        A 1D array containing the interevent times.
    nbins : int, optional
        The number of bins to use when creating the histogram of interevent times.
        Default is 10.
    bins : array_like, optional
        The bin edges to use when creating the histogram of interevent times.
        Overrides the `nbins` parameter if provided.
    return_midbins : bool, optional
        Whether to return the midpoints of the bins as the second output.
        If False, returns the bin edges instead.
        Default is True.
    min_events_per_bin : int, optional
        Minimum number of events in a bin to estimate the pdf. Defaults to 0.
    num_resamplings : int, optional
        Number of bootstrap replica used to estimate the uncertainty on
            `ie_pdf(w)`
        Defaults to 50.
    confidence_level_pct : float, optional
        The returned arrays, `Phi_lower` and `Phi_upper`, are the lower
        and upper `(100-confidence_level_pct)/2` percentile of `Phi` from
        bootstrapping.

    Returns
    -------
    ie_pdf : ndarray
        The interevent time probability density function (PDF).
    ie_pdf_lower : numpy.ndarray
        Lower bound of the `confidence_level_pct`% confidence
        interval on `ie_pdf`.
    ie_pdf_upper : numpy.ndarray
        Upper bound of the `confidence_level_pct`% confidence
        interval on `ie_pdf`.
    ie_bins : ndarray
        The bin midpoints or edges used to compute the PDF, depending on the
        `return_midbins` parameter.

    Notes
    -----
    The interevent time PDF is calculated by first computing a histogram of the
    interevent times using the provided `nbins` or `bins` parameters. The bin
    sizes are then used to normalize the histogram to create the PDF.
    """
    interval_uncertainty_pct = 100. - confidence_level_pct

    if bins is None:
        ie_time_bins = np.logspace(
            np.log10(ie_times.min()), np.log10(ie_times.max()), nbins
        )
    else:
        ie_time_bins = bins
    num_events = len(ie_times)

    # ---------------------------------------
    # estimate pdf on the whole data set
    ie_times_count, _ = np.histogram(ie_times, bins=ie_time_bins)
    # if not enough events in a bin, don't trust the statistic
    non_trusted_bins = ie_times_count < min_events_per_bin
    if np.sum(non_trusted_bins) > 0:
        print(f"Not enough events in bins: {ie_time_bins[:-1][non_trusted_bins]}")
    ie_times_count[ie_times_count < min_events_per_bin] = 0
    bin_sizes = ie_time_bins[1:] - ie_time_bins[:-1]
    ie_pdf = ie_times_count / (bin_sizes * ie_times_count.sum())

    # ---------------------------------------
    # estimate pdf uncertainties with bootstrapping
    rng = np.random.default_rng()
    ie_pdf_b = np.zeros((num_resamplings, len(ie_pdf)), dtype=np.float64)
    for i in range(num_resamplings):
        ie_times_b = rng.choice(ie_times, size=num_events, replace=True)
        ie_times_count_b, _ = np.histogram(ie_times_b, bins=ie_time_bins)
        # if not enough events in a bin, don't trust the statistic
        non_trusted_bins = ie_times_count_b < min_events_per_bin
        ie_times_count_b[ie_times_count_b < min_events_per_bin] = 0
        ie_pdf_b[i, :] = ie_times_count_b / (bin_sizes * ie_times_count_b.sum())
    ie_pdf_lower = np.zeros(len(ie_pdf), dtype=np.float64)
    ie_pdf_upper = np.zeros(len(ie_pdf), dtype=np.float64)
    for j in range(len(ie_pdf)):
        valid = ie_pdf_b[:, j] > 0.0
        if np.sum(valid) == 0:
            ie_pdf_lower[j], ie_pdf_upper[j] = 0.0, 0.0
            continue
        ie_pdf_lower[j] = np.percentile(
            ie_pdf_b[valid, j],
            interval_uncertainty_pct / 2.0,
        )
        ie_pdf_upper[j] = np.percentile(
            ie_pdf_b[valid, j],
            100.0 - interval_uncertainty_pct / 2.0,
        )

    if return_midbins:
        ie_time_log_bins = np.log10(ie_time_bins)
        ie_time_bins = 10.**((ie_time_log_bins[1:] + ie_time_log_bins[:-1]) / 2.0)
    return ie_pdf, ie_pdf_lower, ie_pdf_upper, ie_time_bins


def fit_interevent_pdf(
    ie_times,
    pdf,
    tau_max_fit=1.0,
    tau_min_fit=None,
    #normalized_waiting_times=True,
    model="powerlaw",
    loss="l2_log",
    fix_beta=True,
    min_pdf=1.0e-6,
    **kwargs,
):
    """
    Fit a power law to the interevent time probability density function (PDF).

    Parameters
    ----------
    ie_times : array_like
        A 1D array containing the interevent times.
    pdf : array_like
        A 1D array containing the probability density function (PDF) of the
        interevent times.
    tau_min_fit : scalar, optional
        The minimum inter-event time used for fitting the model.
        Default is None, in which case the whole range given by
        `ie_times` is used.
    tau_max_fit : scalar, optional
        The maximum inter-event time used for fitting the model.
        Default is 1, assuming the inter-event times are normalized.
        If None, uses the whole range given by `ie_times`.
    model : string, optional
        Can either be 'fractal', 'powerlaw' or 'gamma'.
    loss : string, optional
        Either of 'l2_log' or 'relative_entropy'.
        - 'l2_log' (default): Minimize the l2 norm of the difference
                              of the log(pdf).
        - 'relative_entropy': Minimize the relative entropy as
                              given by `scipy.stats.entropy`.
    fix_beta : boolean, optional
        Relevant only if model='gamma'. If True, the beta parameter
        is fixed such that beta = 1/gamma. Note that if the waiting
        times are normalized such that their mean is 1, then beta
        *should* be fixed to 1/gamma in order to satisfy the definition
        of a pdf. Default is True.
    min_pdf : float, optional
        Only pdf values above this threshold are considered when fitting.
        This is to avoid numerical issues related to log values of very
        small numbers.

    Returns
    -------
    inferred_parameter1 : float
        Value of optimal model parameter #1.
    inferred_parameter2 : float
       Value of optimal model parameter #2.
    ...
    err_parameter1 : float
       Error on inferred model parameter #1.
    err_parameter2 : float
       Error on inferred model parameter #2.
    ...
    """
    model = model.lower()
    assert model in [
        "powerlaw",
        "gamma",
        "fractal",
    ], "model should be 'powerlaw', 'gamma', or 'fractal'"

    param_bounds = {
        "n_min": 0.0,
        "n_max": 1.0,
        "alpha_min": 0.25,
        "alpha_max": 5.00,
        "gamma_min": 0.0 + 1.e-5,
        "gamma_max": 1.0,
        "beta_min": 0.0,
        "beta_max": np.inf,
    }

    if loss == "l2_log":
        model_kwargs = {"log": True}
    elif loss == "relative_entropy":
        model_kwargs = {"log": False}
    if model == "powerlaw":
        from scipy.stats import linregress

        normalized_ie_times = ie_times / ie_times.mean()
        valid = (pdf > min_pdf) & (normalized_ie_times < tau_max_fit)  # for empty bins
        slope, intercept, _, _, err = linregress(
            np.log10(ie_times[valid]), np.log10(pdf[valid])
        )
        return slope, intercept, err
    elif model == "gamma":
        valid = pdf > min_pdf
        #model_kwargs["normalized"] = normalized_waiting_times
        model_kwargs["C"] = "truncated"
        gamma0 = 0.67
        if fix_beta:
            # this is the only correct solution so that
            # the gamma function satisfies the pdf constraints
            p0 = [gamma0]
            model_kwargs["normalized"] = True
            model = lambda w, gamma: gamma_waiting_times(w, gamma, **model_kwargs)
            if loss == "l2_log":
                bounds = ((param_bounds["gamma_min"]), (param_bounds["gamma_max"]))
            elif loss == "relative_entropy":
                bounds = [
                    (param_bounds["gamma_min"], param_bounds["gamma_max"]),
                ]
        else:
            p0 = [gamma0, 1.0 / gamma0]
            model_kwargs["normalized"] = False
            model = lambda w, gamma, beta: gamma_waiting_times(
                w, gamma, beta=beta, **model_kwargs
            )
            if loss == "l2_log":
                bounds = (
                    (param_bounds["gamma_min"], param_bounds["beta_min"]),
                    (param_bounds["gamma_max"], param_bounds["beta_max"]),
                )
            elif loss == "relative_entropy":
                bounds = [
                    (param_bounds["gamma_min"], param_bounds["gamma_max"]),
                    (param_bounds["beta_min"], param_bounds["beta_max"]),
                ]
    elif model == "fractal":
        valid = pdf > min_pdf
        # first guess parameters
        D_random = 0.1
        tau_c_max = theoretical_tau_c(1.0 - D_random, ie_times.min())
        sharpness = 2.0
        # model with data-dependent theta_min
        p0 = [1.0 - D_random, tau_c_max, sharpness]
        model = lambda w, n, tau_c, alpha: fractal_waiting_times(
            w, n, tau_c, alpha, **model_kwargs
        )
        if loss == "l2_log":
            bounds = (
                (
                    param_bounds["n_min"],
                    min(ie_times[valid]),
                    param_bounds["alpha_min"],
                ),
                (
                    param_bounds["n_max"],
                    max(ie_times[valid]),
                    param_bounds["alpha_max"],
                ),
            )
        elif loss == "relative_entropy":
            bounds = [
                (param_bounds["n_min"], param_bounds["n_max"]),
                (min(ie_times[valid]), max(ie_times[valid])),
                (param_bounds["alpha_min"], param_bounds["alpha_max"]),
            ]
    if loss == "l2_log":
        popt, pcov = curve_fit(
            model,
            ie_times[valid],
            np.log10(pdf[valid]),
            p0=p0,
            bounds=bounds,
            **kwargs,
        )
        std_err = np.sqrt(np.diag(pcov))
    elif loss == "relative_entropy":
        loss = lambda params: scipy.stats.entropy(
            pdf[valid],
            qk=model(ie_times[valid], *params),
        )
        # call minimize
        results = scipy.optimize.minimize(
            loss,
            p0,
            bounds=bounds,
        )
        popt = results.x
        std_err = np.sqrt(np.diag(results.hess_inv.todense()) * results.fun)
    return *popt, *std_err


def occupation_Poissonian_uncertainty(
    tau, total_duration, num_events, num_possible_fractions=10
):
    from scipy.special import binom
    from tqdm import tqdm

    average_rate = num_events / total_duration
    n_bins = int(total_duration / tau)
    possible_fractions = np.arange(0, 1 + 0.90 / n_bins, 1.0 / n_bins)
    # possible_fractions = np.linspace(0., 1., num_possible_fractions)
    bins = np.arange(n_bins + 1)
    print(bins)
    print(possible_fractions * n_bins)
    uncertainty = np.zeros(len(bins), dtype=np.float32)
    for i in tqdm(range(len(bins)), desc="outer loop"):
        num_occupied_bins = bins[i]
        if num_occupied_bins == 0:
            uncertainty[i] = binom(n_bins, n_bins - num_occupied_bins)
            continue
        for j in range(num_occupied_bins):
            uncertainty[i] += (
                (-1.0) ** j
                * binom(num_occupied_bins, j)
                * (possible_fractions[i] - float(j) / n_bins) ** num_events
            )
        uncertainty[i] *= binom(n_bins, n_bins - num_occupied_bins)
    return possible_fractions, uncertainty


def compute_aic(x, model, num_params):
    """Akaike Information Criterion.

    !!! This function assumes that the parameters were
    inferred by optimizing a least-squares problem !!!

    `AIC = -2 * sum(log(likelihood)) + 2 * num_params`

    Parameters
    ----------
    x : numpy.ndarray
        Array of observations.
    model : func
        Function so that `likelihood=model(x)`.
    num_params : int
        Number of model parameters.

    Returns
    -------
    AIC : float
        The Akaike information criterion.
    """
    if len(x) == 0:
        return np.inf
    likelihood = model(x)
    argmax = likelihood.argmax()
    if np.isinf(likelihood[argmax]):
        print(f"Inf detected!!: waiting time was: {x[argmax]:.2e}")
    log_likelihood = np.log(likelihood[likelihood != 0.0])
    return -2.0 * np.sum(log_likelihood) + 2.0 * num_params


# ===============================================================
#                       Models
# ===============================================================


# ----------------------- fractal


def occupation_probability_fractal_model(tau, n, tau_c, alpha, log=False):
    """
    Compute the occupation probability of a fractal model for given interval sizes `tau`.

    Parameters
    ----------
    tau : array_like
        Array of time interval sizes.
    n : float
        Fractal dimensionality parameter.
    tau_c : float
        Cutoff time scale.
    alpha : float
        Scaling exponent which controls the sharpness of the cutoff.
    log : bool, optional
        Whether to return the logarithm of the occupation probability.
        Default is False.

    Returns
    -------
    array_like
        Occupation probability values corresponding to each interval size value.

    Notes
    -----
    The occupation probability of a fractal model as a function of
    interval size is given by:

    p(tau) = (1 + (tau_c / tau)^(n * alpha))^(-1 / alpha)  if log = False
    p(tau) = -1 / alpha * log10(1 + (tau_c / tau)^(n * alpha)) if log = True
    """
    if log:
        return -1.0 / alpha * np.log10(1.0 + (tau_c / tau) ** (n * alpha))
    else:
        return np.power(1.0 / (1.0 + (tau_c / tau) ** (n * alpha)), 1.0 / alpha)


def cdf_fractal(w, n, tau_c, alpha, lbd=1.0):
    """Cumulative distribution function of fractal model.

    Parameters
    ----------
    w : numpy.ndarray
        Inter-event times.
    n : float
        Short-time exponent: `n = 1 - D_tau`, where `D_tau` is the
        fractal dimension.
    tau_c : float
        Cross-over time scale.
    alpha : float
        Cross-ver sharpness.
    lbd : float, optional
        Average rate of seismicity. Defaults to 1.

    Returns
    -------
    cdf : numpy.ndarray
        Array of same shape as `w` with the cdf of the fractal model.
    """
    A = (tau_c / w) ** (n * alpha)
    return 1.0 - n / lbd * (1. / w) * A * (1 + A) ** (-1 / alpha - 1)

def tau_min_fractal2(n, tau_c, alpha):
    log_theta_min = (
            1. / (n * alpha) * np.log10((1. - n) / (1. + alpha))
            + np.log10(tau_c)
            )
    return 10.**log_theta_min


def tau_min_fractal(n, tau_c, alpha, lbd=1.0):
    """Minimum inter-event time in fractal model.

    Parameters
    ----------
    n : float
        Short-time exponent: `n = 1 - D_tau`, where `D_tau` is the
        fractal dimension.
    tau_c : float
        Cross-over time scale.
    alpha : float
        Cross-ver sharpness.
    lbd : float, optional
        Average rate of seismicity. Defaults to 1.

    Returns
    -------
    w_min : float
        Minimum inter-event time.
    """
    fun = partial(cdf_fractal, n=n, tau_c=tau_c, alpha=alpha, lbd=lbd)
    fprime = partial(fractal_waiting_times, n=n, tau_c=tau_c, alpha=alpha, lbd=lbd)
    p0 = float(tau_c)
    while fun(p0 / 10.0) > 0.0:
        p0 /= 10.0
    while fun(p0 / 2.0) > 0.0:
        p0 /= 2.0

    try:
        root = scipy.optimize.newton(fun, p0, maxiter=500, fprime=fprime)
        # root = scipy.optimize.newton(fun, p0, maxiter=500)
    except Exception as e:
        print(f"Could not determine tau_min, returning p0={p0:.2e}")
        print(e)
        return p0
    # print(root)
    return root


def fractal_waiting_times(w, n, tau_c, alpha, lbd=1.0, tau_min=None, log=False):
    """
    Parameters
    ----------
    w : numpy.ndarray
        Earthquake waiting times.
    n : float
        Short waiting times exponent. The asymptotic behavior of the pdf
        at short waiting times is entirely controlled by `n`:
            `pdf ~ w**-(2-n)`
        `n` is related to the "fractal dimension" `D_tau`: `n = 1 - D_tau`
    tau_c : float
        Cross-over time from the short waiting time behavior `pdf ~ w**-(2-n)`
        to the long waiting time behavior `pdf ~ w**-(2+n*alpha)`
    alpha : float
        Long waiting times exponent. The asymptotic behavior of the pdf
        at long waiting times is controlled both by `n` and `alpha`:
            `pdf ~ w**-(2+n*alpha)`
    lbd : float, optional
        Average rate of seismicity, that is, the inverse of the expected waiting
        time. Default to 1 (i.e., assumes that waiting times are normalized).
    tau_min : float or None, optional
        If not None (default), the support of the pdf is defined between `tau_min` and
        infinity. Note that this model usually requires `tau_min > 0` in order
        to satisfies the definition of a pdf.
    log : boolean, optional
        If True, returns the logarithm of pdf. Defaults to False.

    Returns
    -------
    pdf : numpy.ndarray
        Probability density function of waiting times. If `log=True`, it
        returns the logarithm (base 10) of the pdf.
    """
    w = np.asarray(w)
    omega = (tau_c / w) ** (n * alpha)
    pdf = (
        (n / (lbd * w**2))
        * omega
        * (1.0 / (1.0 + omega) ** (1.0 / alpha + 2.0))
        * (1.0 + n * alpha + (1.0 - n) * omega)
    )
    if tau_min is not None:
        # define the lower bound of the pdf's support
        pdf[w < tau_min] = 0.0
    if log:
        return np.log10(pdf)
    else:
        return pdf


def theoretical_norm_fractal(n, lbd=1.0):
    """Remove?
    """
    return 1.
    #return n / lbd


def truncated_norm_fractal(n, tau_c, alpha, tau_min, tau_max, lbd=1.0):
    """Norm of fractal pdf over a finite support.

    Parameters
    ----------
    n : float
        Short-time exponent: `n = 1 - D_tau`, where `D_tau` is the
        fractal dimension.
    tau_c : float
        Cross-over time scale.
    alpha : float
        Cross-ver sharpness.
    tau_min : float
        Lower bound of support.
    tau_max : float
        Upper bound of support.
    lbd : float, optional
        Average rate of seismicity. Defaults to 1.

    Returns
    -------
    norm : float
        The inverse of the integral of the pdf over the finite support.
    """
    integral_over_finite_support = cdf_fractal(
        tau_max, n, tau_c, alpha, lbd=lbd
    ) - cdf_fractal(tau_min, n, tau_c, alpha, lbd=lbd)
    return 1.0 / integral_over_finite_support


def estimate_sample_rate_vs_real_rate(wt_bins, pdf, n, theta_c, alpha):
    """Estimate the real rate of seismicity using misfit between model and data.

    After correcting for the finiteness of the support of the observed pdf,
    the remaining misfit between the data and the model is assumed to be
    caused by the discrepancy between the empirical rate of seismicity and
    the real one. Thus, we can estimate the real rate of seismicity.

    Parameters
    ----------
    wt_bins : numpy.ndarray
        Normalized waiting time bins.
    pdf : numpy.ndarray
        Empirical probability density function.
    n : float
        Parameter of the fractal model.
    theta_c : float
        Parameter of the fractal model.
    alpha : float
        Parameter of the fractal model.

    Returns
    -------
    hat_rate_vs_real_rate : float
        Presumably, the ratio between the empirical rate of seismicity `hat_lbd`
        and the real one `lbd`.
    """
    # correction factor for the fractal distribution, taking into account that
    # the empirical distribution is normalized over a finite range of waiting times
    trunc_norm = truncated_norm_fractal(
        n,
        theta_c,
        alpha,
        wt_bins.min(),
        wt_bins.max(),
    )
    # theo_norm = theoretical_norm_fractal(
    #    n,
    # )
    # correction = theo_norm / trunc_norm
    # if the ratio `lbd_hat / lbd` were to be 1, then `pdf * correction`
    # should match the model
    # integrate the pdf over its finite support
    # make sure bins are in increasing order
    indexes = np.argsort(wt_bins)
    wt_bins = wt_bins[indexes]
    pdf = pdf[indexes]
    bin_width = wt_bins[1:] - wt_bins[:-1]
    pdf_midbin = (pdf[1:] + pdf[:-1]) / 2.0
    integral = np.sum(pdf_midbin * bin_width)
    # integral should be equal to trunc_norm,
    # but, instead, there's a difference due to
    # the fact that lbd_hat / lbd_real is not exactly 1
    hat_rate_vs_real_rate = integral * trunc_norm
    return hat_rate_vs_real_rate


# -------------------------- gamma


def occupation_probability_unconstrained_gamma_model(tau, gamma, beta, lamb, log=False):
    Phi_tau = (lamb * tau) / (beta * gamma) * scispec.gammaincc(
        gamma, lamb * tau / beta
    ) + scispec.gammainc(gamma + 1.0, lamb * tau / beta)
    if log:
        return np.log10(Phi_tau)
    else:
        return Phi_tau


def occupation_probability_gamma_model(tau, gamma, beta=None, lamb=None, log=False):
    """Occupation probability of a time interval with length `tau`.

    Parameters
    ----------
    tau : float or numpy.ndarray
        Length of time interval at which the occupation probability
        is evaluated.
    gamma : float
        Gamma model exponent.
    beta: float, optional
        Cross-over time scale of gamma model. If None (default), beta
        is taken to be `beta=1/gamma`, which implies that the average
        inter-event time is 1.
    lamb : float, optional
        Average rate of seismicity. If None (default), we assume it is 1.
    log : boolean, optional
        If True, returns the logarithm of the occupation probability.
        Defaults to False.

    Returns
    -------
    Phi_tau : float or numpy.ndarray
        Occupation probability evaluated at `tau`.
        If `log=True`, returns the logarithm of the occupation probability.
    """
    if lamb is None:
        lamb = 1.0
    if beta is None and np.abs(lamb - 1.0) < 0.01:
        # lambda = 1, beta = 1/gamma
        Phi_tau = tau * scispec.gammaincc(gamma, gamma * tau) + scispec.gammainc(
            gamma + 1.0, gamma * tau
        )
    elif beta is None:
        # beta = 1/gamma
        Phi_tau = (lamb * tau) * scispec.gammaincc(
            gamma, lamb * gamma * tau
        ) + scispec.gammainc(gamma + 1.0, lamb * gamma * tau)
    elif lamb is None or np.abs(lamb - 1.0) < 0.01:
        # lamb = 1
        Phi_tau = tau / (beta * gamma) * scispec.gammaincc(
            gamma, tau / beta
        ) + scispec.gammainc(gamma + 1.0, tau / beta)
    else:
        Phi_tau = (lamb * tau) / (beta * gamma) * scispec.gammaincc(
            gamma, lamb * tau / beta
        ) + scispec.gammainc(gamma + 1.0, lamb * tau / beta)
    if log:
        return np.log10(Phi_tau)
    else:
        return Phi_tau

def cdf_gamma(w, gamma, beta, lbd=1.):
    """Cumulative distribution function of the gamma model.

    Parameters
    ----------
    w : float or numpy.ndarray
        Inter-event time(s).
    gamma : float
        Gamma model exponent.
    beta: float, optional
        Cross-over time scale of gamma model. If None (default), beta
        is taken to be `beta=1/gamma`, which implies that the average
        inter-event time is 1.
    lamb : float, optional
        Average rate of seismicity. If None (default), we assume it is 1.

    Returns
    -------
    cdf : float or numpy.ndarray
        The cdf of the gamma model evaluated at `w`.
    """
    return scispec.gammainc(gamma, lbd * w / beta)

def inverse_cdf_gamma(r, gamma, beta, lbd=1.):
    """Inverse of gamma cdf using `scipy.special.gammaincinv`.

    Parameters
    ----------
    r : float or numpy.ndarray
        Value(s) of the cumulative distribution function.
    gamma : float
        Gamma model exponent.
    beta: float, optional
        Cross-over time scale of gamma model. If None (default), beta
        is taken to be `beta=1/gamma`, which implies that the average
        inter-event time is 1.
    lamb : float, optional
        Average rate of seismicity. If None (default), we assume it is 1.

    Returns
    -------
    inverse_cdf : float or numpy.ndarray
        The inverse of the cdf of the gamma model equal to `cdf=r`.
    """
    return beta / lbd * scispec.gammaincinv(gamma, r)

def truncated_norm_gamma(gamma, beta, tau_min, tau_max):
    """Normalization pre-factor in the gamma pdf over a truncated support.

    Calculates the normalization pre-factor so that the integral
    of the pdf over its truncated support, from `tau_min` to `tau_max` is 1.

    Parameters
    ----------
    gamma : float
        Shape parameter.
    beta : float
        Rate parameter.
    tau_min : float
        Lower bound of support.
    tau_max : float
        Upper bound of support.

    Returns
    -------
    float
        Normalization pre-factor.

    """
    return beta ** (-gamma) / (
        upper_incomplete_gamma(gamma, tau_min / beta)
        - upper_incomplete_gamma(gamma, tau_max / beta)
    )


def theoretical_norm_gamma(gamma, beta):
    """Normalization pre-factor in the gamma pdf.

    Calculates the normalization pre-factor so that the integral
    of the pdf over its support is 1.

    Parameters
    ----------
    gamma : float
        Shape parameter.
    beta : float
        Rate parameter.

    Returns
    -------
    float
        Normalization pre-factor.
    """
    return beta ** (-gamma) / scispec.gamma(gamma)


def gamma_waiting_times(
    waiting_time, gamma, beta=None, normalized=True, log=False, C="theoretical"
):
    """
    Returns the value of the gamma distribution function at a given point.

    Parameters
    ----------
    waiting_time : float
        The waiting times at which to evaluate the gamma distribution function.
    gamma : float
        The shape parameter of the gamma distribution.
    beta : float
        The scale parameter of the gamma distribution.
    normalized : boolean, optional
        If True, assumes that the waiting times are normalized so that their
        expected value is 1. It follows that the two parameters of the gamma law
        are dependent and `beta = 1/gamma`.
        Note: `normalized=True` could just be achieved with `beta=None`.
    log : boolean, optional
        If True, returns the base-10 logarithm of the gamma distribution.
        Defaults to False.
    C : string, optional
        Either of 'theoretical' or 'truncated'.
        - 'theoretical': Uses the theoretical normalization constant, that is,
          the constant that ensures that the pdf integrates to 1 over its
          theoretical support (0 to +infinity).
        - 'truncated': Uses the normalization constant for a truncated support
          defined by (`waiting_time.min()`, `waiting_time.max()`).
        Defaults to 'theoretical'.

    Returns
    -------
    float
        The value of the gamma distribution function evaluated at `tau`.
    """
    if normalized:
        # assume that the average waiting time is one
        # therefore beta = 1/gamma
        beta = 1.0 / gamma
    elif beta is None:
        raise ("beta should not be None is the average waiting time is not one")
    if C == "theoretical":
        C = theoretical_norm_gamma(gamma, beta)
    elif C == "truncated":
        C = truncated_norm_gamma(gamma, beta, waiting_time.min(), waiting_time.max())
    if log:
        return (
                np.log10(C) + (gamma - 1.0) * np.log10(waiting_time)
                - (waiting_time / beta) * np.log10(np.exp(1.)) 
                )
    else:
        return C * waiting_time ** (gamma - 1.0) * np.exp(-waiting_time / beta)

def timestamp_range(start_date, end_date, freq):
    """A date range returned as timestamps (seconds).

    Parameters
    -----------
    start_date: string or datetime
        Start of the date range.
    end_date: string or datetime
        End of the date range.
    freq: string
        String specifying the time step and time unit of the step. For
        example, '2.0S' or '1D'.

    Returns
    ---------
    date_range: (n_times,) float numpy.ndarray
        Array with regularly spaced timestamps.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    timestamps = [time.timestamp() for time in date_range]
    return np.float64(timestamps)


# ------------------------------------------------------------
#               wrapper for whole workflow
# ------------------------------------------------------------


def run_occupation_analysis1(
    timings,
    normalized_tau_min,
    normalized_tau_max,
    min_num_events=5,
    nbins_wt=20,
    shortest_resolved_time=5.0,
    use_valid_only=1,
    fix_beta=True,
    loss_pdf="relative_entropy",
    loss_phi="l2_log",
    base_log=2.0,
    num_resamplings=50,
):
    """
    Perform occupation analysis on event timings and inter-event times.

    Fit the fractal model to the occupation probability and the gamma model
    to the inter-event time distribution.

    Parameters:
    -----------
    timings : numpy.ndarray
        Array of event timings.
    normalized_tau_min : float
        Minimum normalized time bin size and inter-event time.
    normalized_tau_max : float
        Maximum normalized time bin size and inter-event time.
    min_num_events : int, optional
        Minimum number of events for analysis (default is 5).
    nbins_wt : int, optional
        Number of bins for the inter-event time PDF (default is 20).
    shortest_resolved_time : float, optional
        Minimum resolved inter-event time (default is 5.0).
    use_valid_only : int, optional
        Integer taking a value in [0, 1, 2]. Defaults to 1.
        - 0: Use all time bins between `normalized_tau_min` and
          `normalized_tau_max`
        - 1: Do not use time bins below which the maximum number of earthquakes
          in any given bin is exactly 1. Below that, the occupation probability
          trivially decreases proportionally to `tau`.
        - 2: Do not use time bins below which the maximum number of earthquakes
          in any given bin is exactly 1. In addition, do not use time bins that
          are shorter than `shortest_resolved_interevent_time`.
    fix_beta : bool, optional
        Whether to fix the beta parameter in the gamma model (default is True).
    loss_pdf : str, optional
        Loss function for inter-event time PDF fitting (default is "relative_entropy").
    loss_phi : str, optional
        Loss function for occupation probability fractal model fitting (default is "l2_log").
    base_log : float, optional
        Base for logarithm used in bin log spacing (default is 2.0).
    num_resamplings : int, optional
        Number of resampling iterations (default is 50).

    Returns:
    --------
    dict
        A dictionary containing the results of the occupation analysis, including
        occupation probability, inter-event time PDF, gamma model parameters, and
        fractal model parameters.

    Notes:
    ------
    This function computes occupation probability, inter-event time PDF, fits a gamma model
    to the PDF, and fits a fractal model to occupation probability. It returns the results
    as a dictionary.
    """
    output = {}
    # compute occupation probability
    Phi, Phi_lower, Phi_upper, tau = compute_occupation_probability(
        timings,
        normalized_tau_min=normalized_tau_min,
        normalized_tau_max=normalized_tau_max,
        base_log=base_log,
        min_num_events=min_num_events,
        use_valid_only=use_valid_only,
        shortest_resolved_interevent_time=shortest_resolved_time,
        num_resamplings=num_resamplings,
    )
    output["Phi"] = Phi
    output["Phi_lower"] = Phi_lower
    output["Phi_upper"] = Phi_upper
    output["tau"] = tau
    # compute inter-event time pdf
    waiting_times = timings[1:] - timings[:-1]
    wt_bins = np.logspace(np.log10(tau.min()), np.log10(tau.max()), nbins_wt)
    wt_pdf, wt_pdf_lower, wt_pdf_upper, wt_bins = interevent_pdf(
        waiting_times / waiting_times.mean(), return_midbins=True, bins=wt_bins
    )
    output["wt_pdf"] = wt_pdf
    output["wt_pdf_lower"] = wt_pdf_lower
    output["wt_pdf_upper"] = wt_pdf_upper
    output["wt_bins"] = wt_bins

    # fit gamma model to inter-event time pdf
    if fix_beta:
        output["gamma"], output["gamma_err"] = fit_interevent_pdf(
            wt_bins,
            wt_pdf,
            tau_max_fit=normalized_tau_max,
            tau_min_fit=normalized_tau_min,
            #normalized_waiting_times=True,
            model="gamma",
            fix_beta=fix_beta,
            loss=loss_pdf,
        )
        output["beta"] = 1.0 / output["gamma"]
    else:
        (
            output["gamma"],
            output["beta"],
            output["gamma_err"],
            output["beta_err"],
        ) = fit_interevent_pdf(
            wt_bins,
            wt_pdf,
            tau_max_fit=normalized_tau_max,
            tau_min_fit=normalized_tau_min,
            #normalized_waiting_times=True,
            model="gamma",
            fix_beta=fix_beta,
            loss=loss_pdf,
        )

    # fit fractal model to occupation probability
    _, _, _, _, fractal_model_parameters = occupation_analysis(
        timings,
        plot_above=2.0,
        model="fractal",
        normalized_tau_min=normalized_tau_min,
        normalized_tau_max=normalized_tau_max,
        base_log=base_log,
        shortest_resolved_interevent_time=shortest_resolved_time,
        use_valid_only=use_valid_only,
        return_figure=False,
        loss=loss_phi,
        pre_computed_Phi=output
    )

    output["n"] = fractal_model_parameters["n"]
    output["n_err"] = fractal_model_parameters["n_err"]
    output["tau_c"] = fractal_model_parameters["tau_c"]
    output["tau_c_err"] = fractal_model_parameters["tau_c_err"]
    output["alpha"] = fractal_model_parameters["alpha"]
    output["alpha_err"] = fractal_model_parameters["alpha_err"]
    output["fractal_rms"] = fractal_model_parameters["rms"]
    output["tau_min"] = fractal_model_parameters["tau_min"]
    output["D_tau"] = 1.0 - output["n"]
    output["D_tau_err"] = output["n_err"]
    output["rms"] = fractal_model_parameters["rms"]
    output["var_reduction"] = fractal_model_parameters["var_reduction"]

    # compute inter-event time pdf
    waiting_times = timings[1:] - timings[:-1]
    output["wt_mean"] = fractal_model_parameters["wt_mean"]
    output["wt_cov"] = np.std(waiting_times) / output["wt_mean"]
    normalized_waiting_times = waiting_times / output["wt_mean"]
    wt_bins = np.logspace(np.log10(tau.min()), np.log10(tau.max()), nbins_wt)
    wt_pdf, wt_pdf_lower, wt_pdf_upper, wt_bins = interevent_pdf(
        normalized_waiting_times, return_midbins=False, bins=wt_bins
    )
    output["wt_pdf"] = wt_pdf
    output["wt_pdf_lower"] = wt_pdf_lower
    output["wt_pdf_upper"] = wt_pdf_upper
    wt_bin_width = wt_bins[1:] - wt_bins[:-1]
    wt_log_bins = np.log10(wt_bins)
    output["wt_bins"] = 10.**((wt_log_bins[1:] + wt_log_bins[:-1]) / 2.)
    output["wt_bin_width"] = wt_bin_width
    output["wt_bin_left_egde"] = wt_bins[:-1]
    output["wt_bin_right_edge"] = wt_bins[1:]

    # compute the Akaike Information Criterion for each model
    # discard waiting times below the smallest bin used here
    # because likelihood is extremely sensitive to noise at
    # very short waiting times
    wt_min = output["wt_bin_left_egde"][wt_pdf > 0].min()
    normalized_waiting_times = normalized_waiting_times[
            normalized_waiting_times > wt_min
            ]
    # --------- gamma aic
    model = partial(
        gamma_waiting_times,
        gamma=output["gamma"],
        beta=output["beta"],
        normalized=False,
        C="theoretical",
    )
    num_params = 1 if fix_beta else 2
    output["aic_gamma"] = compute_aic(
            normalized_waiting_times, model, num_params=num_params
            )
    if np.isinf(output["aic_gamma"]):
        # very few waiting times fell in the tested range of normalized tau
        output["gamma"] = np.nan
        output["beta"] = np.nan
    # --------- fractal aic
    model = partial(
        fractal_waiting_times,
        n=output["n"],
        tau_c=output["tau_c"],
        alpha=output["alpha"],
        lbd=1.0,
        tau_min=output["tau_min"],
    )
    num_params = 3
    output["aic_fractal"] = compute_aic(
            normalized_waiting_times, model, num_params=num_params
            )
    if np.isinf(output["aic_fractal"]):
        # very few waiting times fell in the tested range of normalized tau
        output["n"] = np.nan
        output["tau_c"] = np.nan
        output["alpha"] = np.nan
    return output


def run_occupation_analysis2(
    timings,
    normalized_tau_min,
    normalized_tau_max,
    min_num_events=5,
    nbins_wt=20,
    shortest_resolved_time=5.0,
    fix_beta=False,
    loss_phi="l2_log",
    use_valid_only=1,
    base_log=2.0,
    num_resamplings=50,
):
    """
    Perform occupation analysis on event timings and inter-event times.

    Fit the fractal model to the occupation probability and the gamma model
    to the occupation probability.

    Parameters:
    -----------
    timings : numpy.ndarray
        Array of event timings.
    normalized_tau_min : float
        Minimum normalized time bin size and inter-event time.
    normalized_tau_max : float
        Maximum normalized time bin size and inter-event time.
    min_num_events : int, optional
        Minimum number of events for analysis (default is 5).
    nbins_wt : int, optional
        Number of bins for the inter-event time PDF (default is 20).
    shortest_resolved_time : float, optional
        Minimum resolved inter-event time (default is 5.0).
    use_valid_only : int, optional
        Integer taking a value in [0, 1, 2]. Defaults to 1.
        - 0: Use all time bins between `normalized_tau_min` and
          `normalized_tau_max`
        - 1: Do not use time bins below which the maximum number of earthquakes
          in any given bin is exactly 1. Below that, the occupation probability
          trivially decreases proportionally to `tau`.
        - 2: Do not use time bins below which the maximum number of earthquakes
          in any given bin is exactly 1. In addition, do not use time bins that
          are shorter than `shortest_resolved_interevent_time`.
    loss_phi : str, optional
        Loss function for occupation probability fractal model fitting (default is "l2_log").
    base_log : float, optional
        Base for logarithm used in bin log spacing (default is 2.0).
    num_resamplings : int, optional
        Number of resampling iterations (default is 50).

    Returns:
    --------
    dict
        A dictionary containing the results of the occupation analysis, including
        occupation probability, inter-event time PDF, gamma model parameters, and
        fractal model parameters.

    Notes:
    ------
    This function computes occupation probability, inter-event time PDF, fits a
    gamma and fractal models to occupation probability. It returns the results
    as a dictionary.
    """
    output = {}
    # compute occupation probability
    Phi, Phi_lower, Phi_upper, tau = compute_occupation_probability(
        timings,
        normalized_tau_min=normalized_tau_min,
        normalized_tau_max=normalized_tau_max,
        base_log=base_log,
        min_num_events=min_num_events,
        use_valid_only=use_valid_only,
        shortest_resolved_interevent_time=shortest_resolved_time,
        num_resamplings=num_resamplings,
    )
    output["Phi"] = Phi
    output["Phi_lower"] = Phi_lower
    output["Phi_upper"] = Phi_upper
    output["tau"] = tau
    print(f"Largest time interval is: {tau.max():.2e}")

    # fit gamma model to occupation probability
    _, _, _, _, gamma_model_parameters = occupation_analysis(
        timings,
        plot_above=2.0,
        model="gamma",
        normalized_tau_min=normalized_tau_min,
        normalized_tau_max=normalized_tau_max,
        base_log=base_log,
        shortest_resolved_interevent_time=shortest_resolved_time,
        use_valid_only=use_valid_only,
        fix_beta=fix_beta,
        return_figure=False,
        loss=loss_phi,
        pre_computed_Phi=output
    )
    output["gamma"] = gamma_model_parameters["gamma"]
    output["gamma_err"] = gamma_model_parameters["gamma_err"]
    output["gamma_rms"] = gamma_model_parameters["rms"]
    if "beta" in gamma_model_parameters:
        output["beta"] = gamma_model_parameters["beta"]
        output["beta_err"] = gamma_model_parameters["beta_err"]
    else:
        output["beta"] = 1.0 / gamma_model_parameters["gamma"]
        output["beta_err"] = abs(1.0 / output["gamma"] ** 2) * output["gamma_err"]

    # fit fractal model to occupation probability
    _, _, _, _, fractal_model_parameters = occupation_analysis(
        timings,
        plot_above=2.0,
        model="fractal",
        normalized_tau_min=normalized_tau_min,
        normalized_tau_max=normalized_tau_max,
        base_log=base_log,
        shortest_resolved_interevent_time=shortest_resolved_time,
        use_valid_only=use_valid_only,
        return_figure=False,
        loss=loss_phi,
        pre_computed_Phi=output
    )

    output["n"] = fractal_model_parameters["n"]
    output["n_err"] = fractal_model_parameters["n_err"]
    output["tau_c"] = fractal_model_parameters["tau_c"]
    output["tau_c_err"] = fractal_model_parameters["tau_c_err"]
    output["alpha"] = fractal_model_parameters["alpha"]
    output["alpha_err"] = fractal_model_parameters["alpha_err"]
    output["fractal_rms"] = fractal_model_parameters["rms"]
    output["tau_min"] = fractal_model_parameters["tau_min"]
    output["D_tau"] = 1.0 - output["n"]
    output["D_tau_err"] = output["n_err"]
    output["var_reduction"] = fractal_model_parameters["var_reduction"]

    # compute inter-event time pdf
    waiting_times = timings[1:] - timings[:-1]
    output["wt_mean"] = fractal_model_parameters["wt_mean"]
    output["wt_cov"] = np.std(waiting_times) / output["wt_mean"]
    normalized_waiting_times = waiting_times / output["wt_mean"]
    wt_bins = np.logspace(np.log10(tau.min()), np.log10(tau.max()), nbins_wt)
    wt_pdf, wt_pdf_lower, wt_pdf_upper, wt_bins = interevent_pdf(
        normalized_waiting_times, return_midbins=False, bins=wt_bins
    )
    output["wt_pdf"] = wt_pdf
    output["wt_pdf_lower"] = wt_pdf_lower
    output["wt_pdf_upper"] = wt_pdf_upper
    wt_bin_width = wt_bins[1:] - wt_bins[:-1]
    wt_log_bins = np.log10(wt_bins)
    output["wt_bins"] = 10.**((wt_log_bins[1:] + wt_log_bins[:-1]) / 2.)
    output["wt_bin_width"] = wt_bin_width
    output["wt_bin_left_egde"] = wt_bins[:-1]
    output["wt_bin_right_edge"] = wt_bins[1:]

    # compute the Akaike Information Criterion for each model
    # discard waiting times below the smallest bin used here
    # because likelihood is extremely sensitive to noise at
    # very short waiting times
    wt_min = output["wt_bin_left_egde"][wt_pdf > 0].min()
    normalized_waiting_times = normalized_waiting_times[
            normalized_waiting_times > wt_min
            ]
    # --------- gamma aic
    model = partial(
        gamma_waiting_times,
        gamma=output["gamma"],
        beta=output["beta"],
        normalized=False,
        C="theoretical",
    )
    num_params = 1 if fix_beta else 2
    output["aic_gamma"] = compute_aic(
            normalized_waiting_times, model, num_params=num_params
            )
    # --------- fractal aic
    model = partial(
        fractal_waiting_times,
        n=output["n"],
        tau_c=output["tau_c"],
        alpha=output["alpha"],
        lbd=1.0,
        tau_min=output["tau_min"],
    )
    num_params = 3
    output["aic_fractal"] = compute_aic(
            normalized_waiting_times, model, num_params=num_params
            )

    return output


def run_occupation_analysis3(
    timings,
    normalized_tau_min,
    normalized_tau_max,
    min_num_events=5,
    nbins_wt=20,
    shortest_resolved_time=5.0,
    use_valid_only=1,
    loss_phi="l2_log",
    base_log=2.0,
    num_resamplings=50,
):
    """
    Removed? Perform occupation analysis on event timings.

    Parameters:
    -----------
    timings : numpy.ndarray
        Array of event timings.
    normalized_tau_min : float
        Minimum normalized time bin size and inter-event time.
    normalized_tau_max : float
        Maximum normalized time bin size and inter-event time.
    min_num_events : int, optional
        Minimum number of events for analysis (default is 5).
    nbins_wt : int, optional
        Number of bins for the inter-event time PDF (default is 20).
    shortest_resolved_time : float, optional
        Minimum resolved inter-event time (default is 5.0).
    use_valid_only : int, optional
        Integer taking a value in [0, 1, 2]. Defaults to 1.
        - 0: Use all time bins between `normalized_tau_min` and
          `normalized_tau_max`
        - 1: Do not use time bins below which the maximum number of earthquakes
          in any given bin is exactly 1. Below that, the occupation probability
          trivially decreases proportionally to `tau`.
        - 2: Do not use time bins below which the maximum number of earthquakes
          in any given bin is exactly 1. In addition, do not use time bins that
          are shorter than `shortest_resolved_interevent_time`.
    loss_phi : str, optional
        Loss function for occupation probability fractal model fitting (default is "l2_log").
    base_log : float, optional
        Base for logarithm used in bin log spacing (default is 2.0).
    num_resamplings : int, optional
        Number of resampling iterations (default is 50).

    Returns:
    --------
    dict
        A dictionary containing the results of the occupation analysis, including
        occupation probability, inter-event time PDF, gamma model parameters, and
        fractal model parameters.

    Notes:
    ------
    This function computes occupation probability, inter-event time PDF, fits a
    gamma and fractal models to occupation probability. It returns the results
    as a dictionary.
    """
    output = {}
    # compute occupation probability
    Phi, Phi_lower, Phi_upper, tau = compute_occupation_probability(
        timings,
        normalized_tau_min=normalized_tau_min,
        normalized_tau_max=normalized_tau_max,
        base_log=base_log,
        min_num_events=min_num_events,
        shortest_resolved_interevent_time=shortest_resolved_time,
        use_valid_only=use_valid_only,
        num_resamplings=num_resamplings,
    )
    output["Phi"] = Phi
    output["Phi_lower"] = Phi_lower
    output["Phi_upper"] = Phi_upper
    output["tau"] = tau

    # fit gamma model to occupation probability
    # ------------- first, do not fix beta
    _, _, _, _, gamma_model_parameters = occupation_analysis(
        timings,
        plot_above=2.0,
        model="gamma",
        normalized_tau_min=normalized_tau_min,
        normalized_tau_max=normalized_tau_max,
        base_log=base_log,
        shortest_resolved_interevent_time=shortest_resolved_time,
        use_valid_only=use_valid_only,
        fix_beta=False,
        return_figure=False,
        loss=loss_phi,
        pre_computed_Phi=output
    )
    output["gamma"] = gamma_model_parameters["gamma"]
    output["gamma_err"] = gamma_model_parameters["gamma_err"]
    output["gamma_rms"] = gamma_model_parameters["rms"]
    output["beta"] = gamma_model_parameters["beta"]
    output["beta_err"] = gamma_model_parameters["beta_err"]
    # ------------ then, fix beta
    _, _, _, _, gamma_model_parameters = occupation_analysis(
        timings,
        plot_above=2.0,
        model="gamma",
        normalized_tau_min=normalized_tau_min,
        normalized_tau_max=normalized_tau_max,
        base_log=base_log,
        shortest_resolved_interevent_time=shortest_resolved_time,
        fix_beta=True,
        return_figure=False,
        loss=loss_phi,
        pre_computed_Phi=output
    )
    output["gamma_fixed_beta"] = gamma_model_parameters["gamma"]
    output["gamma_err_fixed_beta"] = gamma_model_parameters["gamma_err"]
    output["gamma_rms_fixed_beta"] = gamma_model_parameters["rms"]
    output["beta_fixed_beta"] = 1.0 / gamma_model_parameters["gamma"]
    output["beta_err_fixed_beta"] = (
        abs(1.0 / output["gamma_fixed_beta"] ** 2) * output["gamma_err_fixed_beta"]
    )

    # fit fractal model to occupation probability
    _, _, _, _, fractal_model_parameters = occupation_analysis(
        timings,
        plot_above=2.0,
        model="fractal",
        normalized_tau_min=normalized_tau_min,
        normalized_tau_max=normalized_tau_max,
        base_log=base_log,
        shortest_resolved_interevent_time=shortest_resolved_time,
        use_valid_only=use_valid_only,
        return_figure=False,
        loss=loss_phi,
    )

    output["n"] = fractal_model_parameters["n"]
    output["n_err"] = fractal_model_parameters["n_err"]
    output["tau_c"] = fractal_model_parameters["tau_c"]
    output["tau_c_err"] = fractal_model_parameters["tau_c_err"]
    output["alpha"] = fractal_model_parameters["alpha"]
    output["alpha_err"] = fractal_model_parameters["alpha_err"]
    output["fractal_rms"] = fractal_model_parameters["rms"]
    output["tau_min"] = fractal_model_parameters["tau_min"]
    output["D_tau"] = 1.0 - output["n"]
    output["D_tau_err"] = output["n_err"]
    output["var_reduction"] = fractal_model_parameters["var_reduction"]

    # compute inter-event time pdf
    waiting_times = timings[1:] - timings[:-1]
    output["wt_mean"] = fractal_model_parameters["wt_mean"]
    output["wt_cov"] = np.std(waiting_times) / output["wt_mean"]
    normalized_waiting_times = waiting_times / output["wt_mean"]
    wt_bins = np.logspace(np.log10(tau.min()), np.log10(tau.max()), nbins_wt)
    wt_pdf, wt_pdf_lower, wt_pdf_upper, wt_bins = interevent_pdf(
        normalized_waiting_times, return_midbins=False, bins=wt_bins
    )
    output["wt_pdf"] = wt_pdf
    output["wt_pdf_lower"] = wt_pdf_lower
    output["wt_pdf_upper"] = wt_pdf_upper
    wt_bin_width = wt_bins[1:] - wt_bins[:-1]
    wt_log_bins = np.log10(wt_bins)
    output["wt_bins"] = 10.**((wt_log_bins[1:] + wt_log_bins[:-1]) / 2.)
    output["wt_bin_width"] = wt_bin_width
    output["wt_bin_left_egde"] = wt_bins[:-1]
    output["wt_bin_right_edge"] = wt_bins[1:]

    # compute the Akaike Information Criterion for each model
    # discard waiting times below the smallest bin used here
    # because likelihood is extremely sensitive to noise at
    # very short waiting times
    wt_min = output["wt_bin_left_egde"][wt_pdf > 0].min()
    normalized_waiting_times = normalized_waiting_times[
            normalized_waiting_times > wt_min
            ]

    # --------- gamma aic
    model = partial(
        gamma_waiting_times,
        gamma=output["gamma"],
        beta=output["beta"],
        normalized=False,
        C="theoretical",
    )
    num_params = 2
    output["aic_gamma"] = compute_aic(
            normalized_waiting_times, model, num_params=num_params
            )
    # --------- gamma with fixed beta aic
    model = partial(
        gamma_waiting_times,
        gamma=output["gamma_fixed_beta"],
        beta=output["beta_fixed_beta"],
        normalized=True,
        C="theoretical",
    )
    num_params = 1
    output["aic_gamma_fixed_beta"] = compute_aic(
            normalized_waiting_times, model, num_params=num_params
            )
    # --------- fractal aic
    model = partial(
        fractal_waiting_times,
        n=output["n"],
        tau_c=output["tau_c"],
        alpha=output["alpha"],
        lbd=1.0,
        tau_min=output["tau_min"],
    )
    num_params = 3
    output["aic_fractal"] = compute_aic(
            normalized_waiting_times, model, num_params=num_params
            )

    return output


def plot_gamma_vs_fractal(
    occupation_parameters,
    num_points_fit=50,
    ot_time_series=None,
    figname="occurrence_statistics",
    figtitle="",
    plot_tau_min=False,
    plot_physical_wt=False,
    plot_poisson=True,
    axes=None,
    **kwargs,
):
    """ """
    import string

    # unpack arguments
    wt_pdf = occupation_parameters["wt_pdf"]
    wt_pdf_lower = occupation_parameters["wt_pdf_lower"]
    wt_pdf_upper = occupation_parameters["wt_pdf_upper"]
    wt_bins = occupation_parameters["wt_bins"]
    tau = occupation_parameters["tau"]
    Phi = occupation_parameters["Phi"]
    Phi_lower = occupation_parameters["Phi_lower"]
    Phi_upper = occupation_parameters["Phi_upper"]

    valid_bins = wt_pdf > 0.0

    wt_fit = np.logspace(
        np.log10(wt_bins[valid_bins].min()),
        np.log10(wt_bins[valid_bins].max()),
        num_points_fit,
    )
    tau_fit = np.logspace(np.log10(tau.min()), np.log10(tau.max()), num_points_fit)

    # correction factor for the gamma distribution, taking into account that
    # the empirical distribution is normalized over a finite range of waiting times
    trunc_norm = truncated_norm_gamma(
        occupation_parameters["gamma"],
        occupation_parameters["beta"],
        wt_bins.min(),
        wt_bins.max(),
    )
    theo_norm = theoretical_norm_gamma(
        occupation_parameters["gamma"], occupation_parameters["beta"]
    )
    correction = theo_norm / trunc_norm

    if ot_time_series is not None:
        figsize = (16, 13)
    else:
        figsize = (16, 7)

    if axes is None:
        fig = plt.figure(figname, figsize=figsize)
        if ot_time_series is not None:
            grid = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(grid[0, 0])
            ax2 = fig.add_subplot(grid[0, 1])
            ax3 = fig.add_subplot(grid[1, :])
            axes = [ax1, ax2, ax3]
        else:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            axes = [ax1, ax2]
    else:
        fig = axes[0].get_figure()
    suptitle_y = 0.99 if plot_physical_wt else 0.98
    fig.suptitle(figtitle, y=suptitle_y)

    axes[1].set_title(r"Inter-event time pdf, $\rho_{\Theta}$")
    axes[1].plot(
        wt_bins[valid_bins],
        wt_pdf[valid_bins],
        marker="o",
        ls="",
        color="C0",
        label="Observed distribution",
    )
    axes[1].fill_between(
        wt_bins[valid_bins],
        wt_pdf_lower[valid_bins],
        wt_pdf_upper[valid_bins],
        alpha=0.25,
        color="C0",
    )
    axes[1].set_ylabel(r"Inter-event time pdf, $\rho_{\Theta}$")
    axes[1].set_xlabel(r"Normalized inter-event time, $\theta$")

    axes[0].set_title(r"Occupation probability, $\Phi(\theta)$")
    axes[0].plot(tau, Phi, marker="o", ls="", color="C0", label="Observed occupation")
    axes[0].set_ylabel(r"Occupation probability, $\Phi(\theta)$")
    axes[0].set_xlabel(r"Normalized time interval length, $\theta$")

    axes[0].fill_between(tau, Phi_lower, Phi_upper, alpha=0.25, color="C0")

    wt_gamma_model = gamma_waiting_times(
        wt_fit, occupation_parameters["gamma"],
        beta=occupation_parameters["beta"],
        normalized=False,
    )
    occupation_gamma_model = occupation_probability_gamma_model(
        tau_fit, occupation_parameters["gamma"], beta=occupation_parameters["beta"]
    )
    label = (
        f"Gamma model:\n"
        r"$\gamma=$"
        f"{occupation_parameters['gamma']:.2f}"
        r", $\beta=$"
        f"{occupation_parameters['beta']:.2f}"
    )
    axes[0].plot(tau_fit, occupation_gamma_model, ls="-.", color="black", label=label)
    axes[1].plot(wt_fit, wt_gamma_model / correction, ls="-.", color="black")

    # correction factor for the fractal distribution, taking into account that
    # the empirical distribution is normalized over a finite range of waiting times
    trunc_norm = truncated_norm_fractal(
        occupation_parameters["n"],
        occupation_parameters["tau_c"],
        occupation_parameters["alpha"],
        wt_bins.min(),
        wt_bins.max(),
    )
    theo_norm = theoretical_norm_fractal(
        occupation_parameters["n"],
    )
    # estimate the error made when estimating the rate of seismicity
    # with the sample mean
    hat_rate_vs_real_rate = estimate_sample_rate_vs_real_rate(
        occupation_parameters["wt_bins"],
        occupation_parameters["wt_pdf"],
        occupation_parameters["n"],
        occupation_parameters["tau_c"],
        occupation_parameters["alpha"],
    )
    # use it to refine theo_norm
    #theo_norm *= hat_rate_vs_real_rate
    occupation_parameters["hat_rate_vs_real_rate"] = hat_rate_vs_real_rate

    correction = theo_norm / trunc_norm

    wt_fractal_model = fractal_waiting_times(
        wt_fit,
        occupation_parameters["n"],
        occupation_parameters["tau_c"],
        occupation_parameters["alpha"],
    )
    occupation_fractal_model = occupation_probability_fractal_model(
        tau_fit,
        occupation_parameters["n"],
        occupation_parameters["tau_c"],
        occupation_parameters["alpha"],
    )
    label = (
        f"Fractal model:\n"
        r"$D_\tau=$"
        f"{1. - occupation_parameters['n']:.2f}, "
        r"$\theta_c=$"
        f"{occupation_parameters['tau_c']:.2f}, "
        r"$\alpha=$"
        f"{occupation_parameters['alpha']:.2f}"
    )
    # an offset (in log scale) is visible between the data and the model when
    # the rate of seismicity is not well approximated by the inverse of the
    # waiting time sample mean
    axes[0].plot(
        tau_fit, occupation_fractal_model, ls="--", color="magenta", label=label
    )
    axes[1].plot(wt_fit, wt_fractal_model / correction, ls="--", color="magenta")
    if plot_tau_min:
        axes[1].axvline(occupation_parameters["tau_min"], color="magenta", lw=1.0)

    if plot_poisson:
        axes[0].plot(
            tau,
            1.0 - np.exp(-tau),
            color="dimgrey",
            lw=0.75,
            label=r"Poisson: $1 - e^{-\theta}$",
        )
        axes[1].plot(
            wt_bins,
            np.exp(-wt_bins),
            color="dimgrey",
            lw=0.75,
            label=r"Poisson: $e^{-\theta}$",
        )

    axes[0].legend(loc="lower right", handlelength=0.9)
    axes[1].legend(loc="lower left", handlelength=0.9)

    axes[1].set_ylim(0.9 * wt_pdf[valid_bins].min(), axes[1].get_ylim()[1])

    for i, ax in enumerate(axes):
        ax.grid()
        ax.text(
            0.01,
            0.99,
            f"({string.ascii_lowercase[i]})",
            va="top",
            ha="left",
            fontsize=20,
            transform=ax.transAxes,
        )

    axes[1].loglog()
    if kwargs.get("x_scale_log", True):
        axes[0].set_xscale("log")
    if kwargs.get("y_scale_log", True):
        axes[0].set_yscale("log")

    if "wt_mean" in occupation_parameters and plot_physical_wt:
        wt_mean = occupation_parameters["wt_mean"]
        ax_real_time = axes[1].twiny()
        ax_real_time.set_xlim(
                axes[1].get_xlim()[0] * wt_mean, axes[1].get_xlim()[1] * wt_mean
                )
        ax_real_time.set_xscale("log")
        ax_real_time.tick_params(axis="x", which="both", direction="in", pad=0.5)
        ax_real_time.set_xlabel(r"Inter-event time, $w$", labelpad=-35)

    if ot_time_series is not None:
        ot_time_series = np.sort(np.asarray(ot_time_series))
        interevent_time = ot_time_series[1:] - ot_time_series[:-1]
        axes[2].scatter(
                ot_time_series[1:],
                interevent_time,
                marker=".",
                color="k",
                rasterized=True,
        )
        axes[2].set_yscale("log")
        axes[2].set_ylabel("Inter-event time (s)")
        axes[2].set_xlabel("Origin time (s)")

    return fig
