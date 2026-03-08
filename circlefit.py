# -*- coding: utf-8 -*-
"""Fit S21 data

Fit S21 data using method provide in [1]:https://arxiv.org/abs/1410.3365.
"""

import numpy as np
import scipy.optimize

__all__ = [
    'fit_circle',
    'extract_resonator_parameters',
    'auto_fit',
    's21_func'
]


def fit_circle(complex_datapoints: np.ndarray) -> tuple:
    """Fit a circle to complex data points on compelx plane
    
    Return `xc, yc, r0`.

    See page 3 of [1]:
        1. Find moment matrix M.
        2. Solve characteristic equation det(M - ηB) = 0 for η*.
        3. Solve eigenvector A* corresponding to η*.

    Parameters
    ----------
    s21_complex : ndarray
        The complex S21 data to be fit.

    Returns
    -------
    xc : float
        x-coordinate of the fitted circle's center.
    yc : float
        y-coordinate of the fitted circle's center.
    r0 : float
        Radius of the fitted circle.

    References
    ----------
    [1] Probst et al., "Efficient and robust analysis of resonator data in the linear regime," 
        Rev. Sci. Instrum. 86, 024706 (2015). https://arxiv.org/abs/1410.3365
    """
    # 1.claculate moment matrix
    xi = complex_datapoints.real
    yi = complex_datapoints.imag
    zi = np.power(xi, 2) + np.power(yi, 2)
    n = len(xi)
    moment_matrix = np.array([
        [(zi*zi).sum(), (xi*zi).sum(), (yi*zi).sum(), zi.sum()],
        [(xi*zi).sum(), (xi*xi).sum(), (xi*yi).sum(), xi.sum()],
        [(yi*zi).sum(), (xi*yi).sum(), (yi*yi).sum(), yi.sum()],
        [     zi.sum(),      xi.sum(),      yi.sum(),        n],
    ])
    M = moment_matrix
    
    # 2.solve characteristic equation
    constrain_matirx = np.array([
        [ 0,  0,  0, -2],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0],
        [-2,  0,  0,  0],
    ])
    B = constrain_matirx
    def characteristic_eqn(eta):
        return np.linalg.det(M - eta*B)
    initial_guess = 0.0 # as advised in page3 of [1]
    eta_star = scipy.optimize.fsolve(characteristic_eqn, initial_guess)
    
    # 3.solve eigenvector then obtain result
    _, s, vh = np.linalg.svd(M - eta_star*np.eye(4))
    vec_A_star = vh[np.argmin(s), :]
    a, b, c, d = vec_A_star[0], vec_A_star[1], vec_A_star[2], vec_A_star[3]
    xc = -b / (2*a)
    yc = -c / (2*a)
    r0 =  1 / (2*np.abs(a)) * np.sqrt(b**2 + c**2 - 4*a*d)
    if np.isnan(xc) or np.isnan(yc) or np.isnan(r0):
        raise RuntimeError("fit_circle: NaN encountered in fitted parameters.") from None
    else:
        return xc, yc, r0

def extract_resonator_parameters(f, s21_complex, xc, yc, r0, n_port=1):
    """Extract resonator parameters from circle fit result.
    
    Return `fr, Ql, Qc_abs, Qi, theta`.

    Parameters
    ----------
    f : ndarray
        The frequecny.
    s21_complex : ndarray
        The complex S21 data.
    xc : float
        x-coordinate of the circle fitted circle's center.
    yc : float
        y-coordinate of the circle fitted circle's center.
    r0 : float
        Radius of the circle fitted circle.
    n_port:
        1 for reflection type, 2 for notch type.

    Returns
    -------
    fr : float
        Resonance frequecny of resonator.
    Ql : float
        Loaded quality factor, 1/Ql = 1/Qi + 1/|Qc|
    Qc_abs : float
        Coupling quality factor, its phase is impedance mismatch phase.
    Qi : float
        Intrinsic quality factor.
    theta : float
        impedance mismatch phase.
    
    References
    ----------
    [1] Probst et al., "Efficient and robust analysis of resonator data in the linear regime," 
        Rev. Sci. Instrum. 86, 024706 (2015). https://arxiv.org/abs/1410.3365
    """

    # 1. Impedance mismatch phase
    x_safe = np.clip(yc/r0, -1, 1) # make it within -1~1 to aviod error
    theta = -np.arcsin(x_safe)

    # 2. Translate circle to origin
    s21_trans = s21_complex - (xc + 1j * yc)

    # 3. Unwrap phase and fit Eq. (12)
    phi = np.unwrap(np.angle(s21_trans))
    def phase_model(f, fr, Ql, theta0):
        return theta0 + 2 * np.arctan(2 * Ql * (1 - f / fr))

    p0 = [f[np.argmax(np.abs(np.gradient(phi)))], 1e4, phi[0]]  # rough initial guess
    popt, _ = scipy.optimize.curve_fit(phase_model, f, phi, p0=p0)
    fr, Ql, theta0 = popt

    # 4. Compute complex Qc and Qi
    Qc = Ql / (n_port * r0 * np.exp(-1j * theta))
    Qi = 1 / (1 / Ql - np.real(1 / Qc))

    return fr, Ql, np.abs(Qc), Qi, theta

def auto_fit(f, s21_complex, n_port=1, f0='auto', span0='auto', maximum_trail=20, show_process=True):
    """Perform circle fit and extract resonator parameters.
    
    Return `xc, yc, r0, fr, Ql, Qc_abs, Qi, theta`.
    
    1. If f0 is auto, estimate f0 by steepest phase change point.
    2. If span0 is auto, use 30MHz to start the trial.
    3. Automatically try to modify the window_span within 3~6*linewidth.
    
    Returns
    -------
    xc : float
        x-coordinate of the fitted circle's center.
    yc : float
        y-coordinate of the fitted circle's center.
    r0 : float
        Radius of the fitted circle.
    fr : float
        Resonance frequecny of resonator.
    Ql : float
        Loaded quality factor, 1/Ql = 1/Qi + 1/|Qc|
    Qc_abs : float
        Coupling quality factor, its phase is impedance mismatch phase.
    Qi : float
        Intrinsic quality factor.
    theta : float
        impedance mismatch phase.
    """
    def f_s21_truncate(center, span):
        # choose window
        f_min = center - span / 2.0
        f_max = center + span / 2.0
        window_mask = (f >= f_min) & (f <= f_max)
        f_used = f[window_mask]
        s21_used = s21_complex[window_mask]
        return f_used, s21_used
    
    ## for auto, estimate f0 by phase changes
    if type(f0) == str and f0 == 'auto':
        phase = np.angle(s21_complex)
        unwrapped_phase = np.unwrap(phase)
        phase_gradient = np.gradient(unwrapped_phase)
        f0_index = np.argmax(np.abs(phase_gradient))
        f0 = f[f0_index]
    
    ## for auto, use 30MHz span to start with
    if type(span0) == str and span0 == 'auto':
        span0 = 30e6 # 30MHz to start with, generally good for us
    
    num_trial = 0
    fit_flag = False
    while num_trial < maximum_trail:
        ## trucate a portion of data
        f_used, s21_used = f_s21_truncate(center=f0, span=span0)
    
        ## fit circle
        xc, yc, r0 = fit_circle(s21_used)
        
        ## extract result
        fr, Ql, Qc_abs, Qi, theta = extract_resonator_parameters(f_used, s21_used, xc, yc, r0, n_port)
        linewidth = fr / Ql
        if show_process:
            print(
                f'trial {num_trial}, ' 
                f'f0: {f0/1e9:6.3f} GHz, '
                f'span0: {span0/1e6:7.3f} MHz, '
                f'fr: {fr/1e9:6.3f} GHz, '
                f'LW: {linewidth/1e6:7.3f} MHz.'
            )
        
        ## make sure the span is within 3 ~ 6 linewidth
        if linewidth < 0:
            # un-physical, but appear when data is not resonator.
            span0 *= 1.5 # fast grow, eventually full span of data
            # f0 = fr # DONT MAKE mew f0=fr if non physical
            num_trial += 1
            continue
        elif span0 < 3*linewidth:
            # span too narrow, widen and retry
            span0 *= 1.25
            f0 = fr
            num_trial += 1
            continue
        elif span0 > 6*linewidth:
            # span too wild, narrowing and retry
            span0 /= 1.25
            f0 = fr
            num_trial += 1
            continue
        elif num_trial == 0: 
            # span is witin 3 ~ 6 of linewidth, but try least two times, use fitted fr as new f0
            f0 = fr
            num_trial += 1
            continue
        else:
            # span is now within 3 ~ 6 of linewidth
            fit_flag = True
            break
    
    if not fit_flag: 
        print('warning: Trys to the maxima number of trial set, please manually fit it for more control.')        
    return xc, yc, r0, fr, Ql, Qc_abs, Qi, theta

def s21_func(f, f0, Ql, Qc_abs, theta=0.0, a=1.0, phi0=0.0, delay=0.0, amp_offset=0.0):
    """Complex transmission S21(f) of a resonator in notch configuration.

    Args:
        f (float): Frequency array [Hz]
        f0 (float): Resonance frequency [Hz]
        Ql (float): Loaded quality factor, 1/Ql = 1/Qi + 1/|Qc|
        Qc_abs (float): Coupling quality factor.
        theta (float): Coupling phase, impedance mismatch phase, phase of Qc.
        a (float): Overall amplitude scaling
        phi0 (float): Global rotation (radians)
        delay (float): Cable delay (seconds)
        amp_offset (complex): Complex background offset

    Returns:
        s21 (ndarray): The complex-valued S21(f)
    """
    x = (f - f0) / f0
    s21_res = 1.0 - (Ql / Qc_abs * np.exp(1j * theta)) / (1.0 + 2j * Ql * x)
    s21_total = a * np.exp(1j * phi0) * s21_res * np.exp(-2j * np.pi * delay * f) + amp_offset
    return s21_total

