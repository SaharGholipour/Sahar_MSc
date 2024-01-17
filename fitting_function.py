# packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import scipy.stats as stats
from decimal import Decimal
from matplotlib.offsetbox import AnchoredText
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import Chebyshev, chebval
from scipy.special import eval_jacobi

form_title = {'family': 'helvetica', 'color': 'black', 'size': 15}
form_label = {'family': 'helvetica', 'color': 'black', 'size': 20}

# ***********************************************best candidate picking*****************************************************
def bestcan(data, reset_index=True):
    if reset_index:
        data = data.reset_index()
        
    mincan_mask = data['__candidate__'] == data.groupby('__event__')['__candidate__'].transform('min')
    mincan_data = data[mincan_mask].copy()

    # Remove mincan_data from the original dataset
    nomincan_data = data.drop(mincan_data.index, axis=0)

    return mincan_data, nomincan_data

# ********************************************************Split signal from background***********************************
# Note: for this function you need to reconstruct what you want and then pass it to "bestcan" function then that would be the data to be passed to this function.
# argument "gen" can be 'charged' or 'mixed'
# argument "first_B_rec, second_B_rec" can be 'charged' or 'mixed'
def sig_bkg_spliter(dataset, first_B_rec, second_B_rec, gen, sigpkg_output=False, full_output=False):
    
    # charged Detalnu reconstructed modes
    l_eta_Bp = [1600,1700,2600,2700]
    eta_rec_Bp = list(i+1 for i in l_eta_Bp) + l_eta_Bp
    # mixed Detalnu reconstructed modes
    l_eta_B0 = [1400,1500,2400,2500]
    eta_rec_B0 = list(i+1 for i in l_eta_B0) + l_eta_B0
    
    # generated modes
    # charged Detalnu and DDs generated modes
    eta_gen_Bp = [1019,1020,1039,1040]
    DDs_gen_Bp = [i+1681 for i in range(4)]
    # mixed Detalnu and DDs generated modes
    eta_gen_B0 = [1017,1018,1035,1036]
    DDs_gen_B0 = [i+1833 for i in range(4)]
    
    rec_mode_mapping = {'charged': eta_rec_Bp, 'mixed': eta_rec_B0}
    first_B_rec_mode = rec_mode_mapping.get(first_B_rec, [])
    second_B_rec_mode = rec_mode_mapping.get(second_B_rec, [])
    
    if gen == 'charged':
        particle_variable = 'aBplusMode'
        antiparticle_variable = 'aBminusMode'
        eta_gen_mode = eta_gen_Bp
        DDs_gen_mode = DDs_gen_Bp
        
    if gen == 'mixed':
        particle_variable = 'aB0Mode'
        antiparticle_variable = 'aBbar0Mode'
        eta_gen_mode = eta_gen_B0
        DDs_gen_mode = DDs_gen_B0
    
    # signal
    eta_df_sig = dataset[
    ( 
     ((abs(dataset[antiparticle_variable])%10000).isin(eta_gen_mode)) & ((dataset['B0_decayModeID']).isin(first_B_rec_mode)) & (dataset['pi4_B0_isSignal']==1)
    )
    
    |
    
    (
     (((dataset[particle_variable])%10000).isin(eta_gen_mode)) & ((dataset['B1_decayModeID']).isin(second_B_rec_mode)) & (dataset['pi4_B1_isSignal']==1)
    )
    ]
    # background
    eta_df_bkg = dataset.drop(eta_df_sig.index, inplace=False)
    # peaking background
    eta_df_pkg = eta_df_bkg[
        ( 
        ((abs(eta_df_bkg[antiparticle_variable])%10000).isin(DDs_gen_mode)) & ((eta_df_bkg['B0_decayModeID']).isin(first_B_rec_mode)) & (eta_df_bkg['pi4_B0_isSignal']==1)
        )
        
        |
        
        ( 
        ((abs(eta_df_bkg[particle_variable])%10000).isin(DDs_gen_mode)) & ((eta_df_bkg['B1_decayModeID']).isin(second_B_rec_mode)) & (eta_df_bkg['pi4_B1_isSignal']==1)
        )
    ]
    # combinatorial background
    eta_df_ckg = eta_df_bkg.drop(eta_df_pkg.index)

    # signal + peaking background as my signal
    eta_df_sigpkg = pd.concat([eta_df_sig, eta_df_pkg])

    if full_output:
        return eta_df_sigpkg, eta_df_ckg, eta_df_sig, eta_df_bkg, eta_df_pkg
    elif sigpkg_output:
        return eta_df_sigpkg
    else:
        return eta_df_sigpkg, eta_df_ckg
    
# *********************************************Efficiency Checking********************************
# the data_sets with cuts should be in a dictionary and for the keys you can put the cuts that you defined for your data_set
def efficiency_check(cuts_dict, mcBF, total_NBBbar, first_B_rec, second_B_rec, gen):
    
    efficiency, ratio, signal, background, figure_merit, effi_check = ({} for i in range(6))
    
    for key, data_set in cuts_dict.items():
        sigpkg, ckg, sig, bkg, pkg= sig_bkg_spliter(data_set, first_B_rec, second_B_rec, gen, full_output=True)
        efficiency[key] = len(sig) / (2 * mcBF * total_NBBbar)
        figure_merit[key] = len(sig) / np.sqrt(len(sig) + len(bkg))
        signal[key] = len(sig)
        background[key] = len(bkg)

    prev_key = None
    prev_value = None
    for key, value in efficiency.items():
        if prev_key is not None:
            ratio[f"{prev_key} to {key}"] = value / prev_value

        prev_key = key
        prev_value = value
    
    effi_check['efficiency'] = efficiency
    effi_check['ratio'] = ratio
    effi_check['figure_merit'] = figure_merit
    effi_check['number of signal'] = signal
    effi_check['number of background'] = background
    return effi_check

# "cuts_list is a list of all data_sets with different sets of cuts
def efficiency_check_firt_trial(cuts_list, total_NBBbar, first_B_rec, second_B_rec, gen):
    # B_sig
    mcBF_Detalnu = 2 * 0.002010000 # two is because of e and mu
    mcBF_Dstaretalnu = 2 * 0.002010000 # two is because of e and mu
    mcBF_Bsig_BBcahrged = (mcBF_Detalnu + mcBF_Dstaretalnu)
    efficiency = {}
    for index, data_set in enumerate(cuts_list):
        sigpkg, ckg, sig, bkg, pkg= sig_bkg_spliter(data_set, first_B_rec, second_B_rec, gen, full_output=True)
        efficiency [f'cut{index}'] = len(sig) / (2 * mcBF_Bsig_BBcahrged * total_NBBbar)
        
    return efficiency
# ************************************************** Normalizing the histogram ************************************
# if your bin_width is all teh same so you hust label your plot NOE/10MEV and don't normalize it to bin_width
def compute_weights(data, num_bins, desired_norm):
    _, bin_edges = np.histogram(data, bins=num_bins)
    # bin_width = bin_edges[1] - bin_edges[0]
    # weights = np.ones_like(data) / len(data) * desired_norm / bin_width
    weights = np.ones_like(data) / len(data) * desired_norm
    return weights

def calculate_weights(data, total_events, num_bins, desired_norm):
    hist_height, _ = np.histogram(data, bins=num_bins)
    current_sum = hist_height.sum()
    scaling_factor = desired_norm / total_events
    return scaling_factor
# example usage for barstacked plots
# data = np.concatenate([true_charged, true_mixed])
# scaling_factor = calculate_weights(data, num_bins, desired_norm)
# weights_charged = np.full_like(true_charged, scaling_factor)
# weights_mixed = np.full_like(true_mixed, scaling_factor)

# *************************************************plot signal and background***********************************************
# plotting hist of signal and background with the bin width of 10 (MeV) or 0.01 (GeV)
# also this funcion can add either a flat background or a normalized poisson distribution to your signal to avoid the zero bin counts
# argument "signal_label" you can put sig+pkg or sig for example
def plot_sigbkg_InvM_hist(
    df_background, df_signal, signal_label, *args, df_pkg=None, scale=None, y_axis_sapce=None, full_output=False, flat_bkg=False, poisson_noise=False):
    
    # Invariant mass of a pair of photons
    particle_InvM_bkg = pd.concat([df_background['pi4_B0_InvM'].dropna(), df_background['pi4_B1_InvM'].dropna()])
    particle_InvM_sig = pd.concat([df_signal['pi4_B0_InvM'].dropna(), df_signal['pi4_B1_InvM'].dropna()])

    # I add a poisson distribution as a noise to my signal bin_counts in order not to get zero signal bin_counts
    # bc then the uncertainty for that bin will be zero and for chi_squared we would have problem
    # Note: the noise will add approximately one event in total to signal
    bins = np.arange(np.min(particle_InvM_bkg), np.max(particle_InvM_bkg), 0.01)
    
    poisson_values = np.random.poisson(100, len(bins)-1)
    normalized_poisson = poisson_values/np.sum(poisson_values)
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    # weights and plotting
    if df_pkg is not None:
        particle_InvM_pkg = pd.concat([df_pkg['pi4_B0_InvM'].dropna(), df_pkg['pi4_B1_InvM'].dropna()])
        weights_bkg, weights_pkg, weights_sig = scale
        weights = [np.full_like(particle_InvM_bkg, weights_bkg), np.full_like(particle_InvM_pkg, weights_pkg), np.full_like(particle_InvM_sig, weights_sig)]
        bin_count, bin_edge, _ = ax.hist(
        [particle_InvM_bkg, particle_InvM_pkg, particle_InvM_sig], label=['background', 'peaking background', signal_label], histtype='barstacked', bins=bins, weights=weights, color=['C0', 'limegreen', 'darkorange'])
        
        bkg=bin_count[0]
        pkg=bin_count[1] - bkg
        sig=bin_count[2] - bin_count[1]
        
    else:
        weights_bkg, weights_sig = scale
        weights = [np.full_like(particle_InvM_bkg, weights_bkg), np.full_like(particle_InvM_sig, weights_sig)]
        bin_count, bin_edge, _ = ax.hist(
        [particle_InvM_bkg, particle_InvM_sig], label=['background', signal_label], histtype='barstacked', bins=bins, weights=weights)
        
        bkg=bin_count[0]
        sig=bin_count[1] - bkg
    
    # Set x-axis ticks to show bin edges with one decimal place
    x_axis = np.arange(np.min(bin_edge), np.max(bin_edge), 0.03)
    formatted_ticks2 = ['%.3f'%(tick) for tick in bin_edge]
    formatted_ticks = ['{:.3f}'.format(tick) for tick in x_axis]
    ax.set_xticks(x_axis, formatted_ticks, fontsize=14)
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.set_xlabel(r'$M_{\eta \rightarrow \gamma \gamma}$ (GeV)', fontdict=form_label)
    ax.set_ylabel(r'$NOE/bin_{width}$', fontdict=form_label, rotation=90, ha='center', va='center', labelpad=20)
    # ax.yaxis.set_label_coords(-0.05, 1)
    # ax.set_xlim(eta_edge_sig_BmBp[0],eta_edge_sig_BmBp[-1])
    ax.set_ylim( 0, y_axis_sapce)

    # printing the bin width an dthe number of bins on plot
    bin_width = bin_edge[1]-bin_edge[0]
    bin_numbers = len(bin_count[0])
    denominator_text = r'${bin_{width}\ (GeV)}: $' + f'{bin_width:.2f}' + r', ${bin_{numbers}}: $' + f'{bin_numbers}'
    at = AnchoredText(f'{denominator_text}', loc='upper left', frameon=True, pad=0.4, prop=dict(size=16))
    at2 = AnchoredText(fr'$bin_{{width}}$: {bin_width:.2f}', loc='upper left', frameon=False, pad=0.2)
    ax.add_artist(at)

    # showing the bin counts on the hist for signal and background seperately
    # va: vertical and ha: horizontal
    # background
    # for edge, count in zip(bin_edge, bin_count[0]):
    #     x_coord = edge + (0.5 * (bin_edge[1] - bin_edge[0]))
    #     y_coord = count/2
    #     ax.text(x_coord, y_coord, str(int(count)), ha='center', va='center', fontsize=12)
    # # signal
    # for edge, count, total_count in zip(bin_edge, bin_count[1]-bin_count[0], bin_count[1]):
    #     x_coord = edge + (0.5 * (bin_edge[1] - bin_edge[0]))
    #     y_coord = total_count
    #     ax.text(x_coord, y_coord, str(int(count)), ha='center', va='bottom', fontsize=12)

    ax.legend(fontsize=18)
    
    # x range for fitting
    x_fitting = bin_edge
    # y range for background fitting
    y_bkg_fitting = bkg
    # y range for signal fitting
    # a uniform function 1 as a flat background below signal
    if flat_bkg:
        y_sig_fitting = sig + np.ones(len(bins)-1)
    # an array of random poisson distribution numbers to each signal bin
    elif poisson_noise:
        y_sig_fitting = sig + normalized_poisson
    else:
        y_sig_fitting = sig
    
    # checking if there's any zero in the signal bin_counts after adding the noise (either poisson or flat background)
    if np.any(y_sig_fitting==0):
        print('We still have the problem of 0 signal bin_counts')
    else:
        print('No zero in signal+noise bin_counts list')
        
    if full_output:
        return x_fitting, y_sig_fitting, y_bkg_fitting, bin_count, particle_InvM_sig, particle_InvM_bkg
    else:
        return x_fitting, y_sig_fitting, y_bkg_fitting, bin_count

# ****************************************crystal_ball and legendre polynbomial functions*********************************
# both functions are normalized and mapped to [-1, 1], and they can measure the middle points for your bins bc the data you give to these functions are the bin_edges
# signal
# Normalized crystal_ball function with a constant in case we wanna add a flat background to our signal
def norm_crystal_ball_const(x, mu, sigma, a, n, c, x_min=0, x_max=0, mapping=False, calculate_mp=False):
    x_mapped = x
    if mapping:
        x_mapped = 2 * (x - x_min) / (x_max - x_min) - 1
    if calculate_mp:
        x_mapped = (x_mapped[1:] + x_mapped[:-1]) / 2
        
    aa = abs(a)
    A = (n / aa) ** n * np.exp(- aa ** 2 / 2)
    B = n / aa - aa
    C = n * np.exp(- aa ** 2 / 2) / (aa * (n - 1))
    erf = (2/np.sqrt(np.pi)) * integrate.quad(lambda t: np.exp(- t**2), 0, aa/np.sqrt(2))[0]
    D = np.sqrt(np.pi / 2) * (1 + erf)
    # normalization factor
    normalization_factor = ((sigma * (C + D)) + 2*c) * (x_max - x_min)/2
    
    condition1 = lambda x: (np.exp(- ((x-mu)**2)/(2.*sigma**2)) + c) / normalization_factor
    condition2 = lambda x: (A * (B - (x-mu)/sigma)**(-n) + c) / normalization_factor
    result = np.where((x_mapped - mu)/sigma > (-a), condition1(x_mapped), condition2(x_mapped))
    
    return result

# background
def norm_legend(x, coef1, coef2, coef3, coef0=1, x_min=0, x_max=0, mapping=False, calculate_mp=False):
    # Map x from [a, b] to [-1, 1]
    x_mapped = x
    if mapping:    
        x_mapped = 2 * (x - x_min) / (x_max - x_min) - 1
    if calculate_mp:
        x_mapped = (x_mapped[1:] + x_mapped[:-1]) / 2
    
    # Calculate the polynomial value
    polynomial_value = legval(x_mapped, [coef0, coef1, coef2, coef3])
    
    # Calculate normalization factor based on interval [a, b]
    normalization_factor = 2*coef0*(x_max-x_min)/2
    
    # Return normalized polynomial value
    return polynomial_value / normalization_factor, x_mapped

def norm_chebyshev(x, coef1, coef2, coef3, coef0=1, x_min=0, x_max=0, mapping=False, calculate_mp=False):
    # Map x from [x_min, x_max] to [-1, 1]
    x_mapped = x
    if mapping:    
        x_mapped = 2 * (x - x_min) / (x_max - x_min) - 1
    if calculate_mp:
        x_mapped = (x_mapped[1:] + x_mapped[:-1]) / 2
    
    # Calculate the polynomial value
    polynomial_value = chebval(x_mapped, [coef0, coef1, coef2, coef3])
    
    # Calculate normalization factor based on interval [x_min, x_max]
    if x_min != x_max:
        normalization_factor = (2*coef0-2/3*coef2)*(x_max-x_min)/2
    else:
        normalization_factor = 2*coef0-2/3*coef2
    
    # Return normalized polynomial value
    return polynomial_value / normalization_factor, x_mapped


def norm_jacobi(x, n, alpha, beta, x_min=0, x_max=0, mapping=False, calculate_mp=False):
    # Map x from [a, b] to [-1, 1]
    x_mapped = x
    if mapping:    
        x_mapped = 2 * (x - x_min) / (x_max - x_min) - 1
    if calculate_mp:
        x_mapped = (x_mapped[1:] + x_mapped[:-1]) / 2
    
    jacobi_value = eval_jacobi(n, alpha, beta, x_mapped)
    normalization_factor = 2*(x_max-x_min)/2
    
    return jacobi_value / normalization_factor    
    

# sig + bkg function
def sigbkg_func1(x, N_sig, N_bkg, mu, sigma, a, n, coef1, coef2, coef3, x_min=0, x_max=0, calculate_mp=False):
    normalization_factor = 2 # we don't need this normalization bc we defined for each sig and bkg function one coefficient (N_sig & N_bkg), we didn't defined one N for the whole function
    # c=0 in crystal_ball function
    result = (N_sig*norm_crystal_ball_const(x, mu, sigma, a, n, 0, x_min, x_max, True, calculate_mp)\
            +N_bkg*norm_legend(x, coef1, coef2, coef3, 1, x_min, x_max, True, calculate_mp)[0])
    return result

def sigbkg_func(x, N_sig, N_bkg, mu, sigma, a, n, bkg_param1, bkg_param2, bkg_param3, x_min=0, x_max=0, calculate_mp=False, bkg_func=None):
    normalization_factor = 2 # we don't need this normalization bc we defined for each sig and bkg function one coefficient (N_sig & N_bkg), we didn't defined one N for the whole function
    # c=0 in crystal_ball function
    # It would give you the option to change bkg func, make sure all bkg funcs you define have same number of parameters
    if bkg_func is None:
        bkg_func = norm_legend
        
    result = (N_sig*norm_crystal_ball_const(x, mu, sigma, a, n, 0, x_min, x_max, True, calculate_mp)\
            +N_bkg*bkg_func(x, bkg_param1, bkg_param2, bkg_param3, x_min=x_min, x_max=x_max, mapping=True, calculate_mp=calculate_mp)[0])
    
    return result


# ********************************************some functions used in fitting************************************************
# number of degrees of freedom
def ndf(xdata, params):
    return len(xdata) - len(params)

# round the value based on the uncertainty
def round_to_uncertainty(value, uncertainty):
    # round the uncertainty to 1-2 significant digits
    u = Decimal(uncertainty).normalize()
    # find position of the most significant digit
    exponent = u.adjusted()
    # precision = (uncertainty.as_tuple().digits[0] == 1)  # is the first digit 1?
    precision = 3
    u = u.scaleb(-exponent).quantize(Decimal(10)**-precision)
    
    # round the value to remove excess digits
    v = Decimal(value).scaleb(-exponent).quantize(u)

    # find out how many digits are after the decimal in the uncertainty
    after_decimal = -u.as_tuple().exponent

    # format the value as a string with the same number of decimal places
    v_str = f"{v:.{after_decimal}f}"

    # round the value to remove excess digits (this one doesn't print the same # of decimals as u)
    # return round(Decimal(value).scaleb(-exponent).quantize(u)), u, exponent
    

    # this one checks to make sure that the # of decimals between v and u is the same
    return v, u, exponent

# scientific form of reporting a value +- its ncertainty
def scientific_uncertainty_report(popt, pcov, param_names=None):
    # Initialize the string that will contain all the text
    text_string = ""
    # the enumerate will add a counter to the loop
    for n, (mean, err) in enumerate([(popt[param], np.sqrt(np.diag(pcov))[param]) for param in range(len(popt))]):
        value, uncertainty, exponent = round_to_uncertainty(mean, err)
        # text_string += fr"$a{n}: ({value} \pm {uncertainty}) \times 10^{exponent}$" + '\n' 
        # this one doesn't work bc the value is not a float
        # text_string += fr"$a{n}: ({value:.1e} \pm {uncertainty:.1e})$" + '\n' 
        if param_names!=None:
            text_string += fr"${param_names[n]}: ({value} \pm {uncertainty}).e({exponent})$" + '\n'
        else:
            text_string += fr"$a{n}: ({value} \pm {uncertainty}).e({exponent})$" + '\n'
        
    return text_string
    # because of transform=plt.gca().transAxes (0.5, 0.3) are in axes-relative coordinates
    # where (0, 0) is the bottom left of the plot and (1, 1) is the top right, regardless of the data coordinates
    # plt.text(0.53, 0.3, text_string, transform=plt.gca().transAxes, fontsize=30)
   
   
 
# mapping back the mu value from [-1, 1] to original InvM range
def mu_map_back(mu, x_range):
    mu_map_back = np.min(x_range) + (mu-(-1))*(np.max(x_range)-np.min(x_range))/(1-(-1))
    return mu_map_back
def mu_pcov_map_back(mu_pcov, x_range):
    return mu_pcov*((np.max(x_range)-np.min(x_range))/(1-(-1)))**2

# mapping back the sigma value from [-1, 1] to original InvM range
def sigma_map_back(sigma, x_range):
    return sigma*((np.max(x_range)-np.min(x_range))/(1-(-1)))
def sigma_pcov_map_back(sigma_pcov, x_range):
    return sigma_pcov*((np.max(x_range)-np.min(x_range))/(1-(-1)))**2
    
    
# ****************************************************fitting signal and background**********************************************
# plotting the signal(crystal_ball) and background(legendre polynomial with 4 params) with all the fitting info
class FittingResult:
    def __init__(self, params, covariance_matrix, chi_squared, ndf, p_value):
        self.params = params
        self.covariance_matrix = covariance_matrix
        self.chi_squared = chi_squared
        self.ndf = ndf
        self.p_value = p_value

    def __str__(self):
        return (
            f"Parameters: {self.params}\n"
            f"Covariance Matrix: {self.covariance_matrix}\n"
            f"Chi-squared: {self.chi_squared}\n"
            f"Degrees of Freedom: {self.ndf}\n"
            f"P-value: {self.p_value}\n"
        )

# this new function has the normal crystal_ball function plut a flat background with the appropriate normalization
def fitting_sig_bkg(
    rec_name, gen_name, x_fit_data_range, y_sig_data_range, y_bkg_data_range,
    max_ylim_bkg=700, param_text_location_sig=None, param_text_location_bkg=None, bkg_func=None,
    flat_bkg=None, p0_sig=None, bound_sig=(-np.inf, np.Inf), p0_bkg=None, bound_bkg=(-np.inf, np.Inf)):

    rec_name = r'$' + rec_name + r'$'
    gen_name = r'$' + gen_name + r'$'
    
    if param_text_location_sig is None:
        param_text_location_sig = (0.425, 1/3*np.max(y_sig_data_range))
    
    if param_text_location_bkg is None:
        param_text_location_bkg = (0.55, 1/2*np.max(y_bkg_data_range))
    
    # calculating the uncertainty of the data
    y_sig_err = np.sqrt(y_sig_data_range)
    y_bkg_err = np.sqrt(y_bkg_data_range)
    # middle point to show error bars plot
    x_fit_mp = (x_fit_data_range[1:] + x_fit_data_range[:-1])/2
    # x range for plotting
    x_range = np.linspace(np.min(x_fit_data_range), np.max(x_fit_data_range),100)
    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(20,20))
    
    # Fitting
    # signal
    result_sig = curve_fit(
    lambda x, N, mu, sigma, a, n, c: N*norm_crystal_ball_const(x, mu, sigma, a, n, c, np.min(x_fit_data_range), np.max(x_fit_data_range), mapping=True, calculate_mp=True)
    , x_fit_data_range, y_sig_data_range, p0=p0_sig, sigma=y_sig_err, bounds=bound_sig)
    # fitting parameters
    N_sig, mu, sigma, a, n, c = result_sig[0]
    
    # degrees of freedom
    ndf_sig = ndf(x_fit_mp-1, result_sig[0])
    # chi_squared
    residual = y_sig_data_range - N_sig*norm_crystal_ball_const(x_fit_data_range, mu, sigma, a, n, c, np.min(x_fit_data_range), np.max(x_fit_data_range), mapping=True, calculate_mp=True)
    residual_per_uncertainty = residual / y_sig_err
    chi_squared_sig = np.sum(residual_per_uncertainty**2)

    ax[0].errorbar(x_fit_mp, y_sig_data_range, yerr=y_sig_err, ls='', color='k', fmt='o', capsize=5, label='MC')
    ax[0].plot(x_range, N_sig*norm_crystal_ball_const(x_range, mu, sigma, a, n, c, np.min(x_fit_data_range), np.max(x_fit_data_range), mapping=True, calculate_mp=False), label='Fitting (norm_crystal)')
    ax[0].set_xticks(x_fit_mp, ['%.2f'%(tick) for tick in x_fit_mp], fontsize=14)
    ax[0].tick_params(axis='y', which='major', labelsize=16)
    ax[0].set_xlabel(r'$M_{\eta \rightarrow \gamma \gamma} (GeV)$', fontdict=form_label, labelpad=20)
    ax[0].set_ylabel('NOE/bin_width', fontdict=form_label, ha='center', va='center', labelpad=20)
    ax[0].set_title(f'Signal Fitting for Reconstructed as {rec_name} Generated as {gen_name}', fontdict=form_title)
    ax[0].text(0.56, 2/3*np.max(y_sig_data_range), 
    f'''Fitting Test
        $\chi^2/NDF$: {chi_squared_sig:.4f} / {ndf_sig}
        p_value: {1 - stats.chi2.cdf(chi_squared_sig, ndf_sig):.4f}
        ''', fontsize=30)
    l_param = [N_sig*100, mu_map_back(mu, x_fit_data_range), sigma_map_back(sigma, x_fit_data_range), a, n, c]
    l_variance = [result_sig[1][0]*100**2, mu_pcov_map_back(result_sig[1][1], x_fit_data_range), sigma_pcov_map_back(result_sig[1][2], x_fit_data_range), result_sig[1][3], result_sig[1][4], result_sig[1][5]]
    text = f'sig+flat_bkg ({flat_bkg}): {np.sum(y_sig_data_range):.1f}\n'+ '-'*30 + '\n' +\
    scientific_uncertainty_report(l_param, l_variance, ['N_{sig}', 'mu', 'sigma', 'a', 'n', 'c'])
    text_x_sig, text_y_sig = param_text_location_sig
    ax[0].text(text_x_sig, text_y_sig, text, fontsize=30)
    ax[0].legend(fontsize=18, loc='upper right')
    ax[0].grid(False)
    
    signal = {
            'popt': result_sig[0].tolist(),
            'pcov': result_sig[1].tolist(),
            'params': l_param,
            'uncertainty': np.sqrt(np.diag(l_variance)),
            'chi_squared': chi_squared_sig,
            'ndf': ndf_sig,
            'p_value': 1 - stats.chi2.cdf(chi_squared_sig, ndf_sig)
        }
        
    # ************background**********
    if bkg_func is None:
        print('you forgot to give me the background function')
    lambda_func = lambda x, N, bkg_param1, bkg_param2, bkg_param3: N*bkg_func(x, bkg_param1, bkg_param2, bkg_param3, x_min=np.min(x_fit_data_range), x_max=np.max(x_fit_data_range), mapping=True, calculate_mp=True)
    print(lambda_func)
    result_bkg = curve_fit(
    lambda x, N, bkg_param1, bkg_param2, bkg_param3: N*bkg_func(x, bkg_param1, bkg_param2, bkg_param3, x_min=np.min(x_fit_data_range), x_max=np.max(x_fit_data_range), mapping=True, calculate_mp=True)[0]
    ,x_fit_data_range, y_bkg_data_range, sigma=y_bkg_err, bounds=bound_bkg, p0=p0_bkg)
    # fitting parameters
    N_bkg, bkg_param1, bkg_param2, bkg_param3 = result_bkg[0]
    
    # degrees of freedom
    ndf_bkg = ndf(x_fit_mp-1, result_bkg[0])
    # chi_squared
    residual = y_bkg_data_range - N_bkg*bkg_func(x_fit_data_range, bkg_param1, bkg_param2, bkg_param3, x_min=np.min(x_fit_data_range), x_max=np.max(x_fit_data_range), mapping=True, calculate_mp=True)[0]
    residual_per_uncertainty = residual / y_bkg_err
    chi_squared_bkg = np.sum(residual_per_uncertainty**2)
   
    ax[1].errorbar(x_fit_mp, y_bkg_data_range, yerr=y_bkg_err, ls='', color='k', fmt='o', capsize=5, label='MC')
    ax[1].plot(x_range, N_bkg*bkg_func(x_range, bkg_param1, bkg_param2, bkg_param3, x_min=np.min(x_fit_data_range), x_max=np.max(x_fit_data_range), mapping=True, calculate_mp=False)[0], label=f'Fitting ({bkg_func.__name__})')
    ax[1].set_xticks(x_fit_mp, ['%.2f'%(tick) for tick in x_fit_mp], fontsize=14)
    ax[1].tick_params(axis='y', which='major', labelsize=16)
    ax[1].set_ylim(0, max_ylim_bkg)
    ax[1].set_xlabel(r'$M_{\eta \rightarrow \gamma \gamma} (GeV)$', fontdict=form_label, labelpad=20)
    ax[1].set_ylabel('NOE/bin_width', fontdict=form_label, ha='center', va='center', labelpad=20)
    ax[1].set_title(f'Background Fitting for Reconstructed as {rec_name} Generated as {gen_name}', fontdict=form_title)
    ax[1].text(0.46, 4/5*np.max(y_bkg_data_range), 
    f'''Fitting Test
        $\chi^2/NDF$: {chi_squared_bkg:.4f} / {ndf_bkg}
        p_value: {1 - stats.chi2.cdf(chi_squared_bkg, ndf_bkg):.4f}
        ''', fontsize=30)
    l_param = [N_bkg*100, bkg_param1, bkg_param2, bkg_param3]
    l_variance = [result_bkg[1][0]*100**2, result_bkg[1][1], result_bkg[1][2], result_bkg[1][3]]
    
    if bkg_func is norm_legend or norm_chebyshev:
        l_param_name = ['N_{bkg}', 'coef_1', 'coef_2', 'coef_3']
    elif bkg_func is norm_jacobi:
        l_param_name = ['N_{bkg}', 'n', '\\alpha', '\\beta']
    
    text = f'Total bkg: {np.sum(y_bkg_data_range):.1f}\n'+ '-'*30 + '\n' +\
    scientific_uncertainty_report(l_param, l_variance, l_param_name)
    text_x_bkg, text_y_bkg = param_text_location_bkg
    ax[1].text(text_x_bkg, text_y_bkg, text, fontsize=30)
    ax[1].legend(fontsize=18, loc='lower left')
    ax[1].grid(False)
    
    background = {
            'popt': result_bkg[0].tolist(),
            'pcov': result_bkg[1].tolist(),
            'params': l_param,
            'uncertainty': np.sqrt(np.diag(l_variance)),
            'chi_squared': chi_squared_bkg,
            'ndf': ndf_bkg,
            'p_value': 1 - stats.chi2.cdf(chi_squared_bkg, ndf_bkg)
        }
    
    # Create a dictionary to store the results
    results = {
        'signal': signal,
        'background': background
    }
    
    # Create FittingResult objects for signal and background
#     signal_result = FittingResult(
#         params=result_sig[0].tolist(),
#         covariance_matrix=result_sig[1].tolist(),
#         chi_squared=chi_squared_sig,
#         ndf=ndf_sig,
#         p_value=1 - stats.chi2.cdf(chi_squared_sig, ndf_sig)
#     )
    
#     background_result = FittingResult(
#         params=result_bkg[0].tolist(),
#         covariance_matrix=result_bkg[1].tolist(),
#         chi_squared=chi_squared_bkg,
#         ndf=ndf_bkg,
#         p_value=1 - stats.chi2.cdf(chi_squared_bkg, ndf_bkg)
#     )

#     # Create a dictionary to store the results
#     results = {
#         'signal': signal_result,
#         'background': background_result
#     }
    
    return results


# ************************************************fitting signal+background****************************************
def fitting_sig_plus_bkg(
    x_data, sigbkg_bin_count, result_sig_bkg, label, mclumi, text_params_position, max_ylim=700, MC_info_loc=(0.425, 200), fitting_info_loc=(0.425, 500),
    legend_loc='lower right', bkg_func=None, part_of_sigma=None, p0=None, bounds=(-np.inf, np.inf),
    float_legendre=False, float_mu_legendre=False, float_mu=False, float_musigma_legendre=False, float_mu_sigma=True):
    
    form_title = {'family': 'helvetica', 'color': 'black', 'size': 15}
    form_label = {'family': 'helvetica', 'color': 'black', 'size': 20}
    
    if bkg_func is None:
        bkg_func = norm_legend
    title = input("What's the title for your plot? ")
    # example: Reconstructed \ B^+B^- \ Generated \ B^+B^-
    formatted_title = r'$' + title + r'$'
    # data
    if len(sigbkg_bin_count)==2:
        y_data = sigbkg_bin_count[1]
    elif len(sigbkg_bin_count)==3:
        y_data = sigbkg_bin_count[2] - sigbkg_bin_count[1] + sigbkg_bin_count[0]
    else:
        y_data = sigbkg_bin_count
    # the uncertainty of y values
    y_err = np.sqrt(y_data)

    if float_legendre:
        # fixed params
        # signal
        mu, sigma, a, n = result_sig_bkg['signal']['popt'][1:5]
        if part_of_sigma:
            sigma = sigma - part_of_sigma * sigma
            print('reduced sigma: ', sigma_map_back(sigma, x_data))
        lambda_func =  lambda x, N_sig, N_bkg, coef1, coef2, coef3:sigbkg_func(x, N_sig, N_bkg, mu, sigma, a, n, coef1, coef2, coef3, np.min(x_data), np.max(x_data), True, bkg_func)
    elif float_mu_legendre:
        # fixed params
        # signal
        sigma, a, n = result_sig_bkg['signal']['popt'][2:5]
        if part_of_sigma:
            sigma = sigma - part_of_sigma * sigma
            print('reduced sigma: ', sigma_map_back(sigma, x_data))
        lambda_func = lambda x, N_sig, N_bkg, mu, coef1, coef2, coef3:sigbkg_func(x, N_sig, N_bkg, mu, sigma, a, n, coef1, coef2, coef3, np.min(x_data), np.max(x_data), True, bkg_func)
    elif float_mu:
        # sigma
        sigma, a, n = result_sig_bkg['signal']['popt'][2:5]
        # background'
        coef1, coef2, coef3 = result_sig_bkg['background']['popt'][1:]
        lambda_func = lambda x, N_sig, N_bkg, mu:sigbkg_func(x, N_sig, N_bkg, mu, sigma, a, n, coef1, coef2, coef3, np.min(x_data), np.max(x_data), True, bkg_func)
    elif float_musigma_legendre:
        # fixed params
        # signal
        a, n = result_sig_bkg['signal']['popt'][3:5]
        lambda_func = lambda x, N_sig, N_bkg, mu, sigma, coef1, coef2, coef3:sigbkg_func(x, N_sig, N_bkg, mu, sigma, a, n, coef1, coef2, coef3, np.min(x_data), np.max(x_data), True, bkg_func)
    elif float_mu_sigma:
        # fixed params
        # signal
        a, n = result_sig_bkg['signal']['popt'][3:5]
        # background'
        coef1, coef2, coef3 = result_sig_bkg['background']['popt'][1:]
        lambda_func = lambda x, N_sig, N_bkg, mu, sigma:sigbkg_func(x, N_sig, N_bkg, mu, sigma, a, n, coef1, coef2, coef3, np.min(x_data), np.max(x_data), True, bkg_func)
    
    
    if bounds != (-np.inf, np.inf):
        popt, pcov = curve_fit(lambda_func, x_data, y_data, p0=p0, bounds=bounds, sigma=y_err)
    else:
        popt, pcov, info, msg, ier = curve_fit(lambda_func, x_data, y_data, p0=p0, bounds=bounds, full_output=True, sigma=y_err)
   
    # parameters from fitting
    if float_legendre:
        N_sig, N_bkg, coef1, coef2, coef3 = popt
        fitting_params_names=['N_{sig}', 'N_{bkg}', 'coef_{1}', 'coef_{2}', 'coef_{3}']
        l_param = [N_sig*100, N_bkg*100, coef1, coef2, coef3] # the bin_width is 0.01 (N/0.01 = N*100)
        l_uncertainty = [pcov[0]*100**2, pcov[1]*100**2, pcov[2], pcov[3], pcov[4]]
    elif float_mu_legendre:
        N_sig, N_bkg, mu, coef1, coef2, coef3 = popt
        fitting_params_names=['N_{sig}', 'N_{bkg}', 'mu', 'coef_{1}', 'coef_{2}', 'coef_{3}']
        l_param = [N_sig*100, N_bkg*100, mu_map_back(mu, x_data), coef1, coef2, coef3] # the bin_width is 0.01 (N/0.01 = N*100)
        l_uncertainty = [pcov[0]*100**2, pcov[1]*100**2, mu_pcov_map_back(pcov[2], x_data), pcov[3], pcov[4], pcov[5]]
    elif float_mu:
        N_sig, N_bkg, mu = popt
        fitting_params_names=['N_{sig}', 'N_{bkg}', 'mu']
        l_param = [N_sig*100, N_bkg*100, mu_map_back(mu, x_data)] # the bin_width is 0.01 (N/0.01 = N*100)
        l_uncertainty = [pcov[0]*100**2, pcov[1]*100**2, mu_pcov_map_back(pcov[2], x_data)]
    elif float_musigma_legendre:
        N_sig, N_bkg, mu, sigma, coef1, coef2, coef3 = popt
        fitting_params_names=['N_{sig}', 'N_{bkg}', 'mu', 'sigma', 'coef_{1}', 'coef_{2}', 'coef_{3}']
        l_param = [N_sig*100, N_bkg*100, mu_map_back(mu, x_data), sigma_map_back(sigma, x_data), coef1, coef2, coef3] # the bin_width is 0.01 (N/0.01 = N*100)
        l_uncertainty = [pcov[0]*100**2, pcov[1]*100**2, mu_pcov_map_back(pcov[2], x_data), sigma_pcov_map_back(pcov[3], x_data), pcov[4], pcov[5], pcov[6]]
    elif float_mu_sigma:
        N_sig, N_bkg, mu, sigma = popt
        fitting_params_names=['N_{sig}', 'N_{bkg}', 'mu', 'sigma']
        l_param = [N_sig*100, N_bkg*100, mu_map_back(mu, x_data), sigma_map_back(sigma, x_data)] # the bin_width is 0.01 (N/0.01 = N*100)
        l_uncertainty = [pcov[0]*100**2, pcov[1]*100**2, mu_pcov_map_back(pcov[2], x_data), sigma_pcov_map_back(pcov[3], x_data)]
    

    print('{:} {:>30s} {:>18s}'.format('N_sig + N_bkg', 'Sum of sig + bkg entries', 'difference'))
    print('%.4d %28.4d %26.3f'
        %( ((N_sig+N_bkg)*100), (np.sum(y_data)),
        abs(np.sum(y_data) - (N_sig+N_bkg)*100) )
    )

    # middle points of x
    x_fit_mp = (x_data[1:] + x_data[:-1])/2
    # number of degrees of freedoms
    ndf_data = ndf(x_fit_mp, popt)
    # x range for plotting
    x_range = np.linspace(np.min(x_data), np.max(x_data), 100)

    plt.figure(figsize=(18,8))
    plt.errorbar(x_fit_mp, y_data, yerr=y_err, ls='', color='k', fmt='o', capsize=5, label=label)
    plt.plot(x_range, N_bkg*bkg_func(x_range, coef1, coef2, coef3, 1, np.min(x_data), np.max(x_data), True, False)[0], label=f'bkg: {bkg_func.__name__}')
    plt.plot(x_range, sigbkg_func(x_range, N_sig, N_bkg, mu, sigma, a, n, coef1, coef2, coef3, np.min(x_data), np.max(x_data), bkg_func=bkg_func), label='Fitting')
    plt.xticks(x_fit_mp, ['%.2f'%(tick) for tick in x_fit_mp], fontsize=14)
    plt.ylim(0, max_ylim)
    plt.xlabel(r'$M_{\eta \rightarrow \gamma \gamma} (GeV)$', fontdict=form_label)
    plt.ylabel('NOE/bin_width', fontdict=form_label, labelpad=14)
    plt.title(formatted_title, fontdict=form_title)
    text = '$\int \mathcal{L} dt = $' + f'{mclumi:.2f}' + r' fb$^{-1}$'
    plt.annotate(text, xy=(0, 1.01), 
                fontsize=16,
                xycoords='axes fraction',
                horizontalalignment='left',
                verticalalignment='bottom')

    
    text = scientific_uncertainty_report(l_param, l_uncertainty, fitting_params_names)
    x_text, y_text = text_params_position
    plt.text(x_text, y_text, text, fontsize=30)
    
    if bounds != (-np.inf, np.inf):
        residual = y_data - sigbkg_func(x_data, N_sig, N_bkg, mu, sigma, a, n, coef1, coef2, coef3, np.min(x_data), np.max(x_data), True, bkg_func)
        residual_per_uncertainty = residual / y_err
        chi_squared = np.sum(residual_per_uncertainty**2)
    else:
        chi_squared = np.dot(info['fvec'], info['fvec'])
    x_fitting_info, y_fitting_info = fitting_info_loc
    plt.text(x_fitting_info, y_fitting_info,
    f'''Fitting Test
        $\chi^2/NDF$: {chi_squared:.4f} / {ndf_data}
        p_value: {1 - stats.chi2.cdf(chi_squared, ndf_data):.7f}
        ''', fontsize=30)
    
    if label=='MC':
        x_MC_info, y_MC_info = MC_info_loc
        plt.text(x_MC_info, y_MC_info,
        f"""Sum of entries from {label} not fitting:
        Signal: {np.sum(y_data-sigbkg_bin_count[0]):.2f}
        Background: {np.sum(sigbkg_bin_count[0]):.2f}
        """, fontsize=20)

    plt.legend(fontsize=18, ncol=3, loc=legend_loc)
    plt.show()
    
    
    uncertainty = np.sqrt(np.diag(l_uncertainty))
    
    return {
        'popt': popt,
        'pcov': pcov,
        'params': l_param,
        'uncertainty': uncertainty
    }
    
    
# ****************************************Correlation Matrix***********************************
# Note: even if you wanna plot the corr_matrix for sig or bkg alone you should pass it as a list of including pcov(s)
# e.g. [pcov_sig] or [pcov_sig, pcov_bkg]
# Also for "param_names_list" you should make a list for sig_params and bkg_params and pass one list including two lists of two sets of params
# e.g. [['N_sig', 'mu', 'sigma', 'a', 'n'], ['N_bkg', 'coef1', 'coef2', 'coef3']]
def correlation_matrix(pcov_list, param_names_list):
    num_matrices = len(pcov_list)
    
    if num_matrices == 1:
        fig, ax = plt.subplots(figsize=(18,5))
        ax = [ax]
    else:
        fig, ax = plt.subplots(1, num_matrices, figsize=(18,5))
        
    for index, pcov in enumerate(pcov_list):
        std_devs = np.sqrt(np.diag(pcov))
        corr_matrix = pcov / np.outer(std_devs, std_devs)
        
        param_name = param_names_list[index]
        
        im = ax[index].imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax[index].set_xticks(np.arange(len(param_name)), param_name, fontsize=20, rotation=45)
        ax[index].set_yticks(np.arange(len(param_name)), param_name, fontsize=20)
        
        cb = plt.colorbar(im, ax=ax[index])
        cb.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.show()
    
# *************************************************Efficiency calsulation*********************************
# make sure the N values like N_bkg coming from fitting is divided by the bin_width
# also whatever you do to N you should do the same to the power of 2 to its uncertainty
def calculate_efficiency(mc_Nsig, uncertainty_Nsig, mc_Npkg, mc_BF, NBB):
    # Calculate efficiency
    # signal is sig + pkg so I should remove peaking background from signal
    Nsig = mc_Nsig - mc_Npkg
    efficiency = Nsig / (2 * mc_BF * NBB)
    
    # Calculate uncertainties
    delta_Nsig = uncertainty_Nsig
    delta_Npkg = np.sqrt(mc_Npkg)
    delta_NBB = np.sqrt(NBB)
    
    # Error propagation
    common_term = 1 / (2 * mc_BF * NBB)
    delta_efficiency = np.sqrt(
        (common_term * delta_Nsig)**2 +
        (-common_term * delta_Npkg)**2 +
        ((- Nsig / (2 * mc_BF * NBB**2)) * delta_NBB)**2
    )
    
    return efficiency, delta_efficiency

# ***************************************************BF calculation****************************************
# make sure the N values like Nbkg coming from fitting is divided by the bin_width
# also whatever you do to N you should do the same to the power of 2 to its uncertainty
def calculate_BF(data_Nsig, uncertainty_Nsig, mc_Npkg, mc_efficiency, delta_efficiency, NBB):
    # Calculate efficiency
    Nsig = data_Nsig - mc_Npkg
    decay_BF = Nsig / (2 * mc_efficiency * NBB)
    
    # Calculate uncertainties
    delta_Nsig = uncertainty_Nsig
    delta_Npkg = np.sqrt(mc_Npkg)
    delta_NBB = np.sqrt(NBB)
    
    # Error propagation
    common_term = 1 / (2 * mc_efficiency * NBB)
    delta_decay_BF = np.sqrt(
        (common_term * delta_Nsig)**2 +
        (-common_term * delta_Npkg)**2 +
        ((- Nsig / (2 * mc_efficiency * NBB**2)) * delta_NBB)**2 +
        ((- Nsig / (2 * NBB * mc_efficiency**2)) * delta_efficiency)**2
    )
    
    return decay_BF, delta_decay_BF
