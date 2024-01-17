# !/usrs/sahargholipour/opt/anaconda3/bin/python
# coding: utf-8

# In[11]:

# splitting the signal and background. One B reconstructed as Detalnu and the other of Dlnu

# right now this function is only for B-B+ reconstructed, later we add all reconstructed modes but at that point your function better ask which reconstructed modes should be used
# argument gen can be charged or mixed
def sig_bkg_spliter(dataset, gen=None, sigpkg_output=False, full_output=False):
    
    # charged Detalnu reconstructed modes
    l_eta_Bp = [1600,1700,2600,2700]
    eta_rec_Bp = list(i+1 for i in l_eta_Bp) + l_eta_Bp
    
    # generated modes
    # charged Detalnu and DDs generated modes
    eta_gen_Bp = [1019,1020,1039,1040]
    DDs_gen_Bp = [i+1681 for i in range(4)]
    # mixed Detalnu and DDs generated modes
    eta_gen_B0 = [1017,1018,1035,1036]
    DDs_gen_B0 = [i+1833 for i in range(4)]
    
    # gen = gen
    
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
     ((abs(dataset[antiparticle_variable])%10000).isin(eta_gen_mode)) & ((dataset['B0_decayModeID']).isin(eta_rec_Bp)) & (dataset['pi4_B0_isSignal']==1)
    )
    
    |
    
    (
     (((dataset[particle_variable])%10000).isin(eta_gen_mode)) & ((dataset['B1_decayModeID']).isin(eta_rec_Bp)) & (dataset['pi4_B1_isSignal']==1)
    )
    ]
    # background
    eta_df_bkg = dataset.drop(eta_df_sig.index, inplace=False)
    # peaking background
    eta_df_pkg = eta_df_bkg[
        ( 
        ((abs(eta_df_bkg[antiparticle_variable])%10000).isin(DDs_gen_mode)) & ((eta_df_bkg['B0_decayModeID']).isin(eta_rec_Bp)) & (eta_df_bkg['pi4_B0_isSignal']==1)
        )
        
        |
        
        ( 
        ((abs(eta_df_bkg_BmBp[particle_variable])%10000).isin(DDs_gen_mode)) & ((eta_df_bkg_BmBp['B1_decayModeID']).isin(eta_rec_Bp)) & (eta_df_bkg_BmBp['pi4_B1_isSignal']==1)
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

# ********************************************************************************************
def bBest_cand_newcolumn(data):
    data = data.reset_index()
    cosBY0 = data['cosBY0'].to_numpy()
    cosBY1 = data['cosBY1'].to_numpy()
    event = data['__event__'].to_numpy()
    candidate = data['__candidate__'].to_numpy()
    Y4SScore = data['Y4SScore'].to_numpy()
    lastev = -1
    newBest = {}
    for i in range(len(event)):
        if event[i]==lastev:
            if bestY4SScore-Y4SScore[i] < 2 and ( abs(bestCosBY0)>1.1 or abs(bestCosBY1)>1.1 ) and \
                (bestCosBY0*bestCosBY0+bestCosBY1*bestCosBY1) > cosBY0[i]*cosBY0[i]+cosBY1[i]*cosBY1[i] :
                newBest[lastev] = candidate[i]
                bestCosBY0 = cosBY0[i]
                bestCosBY1 = cosBY1[i]
        else:
            newBest[event[i]] = 0
            bestY4SScore = Y4SScore[i]
            bestCosBY0 = cosBY0[i]
            bestCosBY1 = cosBY1[i]
            lastev = event[i]

    myBest = np.zeros(len(event))
    for i in range(len(event)):
        myBest[i] = newBest[event[i]]

    new_column=pd.DataFrame({'bBest':myBest})
    new_data=pd.concat([data,new_column],axis=1)
    
    return new_data

def mincand_eachevent(data):
    data = data.reset_index()
#   split the data set to different groups of events
    data_grouped = data.groupby(['__event__'])
#   rows where the __candidate__ is minimum in each event
    mincan_index = data_grouped['__candidate__'].idxmin()
#   the whole data set with best candidate(minmum value for __candidate__)
    firstmin_can = data.loc[mincan_index]
#   remove the rows with minimum value for __candidate__, so there're duplicates(same event slightly different candidate)
    except_firsmin = data.drop(mincan_index)
    
    return firstmin_can, except_firsmin
    
    
def bestcan(data, reset_index=True):
    if reset_index:
        data = data.reset_index()
        
    mincan_mask = data['__candidate__'] == data.groupby('__event__')['__candidate__'].transform('min')
    mincan_data = data[mincan_mask].copy()

    # Remove mincan_data from the original dataset
    nomincan_data = data.drop(mincan_data.index, axis=0)

    return mincan_data, nomincan_data

# ********************************************************************************************

# number of degrees of freedom
def ndf(xdata, params):
    return len(xdata) - len(params)

# ********************************************************************************************

# significance calculation
def significance(background, signal):
    n_background = len(background)
    n_signal = len(signal)
    return n_signal / pow((n_background + n_signal),0.5)

# ********************************************************************************************

# finding the best number of bins to plot signal and background on top of each other
# Note: this function is working based on no zeros in signal bin counts
def find_best_number_of_bins1(data_sig, data_bkg, n_start=30):
    n_bins = n_start
    iteration = 0

    while True:
        bins = np.linspace(np.min(data_bkg), np.max(data_bkg), n_bins)
        bin_count, _ = np.histogram(data_sig, bins=bins)

        if np.any(bin_count == 0):
            n_bins -= 1
            iteration += 1
        else:
            return n_bins

    return n_bins

# Note: this function working based on no zeros in the middle bins of signal counts (at the beginning and at the end is fine)
def find_best_number_of_bins2(data, max_bins):
    results = []
    for num_bins in range(1, max_bins + 1):
        counts, _ = np.histogram(data, bins=num_bins)
        if 0 not in counts[1:-1]:  # Check for zeros in the middle of the counts list
            results.append((num_bins, counts))
    return results


# plotting hist of signal and background with the bin width of 10 (MeV) or 0.01 (GeV)
def plot_sigbkg_InvM_hist(df_background, df_signal, y_axis_sapce=None, full_output=False, flat_bkg=False):
    
    # Invariant mass of a pair of photons
    particle_InvM_bkg = pd.concat([df_background['pi4_B0_InvM'].dropna(), df_background['pi4_B1_InvM'].dropna()])
    particle_InvM_sig = pd.concat([df_signal['pi4_B0_InvM'].dropna(), df_signal['pi4_B1_InvM'].dropna()])
    
    # I add a poisson distribution as a noise to my signal bin_counts in order not to get zero signal bin_counts
    # bc then the uncertainty for that bin will be zero and for chi_squared we would have problem
    # Note: the noise will add approximately one event in total to signal
    bins = np.arange(np.min(particle_InvM_bkg), np.max(particle_InvM_bkg), 0.01)
    poisson_values = np.random.poisson(100, len(bins)-1)
    normalized_poisson = poisson_values/np.sum(poisson_values)
    
    fig, ax = plt.subplots(figsize=(18,6))
    bin_count, bin_edge, _ = ax.hist(
        [particle_InvM_bkg, particle_InvM_sig], label=['background', 'signal'], histtype='barstacked', bins=bins)
    
    # Set x-axis ticks to show bin edges with one decimal place
    formatted_ticks2 = ['%.3f'%(tick) for tick in bin_edge]
    formatted_ticks = ['{:.3f}'.format(tick) for tick in bin_edge]
    ax.set_xticks(bin_edge, formatted_ticks, fontsize=14)
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.set_xlabel(r'$M_{\eta \rightarrow \gamma \gamma}$ (GeV)', fontdict=form_label)
    ax.set_ylabel(r'$Events/bin_{width}$', fontdict=form_label, rotation=90, ha='center', va='center', labelpad=20)
    # ax.yaxis.set_label_coords(-0.05, 1)
    # ax.set_xlim(eta_edge_sig_BmBp[0],eta_edge_sig_BmBp[-1])
    ax.set_ylim( 0, y_axis_sapce)

    # printing the bin width an dthe number of bins on plot
    bin_width = bin_edge[1]-bin_edge[0]
    bin_numbers = len(bin_count[0])
    denominator_text = r'${bin_{width}}: $' + f'{bin_width:.2f}' + r', ${bin_{numbers}}: $' + f'{bin_numbers}'
    at = AnchoredText(f'{denominator_text}', loc='upper center', frameon=True, pad=0.4, prop=dict(size=16))
    at2 = AnchoredText(fr'$bin_{{width}}$: {bin_width:.2f}', loc='upper left', frameon=False, pad=0.2)
    ax.add_artist(at)

    # showing the bin counts on the hist for signal and background seperately
    # va: vertical and ha: horizontal
    # background
    for edge, count in zip(bin_edge, bin_count[0]):
        x_coord = edge + (0.5 * (bin_edge[1] - bin_edge[0]))
        y_coord = count/2
        ax.text(x_coord, y_coord, str(int(count)), ha='center', va='center', fontsize=12)
    # signal
    for edge, count, total_count in zip(bin_edge, bin_count[1]-bin_count[0], bin_count[1]):
        x_coord = edge + (0.5 * (bin_edge[1] - bin_edge[0]))
        y_coord = total_count
        ax.text(x_coord, y_coord, str(int(count)), ha='center', va='bottom', fontsize=12)

    ax.legend(fontsize=18)
    
    # x range for fitting
    x_fitting = bin_edge
    # y range for background fitting
    y_bkg_fitting = bin_count[0]
    # y range for signal fitting
    # a uniform function 1 as a flat background below signal
    if flat_bkg:
        y_sig_fitting = (bin_count[1] - bin_count[0]) + np.ones(len(bins)-1)
    # an array of random poisson distribution numbers to each signal bin
    else:
        y_sig_fitting = (bin_count[1] - bin_count[0]) + normalized_poisson
    
    # checking if there's any zero in the signal bin_counts after adding the noise (either poisson or flat background)
    if np.any(y_sig_fitting==0):
        print('We still have the problem of 0 signal bin_counts')
    else:
        print('No zero in signal+noise bin_counts list')
        
    if full_output:
        return x_fitting, y_sig_fitting, y_bkg_fitting, bin_count, particle_InvM_sig, particle_InvM_bkg
    else:
        return x_fitting, y_sig_fitting, y_bkg_fitting, bin_count


# ********************************************************************************************
# signal and background functions

# crystalball function (non_normalized)
def crystal_ball(x, mu, sigma, N, a, n): 
#     crystal_ball(x, *params) when I didn't need to let the parameteres float
#     N, a, n, mu, sig = params
    aa = abs(a)
    A = ((n/aa)**n) * np.exp(- aa**2 / 2)
    B = n/aa - aa
    condition1 = lambda x: N * np.exp(- ((x-mu)**2)/(2.*sigma**2) )
    condition2 = lambda x: N * A * (B - (x-mu)/sigma)**(-n)
    result = np.where((x - mu)/sigma > (-a), condition1(x), condition2(x))

    return result

# Normalized crystalball function
def norm_crystal_ball(x, mu, sigma, a, n, mapping=False, calculate_mp=False):
    
    x_mapped = x
    if mapping:
        center = (np.max(x) + np.min(x)) / 2
        factor = 2 / (np.max(x) - np.min(x))
        x_mapped = ((x - center) * factor)
    if calculate_mp:
        x_mapped = (x_mapped[1:] + x_mapped[:-1]) / 2
        
    aa = abs(a)
    A = (n / aa) ** n * np.exp(- aa ** 2 / 2)
    B = n / aa - aa
    C = n * np.exp(- aa ** 2 / 2) / (aa * (n - 1))
    erf = (2/np.sqrt(np.pi)) * integrate.quad(lambda t: np.exp(- t**2), 0, aa/np.sqrt(2))[0]
    D = np.sqrt(np.pi / 2) * (1 + erf)
    # normalization factor
    
    N = 1 / (sigma * (C + D))
    
    condition1 = lambda x: N * np.exp(- ((x-mu)**2)/(2.*sigma**2) )
    condition2 = lambda x: N * A * (B - (x-mu)/sigma)**(-n)
    result = np.where((x_mapped - mu)/sigma > (-a), condition1(x_mapped), condition2(x_mapped))
    
    return result

# Normalized crystal_ball function with a constant in case we wanna add a flat background to our signal
def norm_crystal_ball_const(x, mu, sigma, a, n, c, mapping=False, calculate_mp=False, x_min=0, x_max=0):
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

# Simple crystalball function, removing the parameter n
def simple_crystal_ball(params, x):
    N, k, mu, sig = params
    kk = abs(k)
    condition1 = lambda x: N * exp(- ((x-mu)**2)/(2.*sig**2))
    condition2 = lambda x: N * exp( (kk**2/2.) + k * ((x-mu)/sig) )
    result = np.where((x - mu)/sig < (-k), condition2(x), condition1(x))
    
    return result


# exponential distribution
def expo(params, x):
    const, slope = params
    return const * exp(-slope*(x-0.5)) # try x-0.5


expo2 = lambda x, c, s: c * exp(-s*x)


# polynomial function with 5 parameters
def polynomial(params, x):
    a0,a1,a2,a3,a4 = params
    return (a0 - a1*x - a2*x**2 - a3*x**3 - a4*x**4)

# orthogonal polynomial (legendre), non-normalized and no mapping
def legendre(x, *coefs):
    return legval(x, coefs)

# fist trial
# legendre function with mapping inside and getting the middle point inside
def legendre_mod(x, *coefs, calculate_middle_point=True):
    center = (np.max(x) + np.min(x))/2
    factor = 2/(np.max(x)-np.min(x))
    
    x_mapped = ((x - center) * factor)
    
    if calculate_middle_point:
        x_mapped = (x_mapped[1:] + x_mapped[:-1]) / 2
    
    return legval(x_mapped, coefs)

# second trial
# mapping and calculating middle point inside but non-normalized
def legend(x, *coefs, calculate_mp=True, mapping=True):
    a, b, c, d = coefs
    # Fix the value of calculate_middle_point to True
    calculate_mp = lambda x: (x[1:] + x[:-1]) / 2 if calculate_mp else x
    
    x_mapped = x
    if mapping:
        center = (np.max(x) + np.min(x)) / 2
        factor = 2 / (np.max(x) - np.min(x))
        x_mapped = ((x - center) * factor)
    x_mapped = calculate_mp(x_mapped)
    
    leg = Legendre([a, b, c, d])
    
    return leg(x_mapped)
        
# normalized legendre function with mapping and middle point calculation inside
def norm_legend(x, coef1, coef2, coef3, coef0=1, mapping=False, calculate_mp=False, x_min=0, x_max=0):
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


def legendre_mod2(x, *coefs):
    a = np.min(x)
    b = np.max(x)
    x_mapped = (2 * (x - a) / (b - a)) - 1
    return legval(x_mapped, coefs)

# signal and background
def crysexpo(params, x, crys_params=None, expo_params=None):
    p1, p2 = params
    return p1 * crystal_ball(crys_params, x) + p2 * expo(expo_params, x)

# ********************************************************************************************
# sig + bkg function

def sigbkg_func(x, N_sig, N_bkg, mu, sigma, a, n, coef1, coef2, coef3, mapping=False, calculate_mp=False):
    
    x_mapped = x
    if mapping: 
        center = (np.max(x) + np.min(x)) / 2
        factor = 2 / (np.max(x) - np.min(x))
        x_mapped = ((x - center) * factor)
    
    if calculate_mp:
        x_mapped = (x_mapped[1:] + x_mapped[:-1]) / 2
    
    result = N_sig*norm_crystal_ball(x_mapped, mu, sigma, a, n, False, False)\
            +N_bkg*norm_legend(x_mapped, coef1, coef2, coef3, 1, False, False)[0]
    return result, x_mapped

# ************************************************************************************

def round_to_uncertainty(value, uncertainty):
    # round the uncertainty to 1-2 significant digits
    u = Decimal(uncertainty).normalize()
    # find position of the most significant digit
    exponent = u.adjusted()
    # precision = (uncertainty.as_tuple().digits[0] == 1)  # is the first digit 1?
    precision = 1
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

# ************************************************************************************

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
    
# ************************************************************************************

# plotting the signal(crystal_ball) and background(legendre polynomial with 4 params) with all the fitting info
def fitting_sig_bkg(rec_name, gen_name, x_fit_data_range, y_sig_data_range, y_bkg_data_range, p0_sig=None, p0_bkg=None, sig_full_output=False, bkg_full_output=False, flat_bkg=False):
    
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
    if flat_bkg:
        popt_sig, pcov_sig, info_sig, msg_sig, ier_sig = curve_fit(
        lambda x, N, mu, sigma, a, n, c: N*norm_crystal_ball_const(x, mu, sigma, a, n, mapping=True, calculate_mp=True)+c
        , x_fit_data_range, y_sig_data_range, p0=p0_sig, full_output=True, sigma=y_sig_err
        )
        # degrees of freedom
        ndf_sig = ndf(x_fit_mp-1, popt_sig)
        
        ax[0].errorbar(x_fit_mp, y_sig_data_range, yerr=y_sig_err, ls='', color='k', fmt='o', capsize=5, label='MC')
        ax[0].plot(x_range, popt_sig[0]*norm_crystal_ball(x_range, *popt_sig[1:-1], True, False)+popt_sig[-1], label='Fitting')
        ax[0].set_xticks(x_fit_mp, ['%.2f'%(tick) for tick in x_fit_mp], fontsize=14)
        ax[0].tick_params(axis='y', which='major', labelsize=16)
        ax[0].set_xlabel(r'$M_{\eta \rightarrow \gamma \gamma} (GeV)$', fontdict=form_label, labelpad=20)
        ax[0].set_ylabel('NOE/bin_width', fontdict=form_label, ha='center', va='center', labelpad=20)
        ax[0].set_title(f'Signal Fitting for Reconstructed as {rec_name} Generated as {gen_name}', fontdict=form_title)
        ax[0].text(0.56, 2/3*np.max(y_sig_data_range), 
        f'''Fitting Test
            $\chi^2/NDF$: {np.dot(info_sig['fvec'], info_sig['fvec']):.4f} / {ndf_sig}
            p_value: {1 - stats.chi2.cdf(np.dot(info_sig['fvec'], info_sig['fvec']), ndf_sig):.4f}
            ''', fontsize=30)
        l_param = [popt_sig[0]*10, popt_sig[1], popt_sig[2], popt_sig[3], popt_sig[4], popt_sig[5]]
        l_variance = [pcov_sig[0]*100, pcov_sig[1], pcov_sig[2], pcov_sig[3], pcov_sig[4], pcov_sig[5]]
        text = f'sig+flat_bkg: {np.sum(y_sig_data_range):.1f}\n'+ '-'*30 + '\n' +\
        scientific_uncertainty_report(l_param, l_variance, ['N_{sig}', 'mu', 'sigma', 'a', 'n', 'c'])
        ax[0].text(0.43,1/3*np.max(y_sig_data_range), text, fontsize=30)
        ax[0].legend(fontsize=18, loc='upper right')
        ax[0].grid(False)
    else:
        popt_sig, pcov_sig, info_sig, msg_sig, ier_sig = curve_fit(
        lambda x, N, mu, sigma, a, n: N*norm_crystal_ball(x, mu, sigma, a, n, mapping=True, calculate_mp=True)
        , x_fit_data_range, y_sig_data_range, full_output=True, sigma=y_sig_err, p0=p0_sig
        )
        # degrees of freedom
        ndf_sig = ndf(x_fit_mp-1, popt_sig)
        
        ax[0].errorbar(x_fit_mp, y_sig_data_range, yerr=y_sig_err, ls='', color='k', fmt='o', capsize=5, label='MC')
        ax[0].plot(x_range, popt_sig[0]*norm_crystal_ball(x_range, *popt_sig[1:], True, False), label='Fitting')
        ax[0].set_xticks(x_fit_mp, ['%.2f'%(tick) for tick in x_fit_mp], fontsize=14)
        ax[0].tick_params(axis='y', which='major', labelsize=16)
        ax[0].set_xlabel(r'$M_{\eta \rightarrow \gamma \gamma} (GeV)$', fontdict=form_label, labelpad=20)
        ax[0].set_ylabel('NOE/bin_width', fontdict=form_label, ha='center', va='center', labelpad=20)
        ax[0].set_title(f'Signal Fitting for Reconstructed as {rec_name} Generated as {gen_name}', fontdict=form_title)
        ax[0].text(0.56, 2/3*np.max(y_sig_data_range), 
        f'''Fitting Test
            $\chi^2/NDF$: {np.dot(info_sig['fvec'], info_sig['fvec']):.4f} / {ndf_sig}
            p_value: {1 - stats.chi2.cdf(np.dot(info_sig['fvec'], info_sig['fvec']), ndf_sig):.4f}
            ''', fontsize=30)
        l_param = [popt_sig[0]*10, popt_sig[1], popt_sig[2], popt_sig[3], popt_sig[4]]
        l_variance = [pcov_sig[0]*100, pcov_sig[1], pcov_sig[2], pcov_sig[3], pcov_sig[4]]
        text = f'sig+poisson_bkg: {np.sum(y_sig_data_range):.1f}\n'+ '-'*30 + '\n' +\
        scientific_uncertainty_report(l_param, l_variance, ['N_{sig}', 'mu', 'sigma', 'a', 'n'])
        ax[0].text(0.43,1/3*np.max(y_sig_data_range), text, fontsize=30)
        ax[0].legend(fontsize=18, loc='upper right')
        ax[0].grid(False)
        
    # background
    popt_bkg, pcov_bkg, info_bkg, msg_bkg, ier_bkg = curve_fit(
    lambda x, N, coef1, coef2, coef3: N*norm_legend(x, coef1, coef2, coef3, 1, True, True)[0]
    ,x_fit_data_range, y_bkg_data_range, full_output=True, sigma=y_bkg_err, p0=p0_bkg
    )
    # degrees of freedom
    ndf_bkg = ndf(x_fit_mp-1, popt_bkg)
   
    ax[1].errorbar(x_fit_mp, y_bkg_data_range, yerr=y_bkg_err, ls='', color='k', fmt='o', capsize=5, label='MC')
    ax[1].plot(x_range, popt_bkg[0]*norm_legend(x_range, *popt_bkg[1:], 1, True, False)[0], label='Fitting')
    ax[1].set_xticks(x_fit_mp, ['%.2f'%(tick) for tick in x_fit_mp], fontsize=14)
    ax[1].tick_params(axis='y', which='major', labelsize=16)
    ax[1].set_xlabel(r'$M_{\eta \rightarrow \gamma \gamma} (GeV)$', fontdict=form_label, labelpad=20)
    ax[1].set_ylabel('NOE/bin_width', fontdict=form_label, ha='center', va='center', labelpad=20)
    ax[1].set_title(f'Background Fitting for Reconstructed as {rec_name} Generated as {gen_name}', fontdict=form_title)
    ax[1].text(0.47, 2/3*np.max(y_bkg_data_range), 
    f'''Fitting Test
        $\chi^2/NDF$: {np.dot(info_bkg['fvec'], info_bkg['fvec']):.4f} / {ndf_bkg}
        p_value: {1 - stats.chi2.cdf(np.dot(info_bkg['fvec'], info_bkg['fvec']), ndf_bkg):.4f}
        ''', fontsize=30)
    l_param = [popt_bkg[0]*10, popt_bkg[1], popt_bkg[2], popt_bkg[3]]
    l_variance = [pcov_bkg[0]*100, pcov_bkg[1], pcov_bkg[2], pcov_bkg[3]]
    text = f'Total bkg: {np.sum(y_bkg_data_range):.1f}\n'+ '-'*30 + '\n' +\
    scientific_uncertainty_report(l_param, l_variance, ['N_{bkg}', 'coef1', 'coef2', 'coef3'])
    ax[1].text(0.56,1/2*np.max(y_bkg_data_range), text, fontsize=30)
    ax[1].legend(fontsize=18, loc='upper right')
    ax[1].grid(False)
    
    
    if sig_full_output:
        return popt_sig, pcov_sig, info_sig, msg_sig, ier_sig, y_sig_err
    if bkg_full_output:
        return popt_bkg, pcov_bkg, info_bkg, msg_bkg, ier_bkg, y_bkg_err
    else:
        return popt_sig, pcov_sig, popt_bkg, pcov_bkg
    
# ************************************************************************************

# Note: even if you wanna plot the corr_matrix for sig or bkg alone you should pass it as a list of including pcov(s)
# e.g. [pcov_sig] or [pcov_sig, pcov_bkg]
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
        ax[index].set_xticks(np.arange(len(param_name)), param_name, fontsize=20)
        ax[index].set_yticks(np.arange(len(param_name)), param_name, fontsize=20)
        
        cb = plt.colorbar(im, ax=ax[index])
        cb.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.show()


# ************************************************************************************

# [a,b] map to [c,d]
def mapping_range(variable_range, output_start, output_end): # which will be edges value for the variable
    a = np.min(variable_range)
    b = np.max(variable_range)
    c = output_start
    d = output_end
    # the mapped range
    y = [( ((x-a)*(d-c)/(b-a)) + c ) for x in variable_range]
    return y

# to return both the mapped range, and if I wanted to know what only one value from the first range will be in the second range
def mapping_range_modified(variable_range, output_start, output_end, input_value):
    a = np.min(variable_range)
    b = np.max(variable_range)
    c = output_start
    d = output_end
    mapped_range = [ ((x - a) * (d - c) / (b - a)) + c for x in variable_range ]
    return mapped_range, mapped_range[variable_range.index(input_value)]

# after mapping in signal fitting I get mu based on the mapped range which is not matched with the plot, so mu in real range is like 0.5 but then it shows 0.01
def mapping_range_backward(variable_range, input_value):
    a = -1
    b = 1
    c = np.min(variable_range)
    d = np.max(variable_range)
    mapped_value = ((input_value - a) * (d - c) / (b - a)) + c
    return mapped_value

# ************************************************************************************

# This function will sort the generated modes for particle and for each mode it will look at the anti_particle modes and sort them and print it in front of the particle mode
def genMode_frequency_B1_basedon_B0(data):
    reco = input("What are Bs reconstructed as: ")
    gen = input("What are Bs generated as: ")
    # the variables and their symbols
    if gen == 'BmBp':
        particle_variable = 'aBplusMode'
        particle_symbol = 'B+'
        anti_particle_variable = 'aBminusMode'
        anti_particle_symbol = 'B-'
    elif gen == 'Bbar0B0':
        particle_variable = 'aB0Mode'
        particle_symbol = 'B0'
        anti_particle_variable = 'aBbar0Mode'
        anti_particle_symbol = 'Bbar0'
    
    print('-' * 26, 'Reconstructed: ', reco, '-' * 21)
    print('-' * 26, 'Generated: ', gen, '-' * 25)
    
    # Group the DataFrame by particle_variable and calculate the frequency
    particle_variable_counts = data[particle_variable].value_counts().reset_index()
    particle_variable_counts.columns = [f'Generated {particle_symbol} Decay Mode ID', 'Frequency']

    # Sort the particle_variable counts based on frequency in descending order
    sorted_particle_variable_counts = particle_variable_counts.sort_values(by='Frequency', ascending=False)

    # Group the DataFrame by particle_variable and anti_particle_variable, and calculate the frequency
    grouped = data.groupby([particle_variable, anti_particle_variable]).size().reset_index(name='Frequency')

    # Create an empty dictionary to store the values for each particle_variable value
    values_dict = {}

    # Iterate over each particle_variable value
    for particle_value in sorted_particle_variable_counts[f'Generated {particle_symbol} Decay Mode ID']:
        # Filter the DataFrame for the current particle_variable value
        mode_values = grouped.loc[grouped[particle_variable] == particle_value]

        # Sort the mode_values based on frequency in descending order
        sorted_mode_values = mode_values.sort_values(by='Frequency', ascending=False)

        # Create a string representation of the anti_particle_variable values and their frequencies
        values_str = '\n'.join(f"{value}: {count}" for value, count in zip(sorted_mode_values[anti_particle_variable], sorted_mode_values['Frequency']))

        # Add the values to the dictionary
        values_dict[particle_value] = values_str

    # Create an empty list to store the rows of the table
    table_rows = []

    # Iterate over each particle_variable value
    for particle_value in sorted_particle_variable_counts[f'Generated {particle_symbol} Decay Mode ID']:
        # Get the values for the particle_variable value
        values = values_dict.get(particle_value, '')

        # Get the frequency of the particle_variable value
        frequency = sorted_particle_variable_counts.loc[sorted_particle_variable_counts[f'Generated {particle_symbol} Decay Mode ID'] == particle_value, 'Frequency'].values[0]

        # Create a row with the particle_variable value and the corresponding values (value: frequency)
        row = [f"{particle_value}: {frequency}", values]

        # Append the row to the table_rows list
        table_rows.append(row)

    # Create the DataFrame for the table
    table_df = pd.DataFrame(table_rows, columns=[f'Generated {particle_symbol} Decay Mode ID: Frequency', f'Generated {anti_particle_symbol} Decay Mode ID: Frequency'])

    # Generate the formatted table
    table_str = tabulate(table_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center')

    # Print the table
    print(table_str)

# counting the frequency of different decay modes
# This function will look at the generated decay modes for both particle and anti_particle separately and sort them based on the frequency of the modes
import pandas as pd
from tabulate import tabulate
def genMode_frequency(data):
    reco = input("What are Bs reconstructed as: ")
    gen = input("What are Bs generated as: ")
    
    if gen=='BmBp':
        particle_variable = 'aBplusMode'
        particle_symbol = 'B+'
        anti_particle_variable = 'aBminusMode'
        anti_particle_symbol = 'B-'
    elif gen=='Bbar0B0':
        particle_variable = 'aB0Mode'
        particle_symbol = 'B0'
        anti_particle_variable = 'aBbar0Mode'
        anti_particle_symbol = 'Bbar0'
    
    # the variables and their symbols
#     particle_variable = input("Enter the particle variable name: ")
#     particle_symbol = input("Enter the particle symbol: ")

#     anti_particle_variable = input("Enter the anti particle variable name: ")
#     anti_particle_symbol = input("Enter the anti particle symbol: ")
    
    print('-'*45,'Reconstructed: ', reco, '-'*41)
    print('-'*45, 'Generated: ', gen, '-'*45)
    
    # Count the frequency of each decay mode
    decay_mode_counts_B = data[particle_variable].value_counts()

    # Create a new DataFrame to store the decay modes and their corresponding frequencies
    decay_mode_table_B = pd.DataFrame({f'Generated {particle_symbol} Decay Mode ID': decay_mode_counts_B.index, 'Frequency': decay_mode_counts_B.values})

    # Sort the decay modes based on frequency in descending order
    sorted_decay_modes_B = decay_mode_table_B.sort_values(by='Frequency', ascending=False)

    # Count the frequency of each decay mode
    decay_mode_counts_Bbar = data[anti_particle_variable].value_counts()

    # Create a new DataFrame to store the decay modes and their corresponding frequencies
    decay_mode_table_Bbar = pd.DataFrame({f'Generated {anti_particle_symbol} Decay Mode ID': decay_mode_counts_Bbar.index, 'Frequency': decay_mode_counts_Bbar.values})

    # Sort the decay modes based on frequency in descending order
    sorted_decay_modes_Bbar = decay_mode_table_Bbar.sort_values(by='Frequency', ascending=False)

    # Generate the formatted tables
    decay_mode_table_str_B = tabulate(sorted_decay_modes_B, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center').split('\n')
    decay_mode_table_str_Bbar = tabulate(sorted_decay_modes_Bbar, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center').split('\n')

    # Determine the longer and shorter tables
    longer_table = decay_mode_table_str_B if len(decay_mode_table_str_B) >= len(decay_mode_table_str_Bbar) else decay_mode_table_str_Bbar
    shorter_table = decay_mode_table_str_B if longer_table == decay_mode_table_str_Bbar else decay_mode_table_str_Bbar

    # Calculate the difference in length between the tables
    length_diff = len(longer_table) - len(shorter_table)

    # Pad the shorter table with empty rows
    shorter_table += [''] * length_diff

    # Add the total frequency count row to each table
    total_frequency = sum(sorted_decay_modes_B['Frequency'])
    longer_table.append(f'Total Frequency: {total_frequency: <40}')
    shorter_table.append(f'Total Frequency: {total_frequency}')

    # Combine the tables side by side
    combined_table = ''
    for row_l, row_s in zip(longer_table, shorter_table):
        combined_table += f'{row_l: <30}   {row_s}\n'

    # Print the combined table
    print(combined_table)


# ************************************************************************************

# initial parameters function
def get_param(xdata, ydata, func):
    if func == crystal_ball:
        N = ydata.max()
        xb = xdata[ydata.argmax()]
        sigma = sum(ydata > ydata.mean())*1. / len(xdata) * (xdata.max() - xdata.min())
        n = 2.
        a = 2
        return ([N, a, n, xb, sigma])
    elif func == simple_crystal_ball:
        N = ydata.max()
        xb = xdata[ydata.argmax()]
        sigma = sum(ydata > ydata.mean())*1. / len(xdata) * (xdata.max() - xdata.min())
        k = random.uniform(0,abs(np.min(xdata)-xb)/sigma)
        return array([N, k, xb, sigma])
    elif func == expo:
        if ydata[0] < ydata[-1]:
            const = random.uniform(50*(np.min(ydata)), np.min(ydata))
        elif ydata[0] > ydata[-1]:
            const = random.uniform(np.max(ydata), 50*(np.max(ydata)))
        slope = 8
        return([const, slope])
    elif func == polynomial:
        a0 = random.uniform(np.max(ydata), 50*(np.max(ydata)))
        a1 = (ydata[-1]-ydata[0]) / (xdata[-1] - xdata[0])
        a2 = 4
        a3 = 2
        a4 = 0.4
        return array([a0,a1,a2,a3,a4])
    elif func == expocrys:
        p1 = 0.06
        p2 = 0.96
        return array([p1, p2])
    
    
# **************************************************************************************    


# Fitting ODR
def fit(func, x,y, y_uncertainty, default_pars=None, data_range=None, we=None, verbose=False, itmax=1000):

  # If this is a histogram output, correct it for the user.
    if len(x) == len(y) + 1:
        x = (x[1:] + x[:-1])/2.
  # Take a slice of data.
  # so means if we give only a slice of our histogram the function will take the x and y values of that part
    if data_range: # is eqaul to say if data_range != None:
        y = y[logical_and(x > data_range[0], x < data_range[1])]
        x = x[logical_and(x > data_range[0], x < data_range[1])]

  # http://www.scipy.org/doc/api_docs/SciPy.odr.odrpack.html
  # see models.py and use ready made models!!!!
    if default_pars != None:
        beta0 = array(default_pars)
    else:
#         beta0 = get_default_params(x,y,func)
        beta0 = get_param(x, y, func)
    model_func = models.Model(func)
  
    # mydata = odr.Data(x, y, we)
    data = RealData(x, y, sy=y_uncertainty)
    myodr  = ODR(data, model_func,maxit=itmax, beta0=beta0)

  # Set type of fit to least-squares (fit_type=2):    
    myodr.set_job(fit_type=2)
  # final can be 0 (no report), 1 (short report), 2 (long report)    
    if verbose == 2: myodr.set_iprint(final=2)
        
    fit = myodr.run()

  # Display results:
    if verbose: fit.pprint(), print('verbose: ',verbose)

    if fit.stopreason[0] == 'Iteration limit reached':
        print('(WWW) poly_lsq: Iteration limit reached, result not reliable!')

  # Results and errors
    coeff = fit.beta
    err   = fit.sd_beta
    covariance_matrix = fit.cov_beta
    chi   = fit.sum_square
    res_var = fit.res_var

  # The resulting fit.
    xfit = linspace( min(x), max(x), len(x)*10)
    yfit = func(fit.beta, xfit)

    return array([xfit,yfit]),coeff,err,covariance_matrix,chi, res_var



# **********************************************************************************************


# Adding boundaries to ODR fitting function. It's not the same as curve_fit which has the argument for the boundry
# import numpy as np
# from scipy.odr import Model, Data, ODR

# # Define the exponential function
# def exponential_function(beta, x):
#     c, s = beta
#     return c * np.exp(-s * x)

# # Define a custom Model class with bounds on the parameters
# class BoundedExponentialModel(Model):
#     def __init__(self, fcn, bounds):
#         Model.__init__(self, fcn)
#         self.bounds = bounds

#     def fcn(self, beta, x):
#         c, s = beta

#         # Enforce the bounds
#         lower_bounds, upper_bounds = self.bounds
#         c = np.clip(c, lower_bounds[0], upper_bounds[0])
#         s = np.clip(s, lower_bounds[1], upper_bounds[1])

#         return exponential_function((c, s), x)

# # Define your data (x_data and y_data) here
# x_data = x_bkg
# y_data = y_bkg

# # Set the initial parameter estimates (beta0)
# beta0 = [2e5, 8]

# # Define the bounds for the parameters
# bounds = ([beta0[0]-1, beta0[0]+1], [np.inf, np.inf])
# print(bounds)

# # Create the custom model and data instances
# model = BoundedExponentialModel(exponential_function, bounds)
# data = Data(x_data, y_data)

# # Create the ODR instance
# odr = ODR(data, model, beta0=beta0)

# # Run the fit and get the output
# output = odr.run()

# # Print the fitted parameters
# print('Fitted parameters:', output.beta, 'Errors: ', output.sd_beta)






