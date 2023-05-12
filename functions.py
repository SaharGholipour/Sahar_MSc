#!/usr/bin/env python
# coding: utf-8

# In[11]:

# splitting the signal and background. One B reconstructed as Detalnu and the other of Dlnu

def sig_bkg_spliter(y4s_id,particle_name,dataset):
    
    
    if y4s_id=='charged':
        
        # Detalnu reconstructed modes
        l_rec_mode = [1600,1700,2600,2700]
        eta_rec_Bp = list(i+1 for i in l_rec_mode) + l_rec_mode
        # Detalnu generated modes
        eta_gen_Bp = [1019,1020,1039,1040]
        
        # Dpi0lnu reconstructed modes
        l_pi0_Bp = [1300, 1200, 2300, 2200]
        pi0_rec_Bp = list(i+1 for i in l_pi0_Bp) + l_pi0_Bp
        # Dpi0lnu generated modes
        pi0_gen_Bp = [1008,1010,1028,1030]
        
        if particle_name == 'eta':
            rec_mode = eta_rec_Bp
            gen_mode = eta_gen_Bp
        elif particle_name == 'pi0':
            rec_mode = pi0_rec_Bp
            gen_mode = pi0_gen_Bp
            

        # taking the events in which one B as Dlnu and the other one as Detalnu or Dpi0lnu have been reconstructed
        df_Bp = dataset.loc['charged'].query(
            f'((B0_decayModeID=={rec_mode} & B1_decayModeID<500) | (B1_decayModeID=={rec_mode} & B0_decayModeID<500))'
        )

        dfp_bkg = df_Bp.drop(
            df_Bp[
                (
                    ( (df_Bp['aBplusMode']%10000).isin(gen_mode) & (df_Bp['B1_decayModeID']).isin(rec_mode) ) 
                    |
                    ( (abs(df_Bp['aBminusMode'])%10000).isin(gen_mode) & (df_Bp['B0_decayModeID']).isin(rec_mode) )
                )
                &
                ( 
                    (df_Bp['pi4_B0_isSignal']==1) | (df_Bp['pi4_B1_isSignal']==1) 
                ) 
            ].index, inplace=False
        )

        # splitting peaking and combinatoprial background
        dfp_pkbkg = dfp_bkg[
            (
                ( (~(dfp_bkg['aBplusMode']%10000).isin(gen_mode)) | ((dfp_bkg['B1_decayModeID']).isin(rec_mode)) )
                &
                ( (~(abs(dfp_bkg['aBminusMode'])%10000).isin(gen_mode)) | ((dfp_bkg['B0_decayModeID']).isin(rec_mode)) )
            )
            &
            (
                (dfp_bkg['pi4_B0_isSignal']==1) | (dfp_bkg['pi4_B1_isSignal']==1)
            ) 
        ]

        dfp_combkg = dfp_bkg.drop(
            dfp_bkg[
                (
                    ( (~(dfp_bkg['aBplusMode']%10000).isin(gen_mode)) | ((dfp_bkg['B1_decayModeID']).isin(rec_mode)) )
                    &
                    ( (~(abs(dfp_bkg['aBminusMode'])%10000).isin(gen_mode)) | ((dfp_bkg['B0_decayModeID']).isin(rec_mode)) )
                )
                &
                (
                    (dfp_bkg['pi4_B0_isSignal']==1) | (dfp_bkg['pi4_B1_isSignal']==1)
                ) 
            ].index
        )

        # signal
        dfp_sig = df_Bp[
            (
                ( (df_Bp['aBplusMode']%10000).isin(gen_mode) & (df_Bp['B1_decayModeID']).isin(rec_mode) ) 
                |
                ( (abs(df_Bp['aBminusMode'])%10000).isin(gen_mode) & (df_Bp['B0_decayModeID']).isin(rec_mode) )
            )
            &
            ( 
                (df_Bp['pi4_B0_isSignal']==1) | (df_Bp['pi4_B1_isSignal']==1) 
            ) 
        ]


        df = pd.concat([dfp_sig, dfp_pkbkg, dfp_combkg], keys=['signal', 'peaking background', 'combinatorial background'])
        
    
    elif y4s_id=='mixed':
        
        # Detalnu reconstructed modes
        l_rec_mode = [1400,1500,2400,2500]
        eta_rec_B0 = list(i+1 for i in l_rec_mode) + l_rec_mode
        # Detalnu generated modes
        eta_gen_B0 = [1017,1018,1035,1036]
        
        # Dpi0lnu reconstructed modes
        l_pi0_B0 = [1100, 1000, 2100, 2000]
        pi0_rec_B0 = list(i+1 for i in l_pi0_B0) + l_pi0_B0
        # Dpi0lnu generated modes
        pi0_gen_B0 = [1008,1010,1026,1028]
        
        if particle_name == 'eta':
            rec_mode = eta_rec_B0
            gen_mode = eta_gen_B0
        elif particle_name == 'pi0':
            rec_mode = pi0_rec_B0
            gen_mode = pi0_gen_B0

        
        # taking the events in which one B as Dlnu and the other one as Detalnu have been reconstructed
        df_B0 = dataset.loc['mixed'].query(
            f'((B0_decayModeID=={rec_mode} & B1_decayModeID<500) | (B1_decayModeID=={rec_mode} & B0_decayModeID<500))'
        )
        
        df0_bkg = df_B0.drop(
            df_B0[
                (
                    ( (df_B0['aB0Mode']%10000).isin(gen_mode) & (df_B0['B1_decayModeID']).isin(rec_mode) ) 
                    |
                    ( (abs(df_B0['aBbar0Mode'])%10000).isin(gen_mode) & (df_B0['B0_decayModeID']).isin(rec_mode) )
                )
                &
                ( 
                    (df_B0['pi4_B0_isSignal']==1) | (df_B0['pi4_B1_isSignal']==1) 
                ) 
            ].index, inplace=False
        )

        # splitting peaking and combinatoprial background
        df0_pkbkg = df0_bkg[
            (
                ( (~(df0_bkg['aB0Mode']%10000).isin(gen_mode)) | ((df0_bkg['B1_decayModeID']).isin(rec_mode)) )
                &
                ( (~(abs(df0_bkg['aBbar0Mode'])%10000).isin(gen_mode)) | ((df0_bkg['B0_decayModeID']).isin(rec_mode)) )
            )
            &
            (
                (df0_bkg['pi4_B0_isSignal']==1) | (df0_bkg['pi4_B1_isSignal']==1)
            ) 
        ]

        df0_combkg = df0_bkg.drop(
            df0_bkg[
                (
                    ( (~(df0_bkg['aB0Mode']%10000).isin(gen_mode)) | ((df0_bkg['B1_decayModeID']).isin(rec_mode)) )
                    &
                    ( (~(abs(df0_bkg['aBbar0Mode'])%10000).isin(gen_mode)) | ((df0_bkg['B0_decayModeID']).isin(rec_mode)) )
                )
                &
                (
                    (df0_bkg['pi4_B0_isSignal']==1) | (df0_bkg['pi4_B1_isSignal']==1)
                ) 
            ].index
        )

        # signal
        df0_sig = df_B0[
            (
                ( (df_B0['aB0Mode']%10000).isin(gen_mode) & (df_B0['B1_decayModeID']).isin(rec_mode) ) 
                |
                ( (abs(df_B0['aBbar0Mode'])%10000).isin(gen_mode) & (df_B0['B0_decayModeID']).isin(rec_mode) )
            )
            &
            ( 
                (df_B0['pi4_B0_isSignal']==1) | (df_B0['pi4_B1_isSignal']==1) 
            ) 
        ]


        df = pd.concat([df0_sig, df0_pkbkg, df0_combkg], keys=['signal', 'peaking background', 'combinatorial background'])

    return df, rec_mode, gen_mode
    
# ********************************************************************************************
    
    
# crystalball function (non_normalized)
def crystal_ball(params, x):
    
    N, a, n, mu, sig = params
    aa = abs(a)
    A = ((n/aa)**n) * exp(- aa**2 / 2)
    B = n/aa - aa
    condition1 = lambda x: N * exp(- ((x-mu)**2)/(2.*sig**2) )
    condition2 = lambda x: N * A * (B - (x-mu)/sig)**(-n)
    result = np.where((x - mu)/sig > (-a), condition1(x), condition2(x))

    return result


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
    return const * exp(-slope*(x)) # try x-0.5


expo2 = lambda x, c, s: c * exp(-s*x)


# polynomial function with 5 parameters
def polynomial(params, x):
    a0,a1,a2,a3,a4 = params
    return (a0 - a1*x - a2*x**2 - a3*x**3 - a4*x**4)

def crysexpo(params, x, crys_params=None, expo_params=None):
    p1, p2 = params
    return p1 * crystal_ball(crys_params, x) + p2 * expo(expo_params, x)

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


# Fitting
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






