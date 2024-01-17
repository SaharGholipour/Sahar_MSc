def sig_bkg_spliter(dataset, gen=None):
    
    # charged Detalnu reconstructed modes
    l_eta_Bp = [1600,1700,2600,2700]
    eta_rec_Bp = list(i+1 for i in l_eta_Bp) + l_eta_Bp
    
    # generated modes
    # charged Detalnu
    eta_gen_Bp = [1003, 1004, 1005, 1006, 1023, 1024, 1025, 1026]
    
    # mcPDG of D2s
    D2s_mcPDG_Bp = [10421, 423, 10423, 20423, 425]
    
    # gen = gen
    
    if gen == 'charged':
        particle_variable = 'aBplusMode'
        antiparticle_variable = 'aBminusMode'
        D2s_gen_mode = eta_gen_Bp
        
    if gen == 'mixed':
        particle_variable = 'aB0Mode'
        antiparticle_variable = 'aBbar0Mode'
        D2s_gen_mode = eta_gen_B0
    
    # signal
    eta_df_sig = dataset[
    ( 
     ((abs(dataset[antiparticle_variable])%10000).isin(D2s_gen_mode)) & ((dataset['B0_decayModeID']).isin(eta_rec_Bp)) & (dataset['pi4_B0_isSignal']==1) & (dataset['pi4_B0_genMotherPDG_0'].isin(D2s_mcPDG_Bp))
    )
    
    |
    
    (
     (((dataset[particle_variable])%10000).isin(D2s_gen_mode)) & ((dataset['B1_decayModeID']).isin(eta_rec_Bp)) & (dataset['pi4_B1_isSignal']==1) & (abs(dataset['pi4_B1_genMotherPDG_0']).isin(D2s_mcPDG_Bp))
    )
    ]
    # background
    eta_df_bkg = dataset.drop(eta_df_sig.index, inplace=False)
    

    
    return eta_df_sig, eta_df_bkg