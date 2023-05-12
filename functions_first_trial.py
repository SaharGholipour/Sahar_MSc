#!/usr/bin/env python
# coding: utf-8

# # All the defined functions I use in mysignal_comparison analysis


# In[1]:


# function of counting signal and background for both my signal and basf2 signal

def sig_bkg_counter(y4s_id,list_cut):
    n_sig, n_bkg, n_phgbkg= (np.zeros(len(list_cut)) for n_d in range(3))
    datas = []
    cut=0
    print('{:>13s}{:>14s}{:>10s}{:>15s}{:>15s}{:>15s}{:>10s}{:>20s}'.format('length','total_sigbkg','n_signal','n_background','significance','efficiency','p','peaking_bkg'))
    print('my \nisSignal')
    
    if y4s_id=='charged':
        for data in list_cut:
            ns, nb, n_pk = (0 for n in range(3))
            df_sig, df_bkg, df_pkgbkg = (pd.DataFrame() for n_df in range(3))
            for i in range(len(data)):
                if ( ((data['aBplusMode'].iloc[i])%10000 in [1019,1020,1039,1040]) | (abs(data['aBminusMode'].iloc[i])%10000 in [1019,1020,1039,1040]) ) and ( (data['pi4_B0_isSignal'].iloc[i]==1) | (data['pi4_B1_isSignal'].iloc[i]==1) ):
                    ns += 1
                    df_sig = df_sig.append(pd.DataFrame([data.iloc[i]],index=['signal'],columns=data.columns))
                else:
                    nb += 1
                    df_bkg = df_bkg.append(pd.DataFrame([data.iloc[i]],index=['background'],columns=data.columns))
                if ( ((data['aBplusMode'].iloc[i])%10000 not in [1019,1020,1039,1040]) & (abs(data['aBminusMode'].iloc[i])%10000 not in [1019,1020,1039,1040]) ) and ( (data['pi4_B0_isSignal'].iloc[i]==1) | (data['pi4_B1_isSignal'].iloc[i]==1) ):
                    df_pkgbkg = df_pkgbkg.append(pd.DataFrame([data.iloc[i]],index=['peaking background'],columns=data.columns))
                    n_pk += 1
            n_sig[cut] = ns
            n_bkg[cut] = nb
            n_phgbkg[cut] = n_pk
            datas.append(pd.concat([df_sig, df_bkg]))
            print('-'*116)
            print(f'cut{cut}:{len(data):6} {int(n_sig[cut]+n_bkg[cut]):10} {int(n_sig[cut]):12} {int(n_bkg[cut]):12} {n_sig[cut]/(pow((n_sig[cut]+n_bkg[cut]),0.5)):13.2f} {n_sig[cut]/(n_sig[0]):17.4f} {(n_sig[cut]/(n_sig[0]))/(1+pow(n_bkg[cut],0.5)):15.6f} {int(n_phgbkg[cut]):10}')
            cut += 1
        
    elif y4s_id=='mixed':
        for data in list_cut:
            ns, nb, n_pk = (0 for n in range(3))
            df_sig, df_bkg, df_pkgbkg = (pd.DataFrame() for n_df in range(3))
            for i in range(len(data)):
                if ( ((data['aB0Mode'].iloc[i])%10000 in [1017,1018,1035,1036]) | (abs(data['aBbar0Mode'].iloc[i])%10000 in [1017,1018,1035,1036]) ):
                    ns += 1
                    df_sig = df_sig.append(pd.DataFrame([data.iloc[i]],index=['signal'],columns=data.columns))
                else:
                    nb += 1
                    df_bkg = df_bkg.append(pd.DataFrame([data.iloc[i]],index=['background'],columns=data.columns))
                if ( ((data['aB0Mode'].iloc[i])%10000 not in [1017,1018,1035,1036]) & (abs(data['aBbar0Mode'].iloc[i])%10000 not in [1017,1018,1035,1036]) ) and ( (data['pi4_B0_isSignal'].iloc[i]==1) | (data['pi4_B1_isSignal'].iloc[i]==1) ):
                    df_pkgbkg = df_pkgbkg.append(pd.DataFrame([data.iloc[i]],index=['peaking background'],columns=data.columns))
                    n_pk += 1
            n_sig[cut] = ns
            n_bkg[cut] = nb
            n_phgbkg[cut] = n_pk
            datas.append(pd.concat([df_sig, df_bkg]))
            print('-'*116)
            print(f'cut{cut}:{len(data):6} {int(n_sig[cut]+n_bkg[cut]):10} {int(n_sig[cut]):12} {int(n_bkg[cut]):12} {n_sig[cut]/(pow((n_sig[cut]+n_bkg[cut]),0.5)):13.2f} {n_sig[cut]/(n_sig[0]):17.4f} {(n_sig[cut]/(n_sig[0]))/(1+pow(n_bkg[cut],0.5)):15.6f} {int(n_phgbkg[cut]):10}')
            cut += 1
    
    n_sig, n_bkg, n_phgbkg= (np.zeros(len(list_cut)) for n_d in range(3))
    cut=0
    print('*'*116)
    print('basf2 \nisSignal')
    for data in list_cut:
        ns, nb, n_pk = (0 for n in range(3))
        for i in range(len(data)):
            if data['isSignal'].iloc[i]==1:
                ns += 1
            else:
                nb += 1
            if (data['isSignal'].iloc[i]!=1) & ((data['pi4_B0_isSignal'].iloc[i]==1) | (data['pi4_B1_isSignal'].iloc[i]==1)):
                n_pk += 1 
        n_sig[cut] = ns
        n_bkg[cut] = nb
        n_phgbkg[cut] = n_pk
        print('-'*116)
        print(f'cut{cut}:{len(data):6} {int(n_sig[cut]+n_bkg[cut]):10} {int(n_sig[cut]):12} {int(n_bkg[cut]):12} {n_sig[cut]/(pow((n_sig[cut]+n_bkg[cut]),0.5)):13.2f} {n_sig[cut]/(n_sig[0]):17.4f} {(n_sig[cut]/(n_sig[0]))/(1+pow(n_bkg[cut],0.5)):15.6f} {int(n_phgbkg[cut]):10}')
        cut += 1
    
    Tree_cuts = pd.concat(datas, keys=[f'cut{i}' for i in range(len(list_cut))])
    return Tree_cuts
    


# In[10]:


# sum function of list elements
l = [1, 2, 3, 4, 5]
def sum_of_list(l):
  total = 0
  for val in l:
    total = total + val
  return total
sum_of_list(l[:])


# In[3]:


# get different ranges
def x_ranges(window_list, x_from_get_x_ratio, bin_edges_list):
    data = {}
    for width in window_list:
        ranges = []
        for i in range(len(x_from_get_x_ratio[width])):
            start_bin = x_from_get_x_ratio[width][i]
            end_bin = x_from_get_x_ratio[width][i] + width
            ranges.append(f'{bin_edges_list[start_bin]:.2f} - {bin_edges_list[end_bin]:.2f}')
        data[width] = ranges
    return data


# In[4]:


# the function of getting ratio in each window of histogram, and plot the significance
# Note: the histtype should be barstacked
def variable_best_range(width_list, variable_bin_count, variable_bin_edge):
#    print('{:>12}{:>25}{:>20}'.format('window size', 'highest significance', 'range of variable'), '\n', '~*'*29)
    ratio_full, significance_full, ranges_full= ({} for n_d in range(3))
    
    for width in width_list:
        ratio, significance, ranges = ([] for n_l in range(3))
        
        for i in range(len(variable_bin_count[0])):
            if i+width > len(variable_bin_count[0]) :
                break
            else:
                n_sig = sum_of_list(variable_bin_count[1][i:i+width]-variable_bin_count[0][i:i+width])
                n_bkg = sum_of_list(variable_bin_count[0][i:i+width])
                if (n_sig+n_bkg) < 200:
                    significance.append(np.nan)
                else:
                    if n_bkg==0:
                        pass
                    else:
#                         x_value.append((i+(i+width))/2)
#                         x_value.append(i)
                        ratio.append(n_sig/n_bkg)
                        significance.append(n_sig/pow((n_sig+n_bkg), 0.5))
                ranges.append(f'from {variable_bin_edge[i]:.2f} to {variable_bin_edge[i+width]:.2f}')

#        index = significance.index(np.nanmax(significance))
#        print(f'{width:8} {np.nanmax(significance):20.2f} {ranges[index]:>28}', '\n', '-'*58)
#         x_values_full[width] = x_value
        ranges_full[width] = ranges
        ratio_full[width] = ratio
        significance_full[width] = significance
#         show = plt.scatter(bin_edges[], significance_full[width], label=[f'{width}'])
#         plt.xlabel('x_value')
#         plt.ylabel('significance')
        
    return significance_full, ratio_full, ranges_full#, show



# print(x_values_full)
# print(x_values_full[12])


# x_values_full, ratio_full = get_x_ratio([10,11,12])
# plt.plot(x_values_full[10], ratio_full[10])

# for key, value in x_values_full.items():
#     print(f"width is {key}, xvalues is {value}")

# x_values_full


# In[5]:


# get longest element in the dictionary
def GetMaxFlow(flows):        
    maks=max(flows, key=lambda k: len(flows[k]))
    return len(flows[maks]), maks

dic = {'a':[1,7,3], 'b':[3,5,6]}
print(max(dic))
GetMaxFlow(dic)[1]


# In[6]:


# this function will make the length of all items of the dictionary the same and then plot them
# give the significance and ranges from the variable best range function to this function

def plot_variable_best_range(variable_name, width_list, significance_full, ranges_full, variable_bin_edge):
    # label including this form1 will have these properties
    form = {'family': 'helvetica', 'color': 'black', 'size': 20}
    
    data = []
    columns = ('highest significance', f'best range for {variable_name}')
    rows = ['%s bins' % width for width in width_list]
    n_columns = np.arange(len(columns)) + 0.3
    bar_width = 0.8
    
    for width in width_list:
        for i in range(GetMaxFlow(significance_full)[0]):
            if len(significance_full[width]) < GetMaxFlow(significance_full)[0]:
                significance_full[width].append(np.nan)
        # getting the index of the highest significance
        index = significance_full[width].index(np.nanmax(significance_full[width]))
    
        data.append([f'{np.nanmax(significance_full[width]):.2f}', ranges_full[width][index]])
        
        plt.plot(variable_bin_edge[:GetMaxFlow(significance_full)[0]], significance_full[width], 'o', label=[f'{width}'])
        plt.legend()
        
#    making the table bellow the plot
    table = plt.table(cellText=data,colLabels=columns, rowLabels=rows ,loc='bottom', cellLoc = 'center', rowLoc = 'center', bbox=[0.001, -0.9, 1, 0.8])
    table.set_fontsize(16)
    
    plt.xlabel(f'{variable_name}', fontdict=form1)
    plt.ylabel('significance', fontdict=form1)
    
