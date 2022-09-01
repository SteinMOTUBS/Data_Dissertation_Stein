#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
import datetime
import os 
import sys
import configparser
from scipy.optimize import curve_fit
from datetime import timedelta
import statsmodels.api as sm


def asym_percentile(data,p_value):
    '''
    Calculation of the asymetric interquantile ranges
    This code was developed in cooperation with 
    Thomas Martin Dutschmann
    inputs: 
        data: data array as list
        p_value: definition of the percentile
    return:
        x_low: begin of the percentile as float 
        x_high: end of the percentile as float 
        p_low: begin relative rank as float  {0:1}
        p_high: end relative rank as float  {0:1}
        x_delta: range between x_low and x_high as float
        x_mean: mean of data as float
        x_median: median of data as float
        x_05: central value of data as float
        x_variance: variance of data as float
        x_SD: standard deviation of data as float
    Ref:
        'Evaluation of measurement dataSupplement 1 to the Guide to the expression of
        uncertainty in measurement Propagation of distributions using a Monte Carlo method'
    '''
    x_values = np.sort(data)
    perc = np.arange(1,len(data)+1)/len(data)
    Dict_Fit = dict(zip(perc, x_values))
    Dict_x = dict(zip(x_values,perc))
    Delta=[]
    p=p_value
    p_dict = int(p*len(Dict_Fit))
    keys = list(Dict_Fit.keys())
    for i in range(len(Dict_Fit)-p_dict):
        Delta.append(Dict_Fit[keys[i+p_dict]]-Dict_Fit[keys[i]])
    low = Delta.index(np.amin(Delta))
    high = low + p_dict
    x_low = x_values[low]
    x_high = x_values[high]
    p_low = perc[low]
    p_high = perc[high]
    x_delta = x_high - x_low
    x_mean = np.mean(x_values)
    x_median = np.median(x_values)
    x_variance = np.var(x_values)
    x_SD = np.sqrt(x_variance)
    if len(x_values) % 2 == 0:
        x_05 = (x_values[int(len(x_values)/2 -1)] + x_values[int(len(x_values)/2 )])/2
    else:
        x_05 = x_values[int(len(x_values)/2 -0.5)]  
    
    return x_low, x_high, p_low, p_high,x_delta, x_mean, x_median, x_05,x_variance,x_SD

def ECDF_full_data(Fit,p_value):
    '''
    Calculation of the asymetric interquantile ranges
    This code was developed in cooperation with 
    Thomas Martin Dutschmann
    inputs: 
        Fit: data array as list
        p_value: definition of the percentile
    return:
        x_low: begin of the percentile as float 
        x_high: end of the percentile as float 
        inter_quantile_range: range between x_low and x_high as float
        x_mean: mean of data as float
        x_median: median of data as float
        x_variance: variance of data as float
        x_SD: standard deviation of data as float
        x_rel_AIQR: normalized (to mean) interquantile range as float
    Ref:
        'Evaluation of measurement dataSupplement 1 to the Guide to the expression of
        uncertainty in measurement Propagation of distributions using a Monte Carlo method'
    '''
    p=p_value
    X = ecdf_self(Fit)[0]
    Y = ecdf_self(Fit)[1]
    asym_data = asym_percentile(Fit,p)
    p_low_Fit_1 = asym_data[2]
    p_high_Fit_1 = asym_data[3]
    x_low_Fit_1 = asym_data[0]
    x_high_Fit_1 = asym_data[1]
    x_mean =  asym_data[5]
    x_median =  asym_data[6]
    x_mode =  asym_data[7]
    inter_quantile_range =  asym_data[4]
    x_mode =  asym_data[5]
    x_min = np.amin(ecdf_self(Fit)[0])
    x_max = np.amax(ecdf_self(Fit)[0])
    y_min = np.amin(ecdf_self(Fit)[1])
    y_max = np.amax(ecdf_self(Fit)[1])
    x_variance =  asym_data[8]
    x_SD = asym_data[9]
    x_CV = x_SD/x_mean
    x_rel_AIQR = inter_quantile_range/x_mean
    
    return inter_quantile_range, x_mean, x_median, x_variance, x_SD, x_CV, x_rel_AIQR

def ecdf_self(list_of_data):
    '''
    Empirical cumulative density function
    Inputs:
        list_of_data: data array
    Return:
        [0] x_values: sorted values of the input
        [1] perc: percentage of the values
    '''
    x_values = np.sort(list_of_data)
    perc = np.arange(1,len(list_of_data)+1)/len(list_of_data)
    
    return x_values, perc

def ms_ACE_no_eta(c,KD,uc,uf):
    '''
    Implementation of the ms-ACE model function
    without viscosity correction based on KA
    inputs:
        c: ligand concentration as float
        KD: dissociation constant as float c and KD must be expressed in
                the same unit e.g. mol/l
        uf: mobility at concentration = 0 as float
        uc: mobility at concentration = infinity as float
    return:
        ui:mobility at concentration = c as float
    '''
    KA = 1/KD
    u_i = (uf + KA * uc *c)/(1+KA*c)
    
    return u_i

def create_c_in(KD,c_min,c_max,no_c,rep,zero_min = -1):
    '''
    Definition of a concentration set
    Inputs:
        KD: dissociation constant as float
        c_min : minimal concentration as pos . as float
        c_max: maximal concentration as multiples of KD as float
        no_c : number of different cocentrations as int
        rep : number of replicated concentrations as int
    zero_min : default minimum if the real c_min is 0
    return:
        set of concentrations as numpy array
    '''
    log_max = np.log10(c_max * KD)
    zero_in = []
    no_c_in = no_c
    if c_min ==  0:
        log_min = zero_min
        no_c_in -=1
        zero_in = [0]
    else:
        log_min = 0
    c_in = np.concatenate([np.concatenate([zero_in,np.logspace(log_min,log_max,no_c_in)]) for __ in range(rep)])
    c_in.sort()
    return c_in

def variance_function_ms_ACE(c,beta0,beta1):
    '''
    Calculation of the mobility variance as a function of c.
    Inputs:
        c: concentration  as float
        beta0,beta1: empirical parameter  as floats
    return:
       prediced variance as float
    '''
    return beta0 *np.exp(beta1 * c)

def add_zeros(index_begin,max_str_length):
    '''
    Helper function to create file names
    '''
    index_begin_name = str(index_begin)
    add_zeros = max_str_length - len(str(index_begin))
    for i in range(add_zeros):
        index_begin_name = '0'+index_begin_name
    return index_begin_name

'''
DOE full factorial
DS,PS ms-ACE


'''

KD_list = [10,20,30,40,50]
uc_list = [-10,-4,2,8,14]
uf_list = [-1,0,1]

c_min_list = [0,0.5,1]
c_max_list = [0.5,1,2.5,5,10]

no_c_list = [6,8,12]
rep_list = [1,2]

MU_type_list = ['abs','rel_pos_prop_c']
MU_specs_no_list = [0,1,2]
MU_specs_abs = [[0.0025,0],[0.001,0],[0.04,0]]
MU_specs_rel = [[0.00577,0.011],[0.00577,0.012],[0.00577,0.013]]



DS_PS_dict = {'KD':[],'uc':[],'uf':[],'c_min':[],'c_max':[],'no_c':[],'rep':[],'MU_type':[],'MU_specs':[]}
key_list = list(DS_PS_dict.keys())

for c_max in c_max_list:
    for c_min in c_min_list:
        if not c_min < c_max:
            break
        for KD in KD_list:
            for uc in uc_list:
                for uf in uf_list:
                    for no_c in no_c_list:
                        for rep in rep_list:
                            for MU_type in MU_type_list:
                                for MU_no in MU_specs_no_list:
                                    if MU_type == 'abs':
                                        MU_spec = MU_specs_abs[MU_no]
                                    else:
                                        MU_spec = MU_specs_rel[MU_no]
                                        
                                    list_of_parameter = [KD,uc,uf,c_min,c_max,no_c,rep,MU_type,MU_spec]
                                    for i in range(len(list_of_parameter)):
                                        name = key_list[i]
                                        parameter = list_of_parameter[i]
                                        DS_PS_dict[name].append(parameter)
                                
                    
                    

DS_PS_DF = pd.DataFrame(DS_PS_dict)

for key in DS_PS_DF.select_dtypes('number').keys():
    
    data_array = DS_PS_DF[key]
    min_value = data_array.min()
    max_value = data_array.max()
    range_value = max_value - min_value

    new_name = key + '_norm'
    norm = [(data_array[i] - min_value)/range_value for i in range(len(data_array))]
    DS_PS_DF[new_name] = norm


abs_value_min = np.min(MU_specs_abs)
abs_value_max = np.max(MU_specs_abs)
abs_range = abs_value_max - abs_value_min

rel_1_list = [MU_specs_rel[j][0] for j in range(len(MU_specs_rel))]
rel_1_list = np.unique(rel_1_list)

rel_2_list = [MU_specs_rel[j][1] for j in range(len(MU_specs_rel))]
rel_2_list = np.unique(rel_2_list)

rel1_value_min = np.min(rel_1_list)
rel1_value_max = np.max(rel_1_list)
rel1_range = rel1_value_max - rel1_value_min

rel2_value_min = np.min(rel_2_list)
rel2_value_max = np.max(rel_2_list)
rel2_range = rel2_value_max - rel2_value_min
        

MU_type_norm = []
MU_spec_norm = []
for i in range(len(DS_PS_DF)):
    
    MU_type = DS_PS_DF['MU_type'][i]
    MU_specs = DS_PS_DF['MU_specs'][i]
    
    if MU_type == 'abs':    
        MU_type_norm.append(1)
        if len(MU_specs_abs) > 1:
            MU_spec_norm.append((MU_specs - abs_value_min) / abs_range)
        else:
            MU_spec_norm.append(1)
    else:
        MU_type_norm.append(2)
        if len(rel_1_list) > 1:
            rel_norm_1 = (MU_specs[0] - rel1_value_min) / rel1_range
        else:
            rel_norm_1 = 1
        if len(rel_2_list) > 1:
            rel_norm_2 = (MU_specs[1] - rel2_value_min) / rel2_range
        else:
            rel_norm_2 = 1
        
        MU_spec_norm.append([rel_norm_1,rel_norm_2])
    
    
DS_PS_DF['MU_type_norm'] = MU_type_norm
DS_PS_DF['MU_specs_norm'] = MU_spec_norm


config = configparser.ConfigParser()

Parameter_file = sys.argv[1]

config.read(Parameter_file)

if Parameter_file == '-f':
    
    Parameter_file = 'ms_ACE_MC_SIM_Index_specs_cluster_default.ini'
    

config.read(Parameter_file)

index_begin = int(config['Index']['start'])
index_end = int(config['Index']['end'])

no_runs = int(config['runs']['runs'])

exclude = int(config['exclude']['exclude'])
exclude_abs = int(config['exclude']['exclude_abs'])
excl_low = 1/exclude 

p_value = float(config['p_value']['p_value'])

random_index_on = bool(int(config['Index']['random']))
random_no = int(config['Index']['random_no'])


if index_end >= len(DS_PS_DF):
    index_end = len(DS_PS_DF)

max_str_length = len(str(len(DS_PS_DF)))

index_begin_name = add_zeros(index_begin,max_str_length)
index_end_name = add_zeros(index_end,max_str_length)

Index_list =  range(index_begin,index_end+1)

now = datetime.datetime.now()
timestamp = str(now)[0:4] + str(now)[5:7] + str(now)[8:10] 

complete_index_list_reprod_shuffle = np.arange(len(DS_PS_DF))

rdm_suffle_seed = 12151989
np.random.seed(rdm_suffle_seed)
np.random.shuffle(complete_index_list_reprod_shuffle)

Index_list = complete_index_list_reprod_shuffle[index_begin:index_end+1]

if random_index_on:
    Index_list =np.random.choice(complete_index_list_reprod_shuffle,size=random_no, replace=False)
    Index_list =np.random.choice(complete_index_list_reprod_shuffle,size=None, replace=False)
    index_begin_name ='rdm'
    index_end_name ='rdm'
    
if Index_list.ndim == 0:
    Index_list = [Index_list]
    
outname = f'ms_ACE_MC_SIM_{index_begin_name}_to_{index_end_name}_{timestamp}'


outname_valid = False
file_counter = 1
while not outname_valid:
    if f'{outname}.pkl' in os.listdir():
        outname += f'_{file_counter}'
        file_counter += 1
    else: 
        outname_valid = True




print()
print('--------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------')
print()
print('MC Simulation ')
print(f'Parameter file: {Parameter_file}')
print(f'from {index_begin} to {index_end} with {no_runs} no of runs')
print()
print(f'Filename: {outname}')
print()
print('--------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------')

out_pkl = f'{outname}.pkl'
out_csv = f'{outname}.csv'

MC_results_dict = {}

basic_key_list = ['Parameter no','time','duration','rel duration','Seed no', 'no runs','exclude parameter','p value','parameter file','outname']
additional_keys = ['MRR','nMRR','SD_KD','DPP','DPR','NoDP']
opt_keys = ['D opt', 'E opt', 'A opt']

result_keys = ['mean', 'Var','SD','CV', 'AIQR','rel AIQR','Trueness','rel Trueness']
result_key_a = ['KD','uc','uf']

additinal_results = ['abs fails','rel fails']

for key in basic_key_list:
    MC_results_dict[key] = []                 

for key in DS_PS_DF.keys():
    MC_results_dict[key] = []

for add_key in additional_keys:
    MC_results_dict[add_key] = []

for key_a in result_key_a:
    for key_b in result_keys:
        key = key_a + ' ' + key_b
        MC_results_dict[key] = []

for add_key in additinal_results:
    MC_results_dict[add_key] = []


MC_results_dict_prev = MC_results_dict.copy()

duration_total = datetime.timedelta()

for i in Index_list:
    
    now = datetime.datetime.now()
    timestamp = str(now)[0:4] + str(now)[5:7] + str(now)[8:10] + '_'+ str(now)[11:13] + '_' +  str(now)[14:16]
    
    i_in = np.where(Index_list==i)
    i_in = int(i_in[0])+1
    to_go = len(Index_list) - i_in

    print()
    print('-----------------------------------------------')
    print(f'number {i_in} of {len(Index_list)}')
    print()
    print(f'This is row no.: {i}')
    print(f'start time: {now}')
    print()

    parameter_import = DS_PS_DF.iloc[i]

    Seed_number = np.random.randint(10**6)
    np.random.seed(Seed_number)


    
    MC_results_dict['Parameter no'].append(i)
    MC_results_dict['time'].append(timestamp)
    MC_results_dict['Seed no'].append(Seed_number)
    MC_results_dict['no runs'].append(no_runs)
    MC_results_dict['exclude parameter'].append([exclude,exclude_abs])
    MC_results_dict['p value'].append(p_value)
    MC_results_dict['parameter file'].append(Parameter_file)
    MC_results_dict['outname'].append(outname)

    parameter_list = []

    for key in parameter_import.keys():
        MC_results_dict[key].append(parameter_import[key])

    KD, uc, uf, c_min, c_max, no_c, rep, MU_type, MU_specs = parameter_import[0:9]
    
    
    print(f'KD: {KD}, uc: {uc}, uf:{uf}')
    print(f'cmin: {c_min}, cmax: {c_max}, no c: {no_c}, rep: {rep}')
    print(f'MU type: {MU_type}, MU specs:{MU_specs}')
    print()

    c_in = create_c_in(KD,c_min,c_max,no_c,rep,zero_min = -1)

    E_mu_eff_in_list = np.array([ms_ACE_no_eta(c,KD,uc,uf) for c in c_in] )


    E_SD_list = [np.sqrt(variance_function_ms_ACE(c,MU_specs[0],MU_specs[1])) for c in c_in]
    SD_KD = np.sqrt(variance_function_ms_ACE(KD,MU_specs[0],MU_specs[1]))

    NoDP = no_c * rep
    MRR = abs(uc-uf)
    nMRR = MRR/SD_KD
    c_in_min = np.min(c_in)
    alpha_min = c_in_min/(KD+c_in_min)
    c_in_max = np.max(c_in)
    alpha_max = c_in_max/(KD+c_in_max)
    DPP = np.mean([alpha_min,alpha_max])
    DPR = alpha_max - alpha_min

    additional_parameter_list = [MRR,nMRR,SD_KD,DPP,DPR,NoDP]

    for key_i in range(len(additional_keys)):
        key = additional_keys[key_i]
        add = additional_parameter_list[key_i]
        MC_results_dict[key].append(add)

    Simulation_dict = {'KD':[],'uc':[],'uf':[]}

    no_fails = 0
    for run in range(no_runs):

        mu_rdm_list = np.array([np.random.normal(mean_in,SD_in,size=None) for mean_in,SD_in in zip(E_mu_eff_in_list,E_SD_list)])
        
        KD_guess = np.mean(c_in)
        uf_guess = mu_rdm_list[0]
        uc_guess = mu_rdm_list[-1]
        p0_in = [KD_guess, uc_guess, uf_guess]
        
        try:
            popt, pcov = curve_fit(ms_ACE_no_eta, c_in, mu_rdm_list,p0=p0_in)

        except:
            popt =  ['f','f','f']
            no_fails += 1

        if not popt[0] == 'f':
            KD_excl = popt[0]/KD < excl_low or popt[0]/KD > exclude
            uc_excl = popt[1] < uc - exclude_abs or popt[1] > uc + exclude_abs
            uf_excl = popt[2] < uf - exclude_abs or popt[2] > uf + exclude_abs
            if KD_excl or uc_excl or uf_excl:
                popt =  ['f','f','f']
                no_fails += 1

        if not popt[0] == 'f': 
            for j in range(len(Simulation_dict.keys())):
                key = list(Simulation_dict.keys())[j]
                Simulation_dict[key].append(popt[j])

    for j in range(len(Simulation_dict.keys())):
        key = list(Simulation_dict.keys())[j]
        Data = Simulation_dict[key]
        if len(Data)/no_runs > 0.1:
            inter_quantile_range, x_mean, x_median, x_variance, x_SD, x_CV, x_rel_AIQR = ECDF_full_data(Data,p_value)
            E_BP = parameter_import[key]
            Trueness = x_mean - E_BP
            Trueness_rel ='BP: 0'
            if not E_BP == 0:
                Truness_rel = abs(Trueness)/E_BP
            summary_list = [x_mean,x_variance,x_SD, x_CV,inter_quantile_range,x_rel_AIQR,Trueness,Truness_rel]
        else:
            summary_list = ['not evaluable' for __ in range(len(result_keys))]
        
        for q in range(len(result_keys)):
            result_key_in = result_keys[q]
            end_key = key + ' ' + result_key_in
            result = summary_list[q]
            MC_results_dict[end_key].append(result)
            
            
    MC_results_dict['abs fails'].append(no_fails)
    MC_results_dict['rel fails'].append(no_fails/no_runs)
    now2 = datetime.datetime.now()
    duration = now2 - now
    duration_total += duration
    duration_mean = duration_total/i_in
    MC_results_dict['duration'].append(duration.total_seconds())
    run_per_sec = int(no_runs/duration.total_seconds())
    MC_results_dict['rel duration'].append(run_per_sec)

    Average_duration = np.mean(MC_results_dict['duration'])
    Average_runs_per_sec = int(np.mean(MC_results_dict['rel duration']))
       
    now_loop = datetime.datetime.now() 
    time_to_go = to_go * duration_mean
    approximated_end = now_loop + time_to_go
    
    try:
        MC_results_DF = pd.DataFrame(MC_results_dict)
        MC_results_dict_prev = MC_results_dict.copy()
    except:
        print('-------')
        print(f'Line: {i} was not succesful!')
        print('-------')
        MC_results_dict = MC_results_dict_prev.copy()
        for key in list(MC_results_dict.keys()):
            if key == 'Parameter no':
                add = i
            else:
                add = 'fail' 
            MC_results_dict[key].append(add)
        MC_results_dict_prev = MC_results_dict.copy()
    

    print(f'It took {duration}. Average duration: {Average_duration} seconds.')
    print(f'Runs per second: {run_per_sec}. Average: {Average_runs_per_sec}.')
    print()
    print(f'Total time: {duration_total}, approximated end: {approximated_end}')
    print()
    print('-----------------------------------------------')
    print()
        
    
MC_results_DF = pd.DataFrame(MC_results_dict)

MC_results_DF.to_pickle(out_pkl)
MC_results_DF.to_csv(out_csv)

