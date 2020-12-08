import Functions_parameters as parameters
import pandas as pd
import operator

import tensorflow as tf
from tensorflow import keras

from pickle import load
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


# Dictionary with the elements to be used and their atomic masses
atomic_mass_shortened = ['Li', 'Mg', 'Al', 'Si', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
                         'Ni', 'Co', 'Cu', 'Zn', 'Zr', 'Nb', 'Mo', 'Sn', 'Hf', 'Ta', 'W']


param_functions = {
    'VEC': parameters.FVEC,
    'Electronegativity_Difference': parameters.ElecDiff,
    'Atomic_Radius_Diff': parameters.AtmSizeDiff,
    'Mixing_Enthalpy': parameters.EMix,
    'Mixing_Entropy': parameters.Mixentropy,
    'E/A': parameters.FElecAtom,
    'EWF': parameters.FEWF,
    'Mod_Mismatch': parameters.FModMismatch,
    'DeltaG': parameters.deltaG,
    'ShearModG': parameters.FShearG,
    'Tm': parameters.FTm,
    'Ec': parameters.Ec}


def table_compositions(compositions):
    dict_elements = defaultdict(list)
    for i in compositions:
        curr_alloy = parameters.comp_dict(i)
        curr_alloy = parameters.atf_to_atp(curr_alloy)
        for j in atomic_mass_shortened:
            if j in curr_alloy:
                dict_elements[j].append(curr_alloy[j])
            else:
                dict_elements[j].append(0)
    return dict_elements


# Function to calculate the parameters of the HEA
def results(alloy):  
    #This function returns the parameters that are used to describe the HEA in a dictionary
    #Alloy is a dictionary created using parameters.comp_dict() and parameters.atf_to_atp(curr_alloy)
    dict_results = defaultdict(list)
    for i in param_functions:
        dict_results[i] = param_functions[i](alloy) 
    return dict_results


#Function to create a dict with a list of compositions and another list with parameters
def calcparameters(list_of_alloys):
    dict_comp_param_efraction = defaultdict(list)
    for i in list_of_alloys:
        curr_alloy = parameters.comp_dict(i)
        curr_alloy = parameters.atf_to_atp(curr_alloy)
        dict_comp_param_efraction['Composition'].append(parameters.format_comp(curr_alloy))
        params = results(curr_alloy)
        for j in params:
            dict_comp_param_efraction[j].append(params[j])          
    return dict_comp_param_efraction


#Function to create a datafrase with one-hot encoded conditions
def function_condition(condition):
    df_cond = pd.DataFrame(columns = ['Cond__AC','Cond__AM','Cond__HM','Cond__PM','Cond__WR'])
    for i in condition:
        cond_len = len(df_cond)
        if i == 'AC':
            df_cond.loc[cond_len] = [1,0,0,0,0]
        elif i == 'AM':
            df_cond.loc[cond_len] = [0,1,0,0,0]
        elif i == 'HM':
            df_cond.loc[cond_len] = [0,0,1,0,0]
        elif i == 'PM':
            df_cond.loc[cond_len] = [0,0,0,1,0]
        elif i == 'WR':
            df_cond.loc[cond_len] = [0,0,0,0,1] 
    return df_cond


def inputs_to_predict(alloy, condition):
    """Insert two strings, one for the alloy and another for the condition
    Example of alloy: Al0.3Co0.5CrFeNi
    Possible conditions: AC, AM, HM, WR, PM
    Returns: numpy array used in model.predict()
    """   
    # Alloy  and condition are strings, but they need to be lists
    i = [alloy]
    
    # Calculate the parameters, condition and elemental compositions
    part_one_parameters = pd.DataFrame(calcparameters(i))
    part_two_parameters = function_condition([condition])
    part_three_parameters = pd.DataFrame(table_compositions(i))
    
    # Merge the three pandas dataframes
    merge_one = part_one_parameters.merge(part_two_parameters, left_index=True, right_index=True)
    merge_two = merge_one.merge(part_three_parameters, left_index=True, right_index=True)
    
    # Column 'Composition' is not inserted as input, so drop it
    merge_two = merge_two.drop(['Composition'], axis=1)
    
    return merge_two.to_numpy()


def create_input_database(alloys, conditions, hardness):
    # Calculate the parameters, condition and elemental compositions
    part_one_parameters = pd.DataFrame(calcparameters(alloys))
    part_two_conditions = function_condition(conditions)
    part_three_composition = pd.DataFrame(table_compositions(alloys))
    
    # Merge the a,b and c pandas dataframes
    merge_one = part_one_parameters.merge(part_two_conditions, left_index=True, right_index=True)
    merge_two = merge_one.merge(part_three_composition, left_index=True, right_index=True)
    final_df = merge_two.join(hardness)
    
    return final_df


def easy_prediction(alloy, condition, scaler, model):
    """Insert two strings, one for the alloy and another for the condition
    Example of alloy: Al0.3Co0.5CrFeNi
    Possible conditions: AC, AM, HM, WR, PM
    Returns: prediction using tensorflow model
    """
    inp = inputs_to_predict(alloy, condition)
    scaled = scaler.transform(inp)
    prediction = model.predict(scaled)
    return prediction


# example of inputs to predict: a = inputs_to_predict('AlCoCrFeNi2.1', 'AC')
# example of create_inputs_database: b = create_input_database(compositions['Alloy'], compositions['Condition'], compositions['HV']) 