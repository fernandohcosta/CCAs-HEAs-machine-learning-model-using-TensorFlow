# AUTHOR: Fernando Henrique da Costa <fernando.henriquecosta@yahoo.com.br>
# DATE CREATED: 04/11/2020


import pandas as pd
import math
import numpy as np
import re
import os

DECIMALS = 3

scriptdir = os.path.dirname(os.path.abspath(__file__))

# Open database of enthalpy mixing
enthalpy_data = pd.read_csv(os.path.join(scriptdir, './Enthalpydata.csv'))

# Open the database with the elements' properties
eledata = pd.read_csv(os.path.join(scriptdir, './Elementdata.csv'))


# Dictionary with the elements and their atomic masses
atomic_mass = {'H': 1.00797, 'He': 4.0026, 'Li': 6.941, 'Be': 9.01218, 'B': 10.81,
               'C': 12.011, 'N': 14.0067, 'O': 15.9994, 'F': 18.998403, 'Ne': 20.179, 'Na': 22.98977,
               'Mg': 24.305, 'Al': 26.98154, 'Si': 28.0855, 'P': 30.97376, 'S': 32.06, 'Cl': 35.453,
               'K': 39.0983, 'Ar': 39.948, 'Ca': 40.08, 'Sc': 44.9559, 'Ti': 47.9, 'V': 50.9415,
               'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.847, 'Ni': 58.7, 'Co': 58.9332, 'Cu': 63.546,
               'Zn': 65.38, 'Ga': 69.72, 'Ge': 72.59, 'As': 74.9216, 'Se': 78.96,
               'Br': 79.904, 'Kr': 83.8, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.9059,
               'Zr': 91.22, 'Nb': 92.9064, 'Mo': 95.94, 'Tc': 98, 'Ru': 101.07,
               'Rh': 102.9055, 'Pd': 106.4, 'Ag': 107.868, 'Cd': 112.41, 'In': 114.82,
               'Sn': 118.69, 'Sb': 121.75, 'I': 126.9045, 'Te': 127.6, 'Xe': 131.3,
               'Cs': 132.9054, 'Ba': 137.33, 'La': 138.9055, 'Ce': 140.12, 'Pr': 140.9077,
               'Nd': 144.24, 'Pm': 145, 'Sm': 150.4, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.9254,
               'Dy': 162.5, 'Ho': 164.9304, 'Er': 167.26, 'Tm': 168.9342, 'Yb': 173.04,
               'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.85, 'Re': 186.207,
               'Os': 190.2, 'Ir': 192.22, 'Pt': 195.09, 'Au': 196.9665, 'Hg': 200.59,
               'Tl': 204.37, 'Pb': 207.2, 'Bi': 208.9804, 'Po': 209, 'At': 210,
               'Rn': 222, 'Fr': 223, 'Ra': 226.0254, 'Ac': 227.0278, 'Pa': 231.0359,
               'Th': 232.0381, 'Np': 237.0482, 'U': 238.029, 'Pu': 242, 'Am': 243,
               'Bk': 247, 'Cm': 247, 'No': 250, 'Cf': 251, 'Es': 252, 'Hs': 255,
               'Mt': 256, 'Fm': 257, 'Md': 258, 'Lr': 260, 'Rf': 261, 'Bh': 262,
               'Db': 262, 'Sg': 263}


# Function to divide the composition, inserted as a string, into a list using regular expressions
# lcc: list with the complete composition
def lcc(composition):
    return re.findall('[A-Z][a-z]?|[0-9]+\.?[0-9]+|\.[0-9]+|[0-9]+', composition)


# The comp_dict function converts the list created by the lcc function into a dictionary
# Format: {Element1: composition1, ...}
# comp_dict: composition dictionary
# The comp_dict function converts the list created by the lcc function into a dictionary
# Format: {Element1: composition1, ...}
# comp_dict: composition dictionary
def comp_dict(composition):
    # use lcc to parse the composition and transform it into a list
    completelist = lcc(composition)
    # od (organized dictionary) is the dictionary with the composition
    er = False
    erlist = []
    od = {}
    k = 0
    for i, j in enumerate(completelist):
        if j in atomic_mass:
            # Insert 1 in case no number is given. If the next element is a number, it will be changed
            od[j] = 1
            current_element = j
        elif re.search('[0-9]+\.?[0-9]+|\.[0-9]+|[0-9]+', j):
            od[current_element] = float(j)  
        else:
            print(f'{j} was not recognized. Check numbers and elements.')
    return od


# Normalize the composition in atomic fraction
def atf_to_atp(cpaf):
    Tmol = 0
    # cpap: composition atomic percent
    cpap = {}
    for i in cpaf:
        Tmol = Tmol + cpaf[i]
    for i in cpaf:
        cpap[i] = float(cpaf[i] / Tmol)
        if cpap[i] == 0:
            del cpap[i]
    return cpap


# Function to calculate the VEC of the alloy
def FVEC(Cp):
    VEC = 0
    for i in Cp:
        VEC = VEC + float(Cp[i]) * float(eledata.loc[eledata['Symbol'] == i, 'VEC'].item())
    return VEC


# Function to calculate the mixing entropy of the alloy
def Mixentropy(Cp):
    Sum = 0
    for i in Cp:
        Sum = Sum + float(Cp[i]) * np.log(float(Cp[i]))
    DeltaS = -8.3144621 * Sum
    return DeltaS


# Function to calculate the atomic size difference of the alloy
def AtmSizeDiff(Cp):
    Sum = 0
    rbar = 0
    for i in Cp:
        rbar = rbar + float(Cp[i]) * float(eledata.loc[eledata['Symbol'] == i,                                                                                                                      'Radius'].item())
    for i in Cp:
        Sum = Sum + float(Cp[i]) * np.power(
            (1 - float(eledata.loc[eledata['Symbol'] == i, 'Radius'].item()) / rbar), 2)
    ASD = 100 * np.sqrt(Sum)
    return ASD


# Function to calculate the electronegativity difference of the alloy
def ElecDiff(Cp):
    Sum = 0
    xbar = 0
    for i in Cp:
        xbar = xbar + float(Cp[i]) * float(eledata.loc[eledata['Symbol'] == i,                                                                                                                      'PaElec'].item())
    for i in Cp:
        Sum = Sum + float(Cp[i]) * np.power((float(eledata.loc[eledata['Symbol'] == i,                                                                                                                    'PaElec'].item()) - xbar), 2)
    ED = np.sqrt(Sum)
    return ED


# Function to return the mixing enthalpy between two elements E1 and E2
def EM(E1, E2):
    Em = enthalpy_data[E1][(enthalpy_data['Symbol'] == E2)].values[0]
    if math.isnan(Em):
        Em = enthalpy_data[E2][(enthalpy_data['Symbol'] == E1)].values[0]
    return Em


# Function to calculate the mixing enthalpy of the alloy
def EMix(Cp):
    k = 0
    Emix = 0
    Ele = list(Cp.keys())
    for i in range(len(Ele) - 1):
        for j in range(len(Ele) - 1 - k):
            Emix = Emix + 4 * float(EM(Ele[i], Ele[j + 1 + k])) * float(Cp[Ele[i]]) * float(Cp[Ele[j + 1 + k]])
        k = k + 1
    return Emix


# Format the composition to display the result
def format_comp(cp):
    formated = ''
    for i in cp:
        # formated += i+str(cp[i])+' '
        formated += i + "{:.{precision}f}".format(cp[i], precision=DECIMALS) + ' '
    return formated


#Function to calculate the e/a of the alloy
def FElecAtom(Cp):
    ea = 0
    for i in Cp:
        ea = ea + float(Cp[i])*float(eledata.loc[eledata['Symbol'] == i, 'e/a'].item())
    return ea


#Function to calculate the electron work function of the alloy
def FEWF(Cp):
    w = 0
    for i in Cp:
        w = w + float(Cp[i])*float(eledata.loc[eledata['Symbol'] == i, 'EWF'].item())
    return pow(w, 6)


#Function to calculate the shear modulus G of the alloy
def FShearG(Cp):
    G = 0
    for i in Cp:
        G = G + float(Cp[i])*float(eledata.loc[eledata['Symbol'] == i,                                                                                                                   'Shear_M'].item())
    return G


#Function to calculate the modulus mismatch ni from G of the alloy
def FModMismatch(Cp):
    G = FShearG(Cp)
    ni = 0
    for i in Cp:
        Gi = float(eledata.loc[eledata['Symbol'] == i, 'Shear_M'].item())
        Ci = float(Cp[i])
        ni = ni + (Ci*(2*(Gi-G))/(Gi+G))/(1+0.5*abs(Ci*(2*(Gi-G))/(Gi+G)))
    return ni


#Function to calculate the shear modulus difference of the alloy
def deltaG(Cp):
    G = FShearG(Cp)
    dG = 0
    for i in Cp:
        dG = dG + float(Cp[i])*np.power((1-float(eledata.loc[eledata['Symbol'] == i,                                                                                                                    'Shear_M'].item())/G), 2)
    return dG


#Function to calculate the Md of the alloy
def FMd(Cp):
    Md = 0
    for i in Cp:
        Md = Md + float(Cp[i])*float(eledata.loc[eledata['Symbol'] == i, 'Md'].item())
    return Md


#Function to calculate the melting temperature Tm of the alloy
def FTm(Cp):
    Tm = 0
    for i in Cp:
        Tm = Tm + float(Cp[i])*float(eledata.loc[eledata['Symbol'] == i, 'Tm'].item())
    return Tm


#Function to calculate the cohesive energy of the alloy
def Ec(Cp):
    E = 0
    for i in Cp:
        E = E + float(Cp[i])*float(eledata.loc[eledata['Symbol'] == i, 'Ecohesive'].item())
    return E