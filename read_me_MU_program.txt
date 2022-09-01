This application is a part of the dissertaion
'Evaluation of the Measurement Uncertainty in Ligand Binding Assays
with Focus on Mobility Shift Affinity Capillary Electrophoresis'.

It can be employed to predict the measurement uncertainty of ms-ACE measurements.
Author: Matthias Oliver Stein
Mail I: matthias.stein@tu-braunschweig.de
Mail II: stein.matthiasoliver@gmail.de

Date 2022/03/18

Version: 0.1


Inputs:

[No. runs]: number of Monte Carlo simulation cycles as int

[exclude]: relative threshold for valid KD values as float
[excl_low]: inverse of exclude as float
Exclude if KD/E(KD) > exclude or KD/E(KD) < excl_low
[exclude_abs]: threshold for valid uc,uf values as float
Exclude if abs(uc or uf) 
uf < E(uf) - exclude_abs or uf > E(uf) + exclude_abs

[Percentile]: Percentile of AIQR

[KD]: dissociation constant as float
[uf]: mobility at concentration = 0 as float
[uc]: mobility at concentration = infinity as float

[c_min]: minimal concentration as pos. float
[c_max]: maximal concentration as multiples of KD as pos. float
[no_c]: number of different cocentrations as int
[rep]: number of replicated concentrations as int

[MU type]: homo- or heteroscedastic uncertainty
'Abs': homoscedastic, 'Rel': uncertainty relative to c
[MU spec a]: beta_0 of variance function
[MU spec b]: beta_1 of variance function
variance function: var(c) = beta_0 + exp(beta_1 *c)
Thus; beta_0 = variance if beta_1 = 0

[c_type]: defines the type of concentration set 
possible inputs : 'Log_Lin', 'QUAD', 'INV_Lin',
'D', 'E', 'A', '110', '70', 'manual'

[inner_bias]: Bias of every c as N(0,innerbias)
[inner_bias]: Bias of every MC Sim as N(0,innerbias)
Thus input in SD units


[manual c]: list of int if c_type = 'manual'
Direct definition of measurement concentrations

[data_name]: definition of file name

BUTTONS:
blue arrow next to inputs:
IMPORTANT inputs must be transfered to active values!

Only c Set: calculates only the set of concentrations. 
Can be used to calculate optimal designs.

Reset Parameter: resets all parameters to default
Resets the whole Program

Start: starts MC simulation

Exit: close program





