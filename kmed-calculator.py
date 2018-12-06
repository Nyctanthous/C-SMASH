import os
import numpy as np
import pandas as pd
from analysis import *
import pickle
import gc


import random
import time

from sklearn.metrics.pairwise import euclidean_distances
import kmed

NUM_CORES = 8
SUBSET_SIZE = 13000#146600

print("Staring script...")
time.sleep(1)
# ---------------------------------------------------------------------------------------------
sdss_df = import_sdss ("%s/Databases/SDSSMOC4/data/sdssmocadr4.tab" % os.getcwd())

# Drop information not relevant to this problem, along with the SMOC_ID, which just indicates
# an SDSS observation; one asteroid can have multiple SMOC_ID, so it isn't a useful identifier.
sdss_df.drop(labels=['OBJ_ID_RUN', 'OBJ_ID_COL', 'OBJ_ID_FIELD',
                     'OBJ_ID_OBJ', 'ROWC', 'COLC', 'JD_ZERO', 'RA',
                     'DEC', 'LAMBDA', 'BETA', 'PHI', 'VMU', 'VMU_ERROR',
                     'VNU', 'VNU_ERROR', 'VLAMBDA', 'VBETA', 'IDFLAG',
                     'RA_COMPUTED', 'DEC_COMPUTED', 'V_MAG_COMPUTED',
                     'R_DIST', 'G_DIST', 'OSC_CAT_ID', 'ARC',
                     'EPOCH_OSC', 'A_OSC', 'E_OSC', 'I_OSC', 'LON_OSC',
                     'AP_OSC', 'M_OSC', 'PROP_CAT_ID', 'A_PROP',
                     'E_PROP', 'SIN_I_PROP', 'B_MAG', 'H', 'G',
                     'A_MAG', 'A_ERR', 'SMOC_ID', 'PHASE', 'D_COUNTER', 
                     'TOTAL_D_COUNT'],
                     axis=1, inplace=True)

print("Loaded the SDSS")
time.sleep(1)

# ---------------------------------------------------------------------------------------------
tax_df = import_tax("%s/Databases/SDSSTaxonomy/data/sdsstax_ast_table.tab" % os.getcwd())
tax_df.drop(labels=['AST_NAME', 'SCORE', 'NCLASS', 'METHOD', 'BAD',
                    'SEQUENCE', 'PROPER_SEMIMAJOR_AXIS',
                    'PROPER_ECCENTRICITY',
                    'SINE_OF_PROPER_INCLINATION',
                    'OSC_SEMIMAJOR_AXIS', 'OSC_ECCENTRICITY',
                    'OSC_INCLINATION'],
                     axis=1, inplace=True)

merged = merge(sdss_df, tax_df, ["PROV_ID", "AST_NUMBER"])
merged.reset_index(drop=True, inplace=True)

print("Loaded the Tax.")
time.sleep(1)

del tax_df, sdss_df
gc.collect()
time.sleep(1)
print("Ran GC")

# ---------------------------------------------------------------------------------------------
def grad(c1, c2, c1_base, c2_base):
    return -0.4*((c2 - c1) / (c2_base - c1_base))

def reflect(x, y):
    return -2.5*(np.log10(x) - np.log10(y))

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

merged["COLOR_U"] = reflect(merged["U_MAG"], merged["V_MAG"])
merged["COLOR_G"] = reflect(merged["G_MAG"], merged["V_MAG"])
merged["COLOR_R"] = reflect(merged["R_MAG"], merged["V_MAG"])
merged["COLOR_I"] = reflect(merged["I_MAG"], merged["V_MAG"])
merged["COLOR_Z"] = reflect(merged["Z_MAG"], merged["V_MAG"])

merged["G_REFL"] = grad(merged["COLOR_G"], merged["COLOR_U"], 0.354, 0.477)
merged["R_REFL"] = grad(merged["COLOR_U"], merged["COLOR_R"], 0.477, 0.623)
merged["I_REFL"] = grad(merged["COLOR_R"], merged["COLOR_I"], 0.623, 0.763)
merged["Z_REFL"] = grad(merged["COLOR_I"], merged["COLOR_Z"], 0.763, 0.913)

wave_mags = ["G_REFL", "R_REFL", "I_REFL", "Z_REFL"]


training_data = merged[wave_mags].values

print("Training data created.")
time.sleep(1)

# ---------------------------------------------------------------------------------------------
# Pick random indices to sample from. Has no repitition.
arr_size = len(merged) - 1

#index_dict = {}
#while len(index_dict.keys()) < 50000:
#    index_dict[random.randint(0, arr_size)] = 0

#random_sample = np.array([training_data[idx] for idx in index_dict.keys()])
random_sample = training_data

print("Random sample created")
time.sleep(1)

# ---------------------------------------------------------------------------------------------
#D = dense_euclidean_distances(training_data, training_data)
D = kmed.euclidean_distances_slow(random_sample, random_sample)
#D = pdist(training_data, 'euclidean')

print("Pairwise created, shape is ", D.shape)
time.sleep(1)

mediods, classifications, cost = kmed.k_medoids(D, 6)

save_obj(classifications, 'K6')

print("Script complete.")