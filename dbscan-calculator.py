import os
import numpy as np
import pandas as pd
from analysis import *

import random
import time

from sklearn.cluster import DBSCAN

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
                     'E_PROP', 'SIN_I_PROP', 'V_MAG', 'B_MAG', 'H', 'G',
                     'A_MAG', 'A_ERR', 'SMOC_ID', 'PHASE', 'D_COUNTER', 
                     'TOTAL_D_COUNT'],
                     axis=1, inplace=True)

print("Loaded the SDSS")
time.sleep(1)

# ---------------------------------------------------------------------------------------------
wave_mags = ["U_MAG", "G_MAG", "R_MAG", "I_MAG", "Z_MAG"]
wave_errs = ["U_ERR", "G_ERR", "R_ERR", "I_ERR", "Z_ERR"]

# Ensure data integrity. Every observation MUST have a U, G, R, I, and Z value, and associated error.
sdss_df.dropna(subset=wave_mags + wave_errs, inplace=True)

training_data = sdss_df[wave_mags].values.tolist()

print("Training data created.")
time.sleep(1)

# ---------------------------------------------------------------------------------------------

# Pick random indices to sample from. Has no repitition.
arr_size = len(sdss_df) - 1

index_dict = {}
while len(index_dict.keys()) < 100000:
    index_dict[random.randint(0, arr_size)] = 0

random_sample = np.array([training_data[idx] for idx in index_dict.keys()])
#random_sample = training_data

print("Random sample created")
time.sleep(1)
# ---------------------------------------------------------------------------------------------

# Create DBSCAN algorithm for cluster analysis
dbscan_instance = DBSCAN(eps=10, min_samples=1).fit(random_sample)

print(np.unique(dbscan_instance.labels_))
np.savetxt("run-100.txt", dbscan_instance.labels_, fmt='%d')
print("Script complete.")