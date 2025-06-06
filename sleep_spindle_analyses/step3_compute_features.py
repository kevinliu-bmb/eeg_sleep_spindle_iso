from collections import defaultdict
from itertools import groupby, product
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm
import mne

import lunapi as lp
proj = lp.proj()
proj.silence()


def get_spindle_so(luna_db_paths, ch_names, Fs):
    """
7   SPINDLES       CH_F_SPINDLE_mysp --> NO, per spindle info
10  SPINDLES               CH_N_mysp --> NO, per SO info

9   SPINDLES               CH_F_mysp --> YES, overnight spindle summary stats
13  SPINDLES                 CH_mysp --> YES, overnight SO summary stats

5   SPINDLES         CH_F_PHASE_mysp --> YES, SOPL_CWT: SO phase vs spindle wavelet power
11  SPINDLES           CH_PHASE_mysp --> YES, SOPL_EEG: SO phase vs EEG
8   SPINDLES            CH_F_SP_mysp --> YES, SOTL_CWT: SO time vs spindle wavelet power
12  SPINDLES              CH_SP_mysp --> YES, SOTL_EEG: SO time vs EEG

6   SPINDLES        CH_F_RELLOC_mysp --> YES, IF within spindle, 5 bins

4   SPINDLES  CH_F_PHASE_RELLOC_mysp --> NO, IF by PHASE by RELLOC, too much, don't know how to compare
3   SPINDLES               CH_E_mysp --> NO, stats per epoch, not useful
    """
    so_phases = np.arange(10,360,20)
    sp_types = ['all', 'slow', 'fast']
    sp_rellocs = [1,2,3,4,5]
    so_samples = np.arange(-int(Fs), int(Fs)+1)
    
    res = {}
    cols = ['AMP', 'CDENS', 'CHIRP', 'COUPL_ANCHOR', 'COUPL_ANGLE', 'COUPL_MAG',
       'COUPL_OVERLAP', 'DENS', 'DISPERSION', 'DUR', 'FFT', 'ISA_M', 'ISA_S',
       'ISA_T', 'MINS', 'N', 'NE', 'NOSC', 'R_PHASE_IF', 'SEC_AMP', 'SEC_P2P',
       'SEC_TROUGH', 'SYMM', 'SYMM2', 'SYMM_AMP', 'SYMM_TROUGH', 'UDENS']
    #for ch in tqdm(ch_names):
    for ch in ch_names:
        #print(ch)
        if not os.path.exists(luna_db_paths[ch]):
            continue
        proj.import_db(luna_db_paths[ch])
        
        # overnight spindle summary stats
        df = proj.table('SPINDLES', 'CH_F_mysp')
        if df is not None and len(df)>0:
            for t in sp_types:
                ids = df.mysp==t
                nn = ids.sum()
                for p in cols:
                    if nn>0 and p in df.columns:
                        res[f'SP_{p}_{t}@{ch}'] = df[p][ids].iloc[0]
            
        # overnight SO summary stats
        df = proj.table('SPINDLES', 'CH_mysp')
        if df is not None and len(df)>0:
            df = df.drop(columns=['ID', 'CH', 'mysp', 'SO'])
            df = pd.DataFrame(data=np.nanmean(df.values.astype(float), axis=0, keepdims=True), columns=df.columns)
            for p in df.columns:
                res[f'{p}@{ch}'] = df[p].iloc[0]
        
        """
        # SO phase vs spindle wavelet power
        df = proj.table('SPINDLES', 'CH_F_PHASE_mysp')
        if len(df)>0:
            for t,p in product(sp_types, so_phases):
                ids = (df.mysp==t)&(df.PHASE==p)
                if ids.sum()>0:
                    res[f'SP_WAVELET_{t}_AT_SO_PHASE{p:.0f}@{ch}'] = df.SOPL_CWT[ids].iloc[0]
        df = proj.table('SPINDLES', 'CH_PHASE_mysp')
        if len(df)>0:
            for t,p in product(sp_types, so_phases):
                ids = (df.mysp==t)&(df.PHASE==p)
                if ids.sum()>0:
                    res[f'SO_EEG_{t}_AT_SO_PHASE{p:.0f}@{ch}'] = df.SOPL_EEG[ids].iloc[0]
            
        df = proj.table('SPINDLES', 'CH_F_SP_mysp')
        if len(df)>0:
            for t,p in product(sp_types, so_samples):
                ids = (df.mysp==t)&(df.SP==p)
                if ids.sum()>0:
                    res[f'SP_WAVELET_{t}_AT_SO_SAMPLE{p:.0f}@{ch}'] = df.SOTL_CWT[ids].iloc[0]
        df = proj.table('SPINDLES', 'CH_SP_mysp')
        if len(df)>0:
            for t,p in product(sp_types, so_samples):
                ids = (df.mysp==t)&(df.SP==p)
                if ids.sum()>0:
                    res[f'SO_EEG_{t}_AT_SO_SAMPLE{p:.0f}@{ch}'] = df.SOTL_EEG[ids].iloc[0]

        # IF within spindle, 5 bins
        df = proj.table('SPINDLES', 'CH_F_RELLOC_mysp')
        if len(df)>0:
            df['RELLOC'] = df.RELLOC.astype(int)
            for t,p in product(sp_types, sp_rellocs):
                ids = (df.mysp==t)&(df.RELLOC==p)
                if ids.sum()>0:
                    res[f'SP_IF_{t}_AT_BIN{p}@{ch}'] = df.IF[ids].iloc[0]
        """

    return res


def main():
    sp_so_dir = 'detection_luna_q=0.53'
    epoch_time = 30
    stages_txt = ['N3', 'N2', 'N1', 'R', 'W']
    stages_num = [1,2,3,4,5]
    stages_txt2num = {k:v for k,v in zip(stages_txt, stages_num)}
    ch_names = ['EEG_1', 'EEG_2', 'EEG_3', 'EEG_4', 'EEG_5', 'EEG_6', 'EEG_7', 'EEG_8', 'EEG_9', 'EEG_10', 'EEG_11', 'EEG_12', 'EEG_13', 'EEG_14', 'EEG_15', 'EEG_16', 'EEG_17',
    #'EEG_18,  # nasion
    'EEG_19', 'EEG_20',
     'EEG_21', 'EEG_22', 'EEG_23', 'EEG_24', 'EEG_25', 'EEG_26', # not on 64 channel montage
    'EEG_27', 'EEG_28', #not 10-20
    #'EEG_29', 'EEG_30',  # not on head
    #'EEG_31', 'EEG_32',  # not on head
    ]
    ch_names2 = [
    'Fp1', 'Fp2',
    'F3', 'F4',
    'C3', 'C4',
    'P3', 'P4',
    'O1', 'O2',
    'F7', 'F8',
    'T7', 'T8',
    'P7', 'P8',
    'Fz', 'Pz', 'Oz',
    'above_M1', 'above_M2',
    'between_P7_T7', 'between_P8_T8',
    'between_P7_O1', 'between_P8_O2',
    'between_nasion_Fz', 'between_Cz_Fz'
    ]
    ch_name_mapping = {x1:x2 for x1,x2 in zip(ch_names, ch_names2)}
    
    df = pd.read_excel('mastersheet.xlsx')
    #df = df.to_dict(orient='list')
    #df.pop('EDFPath')
    #df.pop('SleepStagePath')
    #df.pop('ArtifactPath')
    #df.pop('Channels')
    fss = df.Fs
    df = df.drop(columns=['EDFPath', 'SleepStagePath','ArtifactPath','Channels','Fs'])

    for i in tqdm(range(30,len(df))):
        sid = df.SID.iloc[i]
        
        sp_so_paths = {ch:os.path.join(sp_so_dir, f'luna_detection_{sid}_{ch}.db') for ch in ch_names}
        sp_so_paths = {k:v for k,v in sp_so_paths.items() if os.path.exists(v)}
        ch_names22 = list(sp_so_paths.keys())
        feats = get_spindle_so(sp_so_paths, ch_names22, fss[i])
        for k,v in feats.items():
            ch = k.split('@')[-1]
            k2 = k.replace('@'+ch, '_'+ch_name_mapping[ch])
            if k2 not in df.columns:
                df[k2] = np.nan
            df.loc[i,k2] = v
    
        if i%10==0 or i==len(df)-1:
            df.to_csv('features_spindles2.csv.zip', index=False)
        

if __name__=='__main__':
    main()

