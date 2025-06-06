import os
from itertools import groupby
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import mne
import lunapi as lp
proj = lp.proj()


def add_symmetric_detection(df1, df2, ch1, ch2, col_ch, col_start, col_stop, col_other):
    df3 = []
    for i in range(len(df1)):
        start1, stop1 = df1.loc[i,col_start], df1.loc[i,col_stop]
        starts2, stops2 = df2[col_start].values, df2[col_stop].values
        if not (((starts2>=start1)&(starts2<=stop1)) | ((stops2>=start1)&(stops2<=stop1)) | ((starts2<=start1)&(stops2>=stop1))).any():
            df3.append({col_ch:ch2, col_start:start1, col_stop:stop1, col_other:df1.loc[i,col_other]})
    for j in range(len(df2)):
        start2, stop2 = df2.loc[j,col_start], df2.loc[j,col_stop]
        starts1, stops1 = df1[col_start].values, df1[col_stop].values
        if not (((starts1>=start2)&(starts1<=stop2)) | ((stops1>=start2)&(stops1<=stop2)) | ((starts1<=start2)&(stops1>=stop2))).any():
            df3.append({col_ch:ch1, col_start:start2, col_stop:stop2, col_other:df2.loc[j,col_other]})
    df_res = []
    if len(df1)>0: df_res.append(df1)
    if len(df2)>0: df_res.append(df2)
    if len(df3)>0: df_res.append(pd.DataFrame(df3))
    if len(df_res)>0:
        df_res = pd.concat(df_res, axis=0, ignore_index=True)
    return df_res
    

def get_spindle_so(sid, edf_path, sleep_stage_path, artifact_path, ch_names, q, save_path, ch_pairs):
    p = proj.inst( str(sid) )
    p.attach_edf(edf_path)
    
    edf = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    Fs = edf.info['sfreq']
    sleep_stages = np.load(sleep_stage_path)['sleep_stages']
    sleep_stages[np.isnan(sleep_stages)] = -1
    artifacts = np.load(artifact_path)['artifacts']
    artifacts[np.isnan(artifacts)] = -1
    mapping = {0:'N2', 1:'R', 2:'W'}
    df_annot = {'description':[], 'start':[], 'end':[]}
    cc = 0
    for k,l in groupby(sleep_stages):
        ll = len(list(l))
        if k in mapping:
            df_annot['description'].append(mapping[k])
            df_annot['start'].append(cc/Fs)
            df_annot['end'].append((cc+ll)/Fs)
        cc += ll
    cc = 0
    for k,l in groupby(artifacts):
        ll = len(list(l))
        if k==1:
            #for ch in ch_names:
            df_annot['description'].append('artifact')#_'+ch)
            df_annot['start'].append(cc/Fs)
            df_annot['end'].append((cc+ll)/Fs)
        cc += ll
    annot_path2 = 'tmp.annot'
    df_annot = pd.DataFrame(df_annot)
    df_annot.to_csv(annot_path2, index=False, header=False, sep='\t')
    p.attach_annot(annot_path2)
    
    # first detect m-spindles, then save as .annot
    annot_path = f'tmp_{sid}.annot'
    df_msps = []
    for ch in ch_names:
        cmd = [
            'FREEZE tag=original',
            f'SIGNALS keep={ch}',
            f'FILTER sig={ch} bandpass=0.3,35 bandstop=45,55 ripple=0.01 tw=0.5',
            'EPOCH',
            f'MASK ifnot=N2,N3',
            f'MASK mask-if=artifact',#edf_annot[artifact_{ch}]',
            'RE',
            f'SPINDLES sig={ch} fc-lower=10.5 fc-upper=15.5 fc-step=0.5 cycles=7 collate-within-channel per-spindle q={q}',
            'THAW tag=original', ]
        p.eval(' & '.join([x.split('%')[0] for x in cmd]))
        
        df_msp = p.table('SPINDLES', 'CH_MSPINDLE')
        print(ch, df_msp)
        if df_msp is None or len(df_msp)==0:
            df_ = pd.DataFrame(data=np.ones((0,4)), columns=['channel','MSP_START','MSP_STOP','MSP_F'])
            df_msps.append(df_)
        else:
            df_msp['channel'] = ch
            #df_msp = df_msp.sort_values('MSP_START')
            df_msps.append(df_msp[['channel','MSP_START','MSP_STOP','MSP_F']])
        p.refresh()
    os.remove(annot_path2)
        
    # add symmetric channel detections
    df_msps_sym = []
    for ch in ch_pairs:
        res = add_symmetric_detection(df_msps[ch[0]], df_msps[ch[1]], ch_names[ch[0]], ch_names[ch[1]], 'channel', 'MSP_START', 'MSP_STOP', 'MSP_F')
        if len(res)>0:
            df_msps_sym.append(res)
            
    df_sp = []
    df_so = []
    if len(df_msps_sym)>0:
        df_msp = pd.concat(df_msps_sym, axis=0, ignore_index=True)
        df_msp['class'] = 'sp_all'
        df_msp['MSP_START'] = df_msp.MSP_START.astype(float)
        df_msp['MSP_STOP'] = df_msp.MSP_STOP.astype(float)
        df_msp2 = df_msp.copy()
        df_msp2.loc[df_msp2.MSP_F<13, 'class'] = 'sp_slow'
        df_msp2.loc[df_msp2.MSP_F>=13, 'class'] = 'sp_fast'
        df_msp = pd.concat([df_msp, df_msp2], axis=0)
        df_msp['instance'] = '.'
        df_msp['meta'] = '.'
        df_annot = df_annot.rename(columns={'description':'class', 'start':'MSP_START', 'end':'MSP_STOP'})
        df_annot['instance'] = '.'
        df_annot['meta'] = '.'
        df_annot['channel'] = '.'
        cols = ['class', 'instance', 'channel', 'MSP_START', 'MSP_STOP', 'meta']
        df_msp = pd.concat([df_msp[cols],df_annot[cols]], axis=0, ignore_index=True)
        df_msp.to_csv(annot_path, float_format='%.4f', index=False, header=None, sep='\t')
    
        # then compute features as .db file
        for ch in ch_names:
            save_path2 = save_path.replace('.db', f'_{ch}.db')
            cmd = [f'SIGNALS keep={ch}',
                f'FILTER sig={ch} bandpass=0.3,35 bandstop=45,55 ripple=0.01 tw=0.5',
                'EPOCH',
                f'MASK ifnot=N2,N3',
                f'MASK mask-if=artifact',#edf_annot[artifact_{ch}]',
                'RE',]
            if (df_msp['class']=='sp_slow').sum()>0:
                cmd.extend([
                'TAG mysp/slow',
                f'SPINDLES precomputed=sp_slow per-spindle if so mag=3 verbose-coupling tl={ch} q={q-0.1}',])
            if (df_msp['class']=='sp_fast').sum()>0:
                cmd.extend([
                'TAG mysp/fast',
                f'SPINDLES precomputed=sp_fast per-spindle if so mag=3 verbose-coupling tl={ch} q={q-0.1}',])
            cmd.extend([
                'TAG mysp/all',
                f'SPINDLES precomputed=sp_all per-spindle if so mag=3 verbose verbose-coupling tl={ch} q={q-0.1}',])
            subprocess.run(['luna', edf_path, 'annot-file='+annot_path, '-o', save_path2,
                '-s', ' & '.join([x.split('%')[0] for x in cmd])])
            proj.import_db(save_path2)
            
            df_sp_ = proj.table('SPINDLES', 'CH_F_SPINDLE_mysp')
            if df_sp_ is not None and len(df_sp_)>0:
                df_sp.append( df_sp_[df_sp_.mysp=='all'].drop(columns='mysp') )
            df_so_ = proj.table('SPINDLES', 'CH_N_mysp')
            if df_so_ is not None and len(df_so_)>0:
                df_so.append( df_so_.drop(columns='mysp') )
            
        os.remove(annot_path)
        
    if len(df_sp)>0:
        df_sp = pd.concat(df_sp, axis=0, ignore_index=True)
    else:
        cols = ['ID', 'CH', 'F', 'SPINDLE', 'AMP', 'ANCHOR', 'CHIRP', 'DUR', 'FFT',
       'FRQ', 'FRQ1', 'FRQ2', 'FWHM', 'IF', 'ISA', 'MAXSTAT', 'MEANSTAT',
       'NOSC', 'PASS', 'PEAK', 'PEAK_SP', 'Q', 'SO_NEAREST', 'SO_NEAREST_NUM',
       'SO_PHASE_ANCHOR', 'START', 'START_SP', 'STOP', 'STOP_SP', 'SYMM',
       'SYMM2', 'TROUGH', 'TROUGH_SP']
        df_sp = pd.DataFrame(data=np.ones((0,len(cols))), columns=cols)
    if len(df_so)>0:
        df_so = pd.concat(df_so, axis=0, ignore_index=True)
    else:
        cols = ['ID', 'CH', 'N', 'DOWN_AMP', 'DOWN_IDX', 'DUR', 'DUR1', 'DUR2',
       'DUR_CHK', 'P2P_AMP', 'SLOPE', 'START', 'START_IDX', 'STOP', 'STOP_IDX',
       'TRANS', 'TRANS_FREQ', 'UP_AMP', 'UP_IDX']
        df_so = pd.DataFrame(data=np.ones((0,len(cols))), columns=cols)
    return df_sp, df_so


def main():
    df = pd.read_excel('mastersheet.xlsx')
    df['SleepStagePath'] = df.SleepStagePath.str.replace('/sleep_stages/','/sleep_stages2/')
    df = df[~df.EDFPath.str.contains('64to32')].reset_index(drop=True)
    
    q = 0.53
    """
    q = 0
    np.random.seed(2024)
    ids = np.r_[
        np.random.choice(np.where(df.Group=='TD')[0], 10, replace=False),
        np.random.choice(np.where(df.Group=='ASD')[0], 10, replace=False), ]
    df = df.iloc[ids].reset_index(drop=True)
    """
    
    detection_dir = f'detection_luna_q={q}'
    os.makedirs(detection_dir, exist_ok=True)
    
    ch_names = ['EEG_1', 'EEG_2', 'EEG_3', 'EEG_4', 'EEG_5', 'EEG_6', 'EEG_7', 'EEG_8', 'EEG_9', 'EEG_10', 'EEG_11', 'EEG_12', 'EEG_13', 'EEG_14', 'EEG_15', 'EEG_16', 'EEG_17',
    #'EEG_18',  # nasion
    'EEG_19', 'EEG_20',
    'EEG_21', 'EEG_22', 'EEG_23', 'EEG_24', 'EEG_25', 'EEG_26', # not on 64 channel montage
    'EEG_27', 'EEG_28',
    #'EEG_29', 'EEG_30',  # not on head
    #'EEG_31', 'EEG_32',  # not on head
    ]
    ch_pairs = [[ch_names.index('EEG_1'), ch_names.index('EEG_2')],
    [ch_names.index('EEG_3'), ch_names.index('EEG_4')],
    [ch_names.index('EEG_5'), ch_names.index('EEG_6')],
    [ch_names.index('EEG_7'), ch_names.index('EEG_8')],
    [ch_names.index('EEG_9'), ch_names.index('EEG_10')],
    [ch_names.index('EEG_11'), ch_names.index('EEG_12')],
    [ch_names.index('EEG_13'), ch_names.index('EEG_14')],
    [ch_names.index('EEG_15'), ch_names.index('EEG_16')],
    [ch_names.index('EEG_21'), ch_names.index('EEG_22')],
    [ch_names.index('EEG_23'), ch_names.index('EEG_24')],
    [ch_names.index('EEG_25'), ch_names.index('EEG_26')],
    #[ch_names.index('EEG_31'), ch_names.index('EEG_32')],
    ]
    """
1     Fp1
2     Fp2
3     F3
4     F4
5     C3
6     C4
7     P3
8     P4
9     O1
10    O2
11    F7
12    F8
13    T7
14    T8
15    P7
16    P8
17    Fz
### 18    nasion
19    Pz
20    Oz
21    above left mastoid
22    above right mastoid
23    between P7 and T7, lower side
24    between P8 and T8, lower side
25    between P7 and O1, lower side
26    between P8 and O2, lower side
27    middle line, between 18:nasion and Fz
28    middle line, between Cz and Fz
### 29    near eye
### 30    near eye
### 31    between Fp1 and F7, lower side
### 32    between Fp2 and F8, lower side
    """
    
    for i in tqdm(range(len(df))):
        sid = df.SID.iloc[i]
        save_path = os.path.join(detection_dir, f'luna_detection_{sid}.db')
        if all([os.path.exists(save_path.replace('.db',f'_{x}.db')) for x in ch_names]):
            continue
            
        edf_path = df.EDFPath.iloc[i]
        if not os.path.exists(edf_path): continue
        sleep_stage_path = df.SleepStagePath.iloc[i]
        if not os.path.exists(sleep_stage_path): continue
        artifact_path = df.ArtifactPath.iloc[i]
        if not os.path.exists(artifact_path): continue
        df_sp, df_so = get_spindle_so(sid, edf_path, sleep_stage_path, artifact_path, ch_names, q, save_path, ch_pairs)
        
        df_sp['CH'] = df_sp.CH.astype(str).str.replace('_', ' ')
        df_sp['DESC'] = [f'SP q{df_sp.Q.iloc[ii]:.1f} f{df_sp.FFT.iloc[ii]:.1f}@@{df_sp.CH.iloc[ii]}' for ii in range(len(df_sp))]
        df_sp = df_sp[['START', 'STOP', 'DUR', 'CH', 'Q', 'DESC']]
        df_sp.to_csv(os.path.join(detection_dir, f'spindle_{sid}-annotation.csv'), index=False)
        
        df_so['CH'] = df_so.CH.astype(str).str.replace('_', ' ')
        df_so['DESC'] = [f'SO@@{df_so.CH.iloc[ii]}' for ii in range(len(df_so))]
        df_so = df_so[['START', 'STOP', 'DUR', 'CH', 'DESC']]
        df_so.to_csv(os.path.join(detection_dir, f'so_{sid}-annotation.csv'), index=False)
        
        
if __name__=='__main__':
    main()

