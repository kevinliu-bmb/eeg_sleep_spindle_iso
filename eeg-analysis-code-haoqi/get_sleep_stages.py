import os, glob, datetime
import numpy as np
import pandas as pd
from scipy.stats import mode
import mne
from tqdm import tqdm
import matplotlib.pyplot as plt


data_paths = glob.glob('asdtdegidata-edf/ASD/*.edf') + glob.glob('asdtdegidata-edf/TD/*.edf')

result_dir = 'sleep_stages'
os.makedirs(result_dir, exist_ok=True)
result_dir_asd = os.path.join(result_dir, 'ASD')
os.makedirs(result_dir_asd, exist_ok=True)
result_dir_td = os.path.join(result_dir, 'TD')
os.makedirs(result_dir_td, exist_ok=True)

ch_names = ['EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 9', 'EEG 10']
epoch_time = 30


def on_keypress(event):
    print('press', event.key)
    if event.key.lower()=='n':
        sleep_stages.append([current_t, 0])
        print(sleep_stages)
    elif event.key.lower()=='r':
        sleep_stages.append([current_t, 1])
        print(sleep_stages)
    elif event.key.lower()=='w':
        sleep_stages.append([current_t, 2])
        print(sleep_stages)
    
    
def on_mousepress(event):
    print(f'current_t = {event.xdata}')
    global current_t
    current_t = event.xdata

    
sids = [path.split(os.sep)[-1].replace('.edf','') for path in data_paths]
            
for path, sid in tqdm(zip(data_paths, sids)):
    #if '4177114' not in sid: continue
    output_path = os.path.join(result_dir, path.split(os.sep)[1], f'sleep_stages_{sid}.npz')
    figure_path = os.path.join(result_dir, path.split(os.sep)[1], f'sleep_stages_{sid}.png')
    if os.path.exists(output_path) and os.path.exists(figure_path):
        continue
    
    edf = mne.io.read_raw_edf(path, verbose=False)
    Fs = edf.info['sfreq']
    start_time = edf.info['meas_date'].replace(tzinfo=None)
    annots = pd.DataFrame(data=edf.annotations).drop(columns='orig_time')
    print(annots)

    #annots = pd.concat([annots, pd.DataFrame(data={'onset':[(3272.272+3554.376)/2], 'duration':[0.001], 'description':['m']})], axis=0).sort_values('onset', ignore_index=True)
    #annots = pd.concat([annots, pd.DataFrame(data={'onset':[3635.873-120, 4070.428-180], 'duration':[0.001, 0.001], 'description':['m','m']})], axis=0).sort_values('onset', ignore_index=True)
    #annots = annots[np.arange(len(annots))!=5].reset_index(drop=True)
    #annots.loc[3,'description']='m'
    #annots = annots[~np.in1d(np.arange(len(annots)), [6,11,12,15,18])].reset_index(drop=True)
    
    eeg = edf.get_data(picks=ch_names)*1e6
    epoch_size = int(round(epoch_time*Fs))
    start_ids = np.arange(0, eeg.shape[1]-epoch_size+1, epoch_size)
    segs = np.array([eeg[:,x:x+epoch_size] for x in start_ids])
    spec, freq = mne.time_frequency.psd_array_multitaper(segs, Fs, fmin=0.5, fmax=20, bandwidth=0.5, normalization='full', verbose=False)
    spec_db = 10*np.log10(spec)
    spec_db[np.isinf(spec_db)] = np.nan
    spec_db = np.nanmean(spec_db, axis=1)
    tt = start_ids/Fs
    
    xticks = np.arange(tt.min(), tt.max()-1200+1, 1200)
    xticklabels = [(start_time+datetime.timedelta(seconds=float(x))).strftime('%H:%M') for x in xticks]
        
    current_t = 0
    sleep_stages = []
    
    figsize = (13,5)
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.imshow(spec_db.T, aspect='auto', origin='lower', cmap='turbo', vmin=0, vmax=25, extent=(tt.min(), tt.max(), freq.min(), freq.max()))
    
    for ii, r in annots.iterrows():
        ax.plot([r.onset]*2, [16,19], c='w', lw=2)
        t = ax.text(r.onset, 16, r.description, ha='center', va='top', color='k', rotation=90)
        t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none'))
        
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_xlim(tt.min(), tt.max())
    
    fig.canvas.mpl_connect('key_press_event', on_keypress)
    fig.canvas.mpl_connect('button_press_event', on_mousepress)
    plt.tight_layout()
    plt.show()
    
    
    sleep_stages2 = np.zeros(eeg.shape[1])
    for t, s in sleep_stages:
        idx = int(round(t*Fs))
        sleep_stages2[idx:] = s
    
    sleep_stages3 = np.zeros(eeg.shape[1])
    for idx in start_ids:
        x = np.array(sleep_stages2[idx:idx+epoch_size])
        x = x[~np.isnan(x)]
        sleep_stages3[idx:idx+epoch_size] = mode(x, keepdims=False).mode
        
    for ii, r in annots.iterrows():
        if r.description not in ['Awake', 'Asleep']:
            idx = np.searchsorted(start_ids/Fs, r.onset)
            start = max(0, start_ids[idx]-epoch_size)
            end = min(eeg.shape[1], start_ids[idx]+epoch_size*2)
            sleep_stages3[start:end] = 3
    #sleep_stages3[int(2031.184*Fs):int(2457.460*Fs)] = 3
    
    np.savez_compressed(output_path, sleep_stages=sleep_stages3)


    plt.close()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2,1,height_ratios=(1,5))
    
    ax = fig.add_subplot(gs[0,0]); ax0 = ax
    ax.plot(np.arange(eeg.shape[1])/Fs, sleep_stages3, c='k')
    ax.set_ylim(-0.2, 3.2)
    ax.set_yticks([0,1,2,3], labels=['NREM', 'REM', 'W', 'Abn'])
    ax.yaxis.grid(True)
    
    ax = fig.add_subplot(gs[1,0], sharex=ax0)
    ax.imshow(spec_db.T, aspect='auto', origin='lower', cmap='turbo', vmin=0, vmax=25, extent=(tt.min(), tt.max(), freq.min(), freq.max()))
    
    for ii, r in annots.iterrows():
        ax.plot([r.onset]*2, [16,19], c='w', lw=2)
        t = ax.text(r.onset, 16, r.description, ha='center', va='top', color='k', rotation=90)
        t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none'))
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_xlim(tt.min(), tt.max())
    ax.set_ylim(freq.min(), freq.max())
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(figure_path, bbox_inches='tight')
    
