import os, glob, datetime
import numpy as np
from scipy.stats import mode
import mne
from tqdm import tqdm
import matplotlib.pyplot as plt


data_paths = glob.glob('asdtdegidata-edf/ASD/*.edf') + glob.glob('asdtdegidata-edf/TD/*.edf')

result_dir = 'artifacts'
os.makedirs(result_dir, exist_ok=True)
result_dir_asd = os.path.join(result_dir, 'ASD')
os.makedirs(result_dir_asd, exist_ok=True)
result_dir_td = os.path.join(result_dir, 'TD')
os.makedirs(result_dir_td, exist_ok=True)

ch_names = ['EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 9', 'EEG 10']
epoch_time = 30


def on_keypress(event):
    print('press', event.key)
    if event.key.lower()=='1':
        artifacts.append([current_t, 1])
        print(artifacts)
    elif event.key.lower()=='0':
        artifacts.append([current_t, 0])
        print(artifacts)
    
    
def on_mousepress(event):
    print(f'current_t = {event.xdata}')
    global current_t
    current_t = event.xdata

    
            
for path in tqdm(data_paths):
    sid = path.split(os.sep)[-1].replace('.edf','')
    output_path = os.path.join(result_dir, path.split(os.sep)[1], f'artifacts_{sid}.npz')
    figure_path = os.path.join(result_dir, path.split(os.sep)[1], f'artifacts_{sid}.png')
    if os.path.exists(output_path) and os.path.exists(figure_path):
        continue
    
    edf = mne.io.read_raw_edf(path, verbose=False)
    Fs = edf.info['sfreq']
    start_time = edf.info['meas_date'].replace(tzinfo=None)
    annots = edf.annotations
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
    artifacts = []
    
    figsize = (13,5)
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.imshow(spec_db.T, aspect='auto', origin='lower', cmap='turbo', vmin=0, vmax=25, extent=(tt.min(), tt.max(), freq.min(), freq.max()))
    
    for annot in annots:
        ax.plot([annot['onset']]*2, [16,19], c='w', lw=2)
        t = ax.text(annot['onset'], 16, annot['description'], ha='center', va='top', color='k', rotation=90)
        t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none'))
        
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_xlim(tt.min(), tt.max())
    
    fig.canvas.mpl_connect('key_press_event', on_keypress)
    fig.canvas.mpl_connect('button_press_event', on_mousepress)
    plt.tight_layout()
    plt.show()
    
    
    artifacts2 = np.zeros(eeg.shape[1])
    for t, s in artifacts:
        idx = int(round(t*Fs))
        artifacts2[idx:] = s
    
    artifacts3 = np.zeros(eeg.shape[1])
    for idx in start_ids:
        x = np.array(artifacts2[idx:idx+epoch_size])
        x = x[~np.isnan(x)]
        artifacts3[idx:idx+epoch_size] = mode(x, keepdims=False).mode
    
    np.savez_compressed(output_path, artifacts=artifacts3)

    artifacts4 = np.array(artifacts3)
    artifacts4[artifacts4==0] = np.nan
    
    plt.close()
    fig = plt.figure(figsize=figsize)
    
    ax = fig.add_subplot(111)
    ax.imshow(spec_db.T, aspect='auto', origin='lower', cmap='turbo', vmin=0, vmax=25, extent=(tt.min(), tt.max(), freq.min(), freq.max()))
    ax.plot(np.arange(eeg.shape[1])/Fs, artifacts4*(freq.max()+0.25), c='k', lw=4)
    for annot in annots:
        ax.plot([annot['onset']]*2, [16,19], c='w', lw=2)
        t = ax.text(annot['onset'], 16, annot['description'], ha='center', va='top', color='k', rotation=90)
        t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none'))
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_xlim(tt.min(), tt.max())
    ax.set_ylim(freq.min(), freq.max()+0.5)
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(figure_path, bbox_inches='tight')
    
