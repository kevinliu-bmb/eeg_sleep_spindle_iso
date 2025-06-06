import os
import mne
from pyedflib.highlevel import write_edf
from tqdm import tqdm


sids = [
'11639535_20220704_143013_fil',
'12543262_20220913_142921_fil',
'11549585_20200702_143515_fil',
'11336460_20220725_100434_fil',
]
groups = [
'TD',
'TD',
'TD',
'ASD',
]

ch_names32 = ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8', 'EEG 9', 'EEG 10', 'EEG 11', 'EEG 12', 'EEG 13', 'EEG 14', 'EEG 15', 'EEG 16', 'EEG 17',
#'EEG 18,  # nasion
'EEG 19', 'EEG 20',
# 'EEG 21', 'EEG 22', 'EEG 23', 'EEG 24', 'EEG 25', 'EEG 26', # not on 64 channel montage
'EEG 27', 'EEG 28',
'EEG 29', 'EEG 30',  # not on head
'EEG 31', 'EEG 32', # not on head
]

ch_64_to_32_mapping = {
10:1, 5:2,
12:3, 60:4,
20:5, 50:6,
28:7, 42:8,
35:9, 39:10,
18:11, 58:12,
24:13, 52:14,
30:15, 44:16,
6:17, #?:18,
34:19,
37:20,
#?:21, ?:22,
#?:23, ?:24,
#?:25, ?:26,
8:27, 4:28,
63:29, 62:30,
64:31, 61:32,
}
ch_32_to_64_mapping = {v:k for k,v in ch_64_to_32_mapping.items()}
ch_names64 = ['EEG '+str(ch_32_to_64_mapping[int(x.split(' ')[1])]) for x in ch_names32]

output_folder = 'edf-64to32'
os.makedirs(output_folder, exist_ok=True)

for sid, group in tqdm(zip(sids, groups), total=len(sids)):
    output_path = os.path.join(output_folder, f'{sid}.edf')
    #if os.path.exists(output_path):
    #    continue
        
    edf_path = f'/data/haoqisun/Autistm-sleep-Kong/asdtdegidata-edf/{group}/{sid}.edf'
    edf = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    #ch_names = edf.get_info['ch_names']
    eeg = edf.get_data(picks=ch_names64)
    eeg *= 1e6
    Fs = edf.info['sfreq']
    annot = [[x['onset'], x['duration'], x['description']] for x in edf.annotations]
    
    signal_headers = [{'label': x, 'dimension': 'uV', 'sample_rate': Fs, 'sample_frequency': Fs, 'physical_max': 5000.0, 'physical_min': -5000.0, 'digital_max': 32767, 'digital_min': -32767, 'prefilter': '', 'transducer': 'AgAgCl electrode'} for x in ch_names32]
    header = {'startdate': edf.info['meas_date'].replace(tzinfo=None), 'annotations': annot}
    write_edf(output_path, eeg, signal_headers, header=header, file_type=1)
    
