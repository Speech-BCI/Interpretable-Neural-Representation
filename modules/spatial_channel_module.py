import numpy as np
import hdf5storage
import mne



class Recording:
    def __init__(self, mne_info, recording_index, study_name, recording_uid):
        self.mne_info = mne_info
        self.recording_index = recording_index
        self.study_name = study_name
        self.recording_uid = recording_uid


def get_mne_info():
    load_dir = 'path to ch_information file'
    ch_info_name = 'ch_information file'
    ch_info = hdf5storage.loadmat(load_dir + ch_info_name)
    ch_names = ch_info['ch_names'].squeeze()
    srate = ch_info['srate'].squeeze()
    ch_locs = ch_info['ch_locs']

    flat_ch_names = [item[0][0] for item in ch_names]
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(flat_ch_names, srate, ch_types)

    ch_pos = {str(ch_locs['labels'][0][i][0][0]):
                  np.array([ch_locs['X'][0][i][0, 0],
                            ch_locs['Y'][0][i][0, 0],
                            ch_locs['Z'][0][i][0, 0]])
              for i in range(ch_locs.shape[1])}
    for i, ch_name in enumerate(flat_ch_names):
        if ch_name in ch_pos:
            info['chs'][i]['loc'][:3] = ch_pos[ch_name]

    recording_index = 0
    study_name = 'allwords_88'
    recording_uid = 'allwords_88_0'
    recording = [Recording(info, recording_index, study_name, recording_uid)]

    return recording

