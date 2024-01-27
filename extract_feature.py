import os
import librosa
import numpy as np
import pandas as pd

def get_metadata(filename):

    y, sr = librosa.load(filename)
    
    audio_length_samples = len(y)
    test_metadata = []

    # Độ dài của mỗi đoạn (10 giây)
    segment_length_samples = sr * 10

    # Tạo danh sách để chứa các đoạn âm thanh
    collection = []

    # Chia tệp âm thanh thành các đoạn
    start = 0
    while start < audio_length_samples:
        end = start + segment_length_samples
        if end > audio_length_samples:
            end = audio_length_samples
        segment = y[start:end]
        collection.append(segment)
        start = end
    first = len(collection[0])
    for y in collection:
        # fetching tempo
        length = len(y)
        #bỏ đoạn cuối(đoạn cuối thường ít hơn 10s)
        if(length != first):
            break

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

        # fetching beats

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # chroma_stft

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

        # rmse
        
        rmse = librosa.feature.rms(y=y)

        # fetching spectral centroid

        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # spectral bandwidth

        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        # fetching spectral rolloff

        spec_rolloff = librosa.feature.spectral_rolloff(y=y+0.01, sr=sr)[0]

        # zero crossing rate

        zero_crossing = librosa.feature.zero_crossing_rate(y)

        # mfcc

        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        # metadata dictionary

        metadata_dict = [length, np.mean(chroma_stft), np.var(chroma_stft), np.mean(rmse), np.var(rmse),
                         np.mean(spec_centroid), np.var(spec_centroid),
                         np.mean(spec_bw),  np.var(spec_bw),
                         np.mean(spec_rolloff),  np.var(spec_rolloff),
                         np.mean(zero_crossing), np.var(zero_crossing), tempo]

        for i in range(1, 21):
            metadata_dict.extend(
                [np.mean(mfcc[i-1]), np.var(mfcc[i-1])])

        test_metadata.append(metadata_dict)
    return np.array(test_metadata)

folder_path = "D:\\test music\\dataset2\\rnb"

all_metadata = pd.DataFrame()
data_feature = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
        
    metadata_for_file = get_metadata(file_path)
        
    metadata_df = pd.DataFrame(metadata_for_file)
            
    data_feature.append(metadata_df)

metadata = pd.concat(data_feature, ignore_index=True)


csv_filename = '.\dataset2_rnb.csv'
metadata.to_csv(csv_filename, index=False)

print(f"Kết quả đã được lưu vào file CSV: {csv_filename}")
