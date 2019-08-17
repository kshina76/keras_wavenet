import os
import argparse
import librosa
import random
import numpy as np
import MuLaw as ml

class AudioReader(object):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, threshold,
                receptive_field, quantize, directory):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.threshold = threshold
        self.mu_law = ml.MuLaw(quantize)
        self.quantize = quantize
        if receptive_field is None:
            self.receptive_field = None
        else:
            self.receptive_field = receptive_field + 1
        self.directory = directory

 	# get file path of all selected directory. argument is from parser
    def get_files(self, directory):
        files = []
        # [directory path,directory name,filename] of os.walk return value
        for dir_path,dir_name,file_name in os.walk(directory):
            for file_name in file_name:	
                files.append(os.path.join(dir_path,file_name))
        return files

	# get audio one by one.
    def get_audio(self, files, sample_rate):
        for file in files:
            # transfer audio shape into column vector.
            audio, _ = librosa.load(file, sr=sample_rate)
            yield audio

    def preprocess(self):
        files = self.get_files(self.directory)
        #print(files)
        for audio in self.get_audio(files, self.sample_rate):
            audio, _ = librosa.effects.trim(audio, self.threshold)
            # nomalization
            audio /= np.abs(audio).max()
            audio = audio.astype(np.float32)

            # mu-law transform
            quantized = self.mu_law.transform(audio)

            # padding or trimming
            if self.threshold is not None:
                if len(audio) <= self.receptive_field:
                    # padding
                    pad = self.receptive_field - len(audio)
                    audio = np.concatenate((audio, np.zeros(pad, dtype=np.float32)))
                    # padding with middle of quantized audio
                    quantized = np.concatenate((quantized, self.quantize // 2 * np.ones(pad)))
                    quantized = quantized.astype(np.int64)
                else:
                    # trimming audio into receptive_field (trimming)
                    start = random.randint(0, len(audio) - self.receptive_field - 1)
                    audio = audio[start:start + self.receptive_field]
                    quantized = quantized[start:start + self.receptive_field]

            # calculate spectrogram
            spectrogram = librosa.feature.melspectrogram(audio, self.sample_rate, 
                                            n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

            # normlaization to spectrogram (/=40 after +=40 can normalize)
            spectrogram += 40
            spectrogram /= 40
            if self.receptive_field is not None:
                spectrogram = spectrogram[:, :self.receptive_field // self.hop_length]
            spectrogram = spectrogram.astype(np.float32)

            # transform row vector into column vector
            one_hot = np.identity(self.quantize, dtype=np.float32)[quantized]
            one_hot = np.expand_dims(one_hot.T, 2)
            spectrogram = np.expand_dims(spectrogram, 2)
            quantized = np.expand_dims(quantized, 1)

            # make target
            one_hot_target = one_hot[:, -1]
            quantized_target = quantized[0]

            '''
            one_hotは、ターゲット変数のために7681サンプルとってきて、特徴量に7680,ターゲットに1 と分割する。
            quantizedも同じ。
            spectrogramは学習するわけではないので、ターゲットは作らなくていい。
            '''

            #yield one_hot[:, :-1], one_hot_target, quantized[1:], quantized_target, spectrogram
            yield one_hot[:, :-1], one_hot_target