import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
import re
from utils import write
import shutil
from pydub import AudioSegment
from collections import Counter

warnings.filterwarnings("ignore")


def calculate_delta(array, mode):

    rows, cols = array.shape

    if mode == 'train':
        print(rows)
        print(cols)
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]]-array[index[0][1]] +
                     (2 * (array[index[1][0]]-array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate, mode='train'):

    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01,
                             20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    if mode == 'train':
        print(mfcc_feature)
    delta = calculate_delta(mfcc_feature, mode)
    combined = np.hstack((mfcc_feature, delta))
    return combined


def record_audio_train():
    Name = (input("Please Enter Your Name:"))
    for count in range(5):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 10
        device_index = 2
        audio = pyaudio.PyAudio()
        print("----------------------record device list---------------------")
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ",
                      audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        print("-------------------------------------------------------------")
        index = int(input())
        print("recording via index "+str(index))
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, input_device_index=index,
                            frames_per_buffer=CHUNK)
        print("recording started")
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print("recording stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = Name+"-sample"+str(count)+".wav"
        WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)
        trainedfilelist = open("training_set_addition.txt", 'a')
        trainedfilelist.write(OUTPUT_FILENAME+"\n")
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()


def record_audio_test():

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 10
    device_index = 2
    audio = pyaudio.PyAudio()
    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ",
                  audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-------------------------------------------------------------")
    index = int(input())
    print("recording via index "+str(index))
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK)
    print("recording started")
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print("recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    OUTPUT_FILENAME = "sample.wav"
    WAVE_OUTPUT_FILENAME = os.path.join("testing_set", OUTPUT_FILENAME)
    trainedfilelist = open("testing_set_addition.txt", 'a')
    trainedfilelist.write(OUTPUT_FILENAME+"\n")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()


def train_model(train_folder, model_path):

    train_file = "./training_set_addition.txt"
    file_paths = open(train_file, 'r')
    count = 1
    features = np.asarray(())

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    for path in file_paths:
        path = path.strip()
        print(path)

        sr, audio = read(os.path.join(train_folder, path))
        print(sr)
        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == 5:
            gmm = GaussianMixture(
                # n_components=12, max_iter=200, covariance_type='diag', n_init=3)
                n_components=24, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = path.split("_")[0]+".gmm"
            pickle.dump(gmm, open(os.path.join(model_path, picklefile), 'wb'))
            print('+ modeling completed for speaker:', picklefile,
                  " with data point = ", features.shape)
            features = np.asarray(())
            count = 0
        count = count + 1


def test_model(test_folder, model_path):

    test_file = "./testing_set_addition.txt"
    file_paths = open(test_file, 'r')

    gmm_files = [os.path.join(model_path, fname) for fname in
                 os.listdir(model_path) if fname.endswith('.gmm')]

    # Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                in gmm_files]

    # Read the test directory and get the list of test audio files
    precision = 0
    num_files = 0

    for path in file_paths:
        num_files += 1

        path = path.strip()
        print(path, end=' ')
        sr, audio = read(os.path.join(test_folder, path))
        vector = extract_features(audio, sr, 'test')

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

        winner = speakers[np.argmax(log_likelihood)]
        target_speaker = path.split('_')[0]

        if winner == target_speaker:
            precision += 1

        print("detected as", winner)
        time.sleep(1.0)

    precision = precision/num_files*100
    print('Precision: {}%'.format(precision))


def infer(test_folder, model_path):
    test_file = "./testing_set_addition.txt"
    file_paths = open(test_file, 'r')

    gmm_files = [os.path.join(model_path, fname) for fname in
                 os.listdir(model_path) if fname.endswith('.gmm')]

    # Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                in gmm_files]

    # Read the test directory and get the list of test audio files
    num_files = 0

    results = {}

    for path in file_paths:
        num_files += 1

        path = path.strip()
        print(path, end=' ')
        sr, audio = read(os.path.join(test_folder, path))
        vector = extract_features(audio, sr, 'test')

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

        winner = speakers[np.argmax(log_likelihood)]

        print("detected as", winner)

        results[path] = winner
        time.sleep(1.0)

    with open('./public-test/results.pkl', 'wb') as f:
        pickle.dump(results, f)

def predict_speaker(file,modelpath="./trained_models/"):
    gmm_files = [os.path.join(modelpath, fname) for fname in
                 os.listdir(modelpath) if fname.endswith('.gmm')]
    
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                in gmm_files]
    sr, audio = read(file)
    vector = extract_features(audio, sr, 'test')
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        gmm = models[i]  # checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
        
    winner = speakers[np.argmax(log_likelihood)]
    slashPos = [x.start() for x in re.finditer('/', winner)][-1]
    winner = winner[slashPos+1:]
    return winner

def infer_long_sample(long_test_folder, window_size=6000, temp_folder='./tmp'):
    long_test_files = os.listdir(long_test_folder)

    os.mkdir(temp_folder)

    for long_file in long_test_files:
        sound = AudioSegment.from_wav(
            os.path.join(long_test_folder, long_file))
        outputs = []
        for i in range(0, len(sound), window_size):
            tmp_sound = sound[i:i+window_size]
            tmp_sound.export(os.path.join(
                temp_folder, str(i)+'.wav'), format='wav')

        temp_files = os.listdir(temp_folder)
        for file in temp_files:
            outputs.append(predict_speaker(os.path.join(temp_folder, file)))
        shutil.rmtree(temp_folder)
        os.mkdir(temp_folder)

        vote_count = Counter(outputs)
        ans = vote_count.most_common()[0][0]

        print(long_file,'detected as',ans)
    shutil.rmtree(temp_folder)

def main():
    while True:
        choice = int(input(
            "\n 1.Train Model \n 2.Test Model \n 3.Infer \n 4.Infer on long audio\n"))
        if(choice == 1):
            print('Enter train folder: ', end='')
            train_folder = input()
            print('Enter model path: ', end='')
            model_path = input()
            train_file = "./training_set_addition.txt"
            write(train_folder, train_file)
            train_model(train_folder, model_path)
        elif(choice == 2):
            print('Enter test folder: ', end='')
            test_folder = input()
            print('Enter model path: ', end='')
            model_path = input()
            test_file = "./testing_set_addition.txt"
            write(test_folder, test_file)
            test_model(test_folder, model_path)

        elif(choice == 3):
            print('Enter test folder: ', end='')
            test_folder = input()
            print('Enter model path: ', end='')
            model_path = input()
            test_file = "./testing_set_addition.txt"
            write(test_folder, test_file)
            infer(test_folder, model_path)

        elif choice == 4:
            print('Enter test folder: ', end='')
            test_folder = input()
            infer_long_sample(test_folder)

        if(choice > 4):
            exit()


if __name__ == "__main__":
    main()
