import os
import shutil
from tqdm import trange


def move_files(folder, dest_folder):
    speaker_ids = os.listdir(folder)

    for i in trange(len(speaker_ids)):
        speaker = speaker_ids[i]
        vers = os.listdir(os.path.join(folder, speaker))
        for ver in vers:
            files = os.listdir(os.path.join(folder, speaker, ver))

            i = 0
            for file in files:
                new_file_name = str(speaker)+'-sample'+str(i)+'.m4a'
                shutil.copyfile(os.path.join(folder, speaker, ver,
                                             file), os.path.join(dest_folder, new_file_name))
                i += 1

def convert(folder,dest_folder):

    files = os.listdir(folder)
    for i in trange(len(files)):
        file = files[i]
        if file[-3:] == 'm4a':
            command = 'ffmpeg -i ' + \
                os.path.join(folder, file)+' ' + \
                os.path.join(dest_folder, file[:-3]+'wav')
            os.system(command)

def write(folder,text_file):
    wav_files=os.listdir(folder)
    with open(text_file,'w',encoding='utf8') as f:
        for file in wav_files:
            f.write(file+'\n')


if __name__ == "__main__":

    train_folder='./timit_training_set'
    train_text_file='training_set_addition.txt'
    write(train_folder,train_text_file)

    test_folder='./testing_set'
    test_text_file='testing_set_addition.txt'
    write(test_folder,test_text_file)
