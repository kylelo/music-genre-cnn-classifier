'''
Toolbox for extracting the segment of Mel-spectrogram from song as patterns.
'''

import os
import math
import pickle
import librosa
import numpy as np



'''
Generate a dict contains a list and each entry stores the array of all patterns for a song
'''
def generate_test_set(a_PathToSongDir, a_Label, a_SizeOfWindow, a_NumOfOverlapSample, a_NameOfOutput):

    # Get name of each song and save as list
    t_AllFiles = os.listdir(a_PathToSongDir)

    # Init list to store the dictionart of each song
    t_AllSongs = []

    # Iterate through all songs
    for i, t_SongName in enumerate(t_AllFiles):

        print("Processing...", t_SongName)

        if t_SongName.endswith(".mp3"):

            # Get full path for single song
            t_PathToSingleSong = f'{a_PathToSongDir}/{t_SongName}'
            t_Patterns, t_Labels, = generate_patterns_for_one_song(t_PathToSingleSong,
                                        a_Label, a_SizeOfWindow, a_NumOfOverlapSample)
            # save dict for single song into list
            t_AllSongs.append({"Patterns" : t_Patterns, "Labels" : t_Labels})

        print("Label:", a_Label, " No.", i+1, "/", len(t_AllFiles), "  {0:.2%}".format((i+1)/len(t_AllFiles)))

    save_as_pkl_file(t_AllSongs, a_NameOfOutput)



'''
Given a path to folder, generate patterns from all the songs in it
'''
def generate_patterns_for_all_songs_in_dir(a_PathToSongDir, a_Label, a_SizeOfWindow, a_NumOfOverlapSample, a_NameOfOutputPatterns):

    # Get name of each song and save as list
    t_AllFiles = os.listdir(a_PathToSongDir)

    # Init array and list to store all patterns and labels
    t_AllPatterns = np.array([])
    t_AllLabels = []

    # Iterate through all songs
    for i, t_SongName in enumerate(t_AllFiles):

        print("Processing...", t_SongName)

        if t_SongName.endswith(".mp3"):

            # Get full path for single song
            t_PathToSingleSong = f'{a_PathToSongDir}/{t_SongName}'

            t_Patterns, t_Labels, = generate_patterns_for_one_song(t_PathToSingleSong,
                                        a_Label, a_SizeOfWindow, a_NumOfOverlapSample)

            t_AllPatterns = np.vstack([t_AllPatterns, t_Patterns]) if t_AllPatterns.size else t_Patterns
            t_AllLabels = t_AllLabels + t_Labels;

        print("Label:", a_Label, " Processing song No.", i+1, "/", len(t_AllFiles), "  {0:.2%}".format((i+1)/len(t_AllFiles)))

    # Save to pkl file
    t_Dict = {"Patterns" : t_AllPatterns, "Labels" : t_AllLabels}
    save_as_pkl_file(t_Dict, a_NameOfOutputPatterns)



'''
Get all patterns for one song
'''
def generate_patterns_for_one_song(a_PathToSingleSong, a_Label, a_SizeOfWindow, a_NumOfOverlapSample):

    t_RawSong, t_SR = get_raw_song(a_PathToSingleSong)
    t_Mel = extract_mel(t_RawSong, t_SR)
    t_Patterns, t_Labels = divide_features(t_Mel, a_Label, a_SizeOfWindow, a_NumOfOverlapSample)
    
    return t_Patterns, t_Labels



'''
Get raw file of a song
'''
def get_raw_song(a_PathToSingleSong):

    # Init raw song. offset cut the head of the song
    t_RawFullSong, t_SR = librosa.load(a_PathToSingleSong, offset=10, mono=True, sr=22050)

    # Trim the beginning and ending silence
    t_RawSong, t_Idx = librosa.effects.trim(t_RawFullSong)

    return t_RawSong, t_SR



'''
extract melspectrogram from a raw song file
'''
def extract_mel(a_RawSong, a_SR):

    t_Mel = librosa.feature.melspectrogram(y=a_RawSong, sr=a_SR, fmax=8000)

    return t_Mel



'''
Use window to divide a feature of all song into many small patterns
'''
def divide_features(a_Features, a_Label, a_SamplesPerWindow, a_OverlapSamples):

    # Get length of feature
    t_Length = a_Features.shape[1]

    # Get bins of feature
    t_Bins = a_Features.shape[0]

    # Get total number of patterns will generate in this features
    t_StepSize = a_SamplesPerWindow - a_OverlapSamples
    t_NumOfPatterns = math.floor((t_Length - a_SamplesPerWindow) / t_StepSize)

    # Init containers
    t_Patterns = np.zeros((t_NumOfPatterns, a_SamplesPerWindow, t_Bins))
    t_Labels = []

    # Break features down into patterns and record labels
    for i in range(0, t_NumOfPatterns):
        t_Patterns[i,:,:] = a_Features[:, i*t_StepSize : i*t_StepSize + a_SamplesPerWindow].transpose()
        t_Labels.append(a_Label)

    return t_Patterns, t_Labels



'''
Save file containing patterns and label into pkl file
'''
def save_as_pkl_file(a_File, a_Filename):
    with open(a_Filename, 'wb') as t_PKLFile:
        pickle.dump(a_File, t_PKLFile)
