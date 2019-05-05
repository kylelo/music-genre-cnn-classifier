'''
Generate melspectrogram training set from all class1 and class2 songs in dir.
Each pattern is a NUM_OF_SAMPLE_PER_WINDOW by 128 (Dim of a sample) matrix.
(This file generates two .pkl file, class1 and class2 respectively)
'''

import sys
sys.path.insert(0, './toolbox')

from featureExtractTool import generate_patterns_for_all_songs_in_dir

# Define pattern size
NUM_OF_SAMPLE_PER_WINDOW = 43
NUM_OF_OVERLAP_SAMPLE = 0

# Set prog songs
NAME_OF_CLASS_1_PATTERNS = "class1Patterns.pkl"
PATH_TO_CLASS_1_SONG_DIR = "../training songs/class1"
CLASS_1_LABEL = "prog"

# Set nonprog songs
NAME_OF_CLASS_2_PATTERNS = "class2Patterns.pkl"
PATH_TO_CLASS_2_SONG_DIR = "../training songs/class2"
CLASS_2_LABEL = "nonprog"


# Get patterns for prog songs
generate_patterns_for_all_songs_in_dir( PATH_TO_CLASS_1_SONG_DIR,
                                        CLASS_1_LABEL,
                                        NUM_OF_SAMPLE_PER_WINDOW,
                                        NUM_OF_OVERLAP_SAMPLE,
                                        NAME_OF_CLASS_1_PATTERNS )


# Get patterns for nonprog songs
generate_patterns_for_all_songs_in_dir( PATH_TO_CLASS_2_SONG_DIR,
                                        CLASS_2_LABEL,
                                        NUM_OF_SAMPLE_PER_WINDOW,
                                        NUM_OF_OVERLAP_SAMPLE,
                                        NAME_OF_CLASS_2_PATTERNS )
