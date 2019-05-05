"""
Generate melspectrogram "Validation" set from all class1 and class2 songs in dir
Each pattern is a NUM_OF_SAMPLE_PER_WINDOW by 128 (Dim of a sample) matrix.
(This file generates two .pkl file, prog and nonprog respectively)

This file is used to

"""

from featureExtractTool import generate_test_set

# Define pattern size
NUM_OF_SAMPLE_PER_WINDOW = 43
NUM_OF_OVERLAP_SAMPLE = 0

# Set prog songs
NAME_OF_PROG_PATTERNS = "class1PatternsValidation.pkl"
PATH_TO_PROG_SONG_DIR = "../../validation songs/class1"
PROG_LABEL = "prog"

# Set nonprog songs
NAME_OF_NONPROG_PATTERNS = "class2PatternsValidation.pkl"
PATH_TO_NONPROG_SONG_DIR = "../../validation songs/class2"
NONPROG_LABEL = "nonprog"


# Get patterns for prog songs
generate_test_set( PATH_TO_PROG_SONG_DIR,
                   PROG_LABEL,
                   NUM_OF_SAMPLE_PER_WINDOW,
                   NUM_OF_OVERLAP_SAMPLE,
                   NAME_OF_PROG_PATTERNS )



# Get patterns for nonprog songs
generate_test_set( PATH_TO_NONPROG_SONG_DIR,
                   NONPROG_LABEL,
                   NUM_OF_SAMPLE_PER_WINDOW,
                   NUM_OF_OVERLAP_SAMPLE,
                   NAME_OF_NONPROG_PATTERNS )
