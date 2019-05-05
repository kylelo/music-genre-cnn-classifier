'''
Evaluate the performace (accuracy) of the trained model
'''

import sys
sys.path.insert(0, './toolbox')

#import keras
import pickle

# Tensorflow
import tensorflow as tf
from tensorflow.python.keras import models

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from evaluateModelTool import test_model, test_model_use_pkl

# Define pattern size
NUM_OF_SAMPLE_PER_WINDOW = 43
NUM_OF_OVERLAP_SAMPLE = 0

# Set prog songs
PATH_TO_CLASS_1_VALIDATION_SONGS = "../validation songs/class1"
CLASS_1_LABEL = "prog"

# Set nonprog songs
PATH_TO_CLASS_2_VALIDATION_SONGS = "../validation songs/class2"
CLASS_2_LABEL  = "nonprog"


# Read model and scaler from files
t_Model = tf.keras.models.load_model('cnnModel.h5')
t_Scalers = pickle.load(open('Scalers.sav', 'rb'))

# Convert label to numbers (binary number in two class classification)
t_Encoder = LabelEncoder()
t_Labels = t_Encoder.fit_transform([CLASS_1_LABEL, CLASS_2_LABEL])
print("Label:", CLASS_1_LABEL, CLASS_2_LABEL, "corresponding to", t_Labels)

test_model(t_Model, t_Scalers, PATH_TO_CLASS_1_VALIDATION_SONGS, t_Labels[0],
                               NUM_OF_SAMPLE_PER_WINDOW, NUM_OF_OVERLAP_SAMPLE)
test_model(t_Model, t_Scalers, PATH_TO_CLASS_2_VALIDATION_SONGS, t_Labels[1],
                               NUM_OF_SAMPLE_PER_WINDOW, NUM_OF_OVERLAP_SAMPLE)



################################################################################
# Remove comment of following code to use extracted validation set             #
# First go to toolbox folder run generate testset to get pkl                   #
################################################################################

#t_TestClass1SongList = pickle.load(open('./toolbox/class1PatternsValidation.pkl', 'rb'))
#test_model_use_pkl(t_Model, t_Scalers, t_TestClass1SongList, t_Labels[0])

#t_TestClass2SongList = pickle.load(open('./toolbox/class2PatternsValidation.pkl', 'rb'))
#test_model_use_pkl(t_Model, t_Scalers, t_TestClass2SongList, t_Labels[1])
