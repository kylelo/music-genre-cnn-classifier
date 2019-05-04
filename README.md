# music-genre-cnn-classifier
A two-class music genre classfier based on CNN (Convolution Neuron Network). 


## What does this project do?
1. Extract Mel-spectrogram as the training set from two group of songs with different genre 
2. Train the CNN model
3. Validate the CNN model using test set
4. Visualize each kernels (For learning purpose)

> NOTE that this project takes the classification of prograssive rock and non-prograssive rock musics as an example. Users can feel free to change them to other music genres.

## Prerequisites

**1. Python3**

Linux
```
sudo apt-get update
sudo apt-get install python3.6
```

MacOS
```
brew install python3
```

**2. Feature Extracion Tool**
```
pip3 install librosa
```

**3. Model Training Tools**
```
pip3 install keras tensorflow
```

## Extract Mel-spectogram from Songs as Training set
> Skip this step if user wants to use the default training set (`progPatterns.pkl` and `nonprogPatterns.pkl`)

1\. Put two groups of songs into `./training song/class1` and `./training song/class2`
2\. Run 
 ```
 python3 generatePatterns.py
 ```  
3\. After extraction, user will get training sets `Class1Patterns.pkl` and `Class1Patterns.pkl`

## Train CNN model 
 




(trainModel.py)
