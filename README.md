# Music Genre CNN Classifier
A two-class music genre classfier based on CNN (Convolution Neuron Network). 
<br>
<br>

## What does this project do?
1\. Extract Mel-spectrogram as the training set from two group of songs with different genre 

2\. Train the CNN model

3\. Validate the CNN model using test set

4\. Visualize each kernels (For learning purpose)

> NOTE that this project takes the classification of prograssive rock and non-prograssive rock musics as an example. Users can feel free to change them to other music genres.
<br>

## Prerequisites

1\. Python3

Linux
```
sudo apt-get update
sudo apt-get install python3.6
```

MacOS
```
brew install python3
```

2\. Feature extracion tool
```
pip3 install librosa
```

3\. Model training tools
```
pip3 install keras tensorflow
```
<br>


## Extract Mel-spectogram from Songs as Training set
> Skip this step if user wants to use the default training set (`progPatterns.pkl` and `nonprogPatterns.pkl`)

1\. Put two groups of songs into `./training song/class1` and `./training song/class2`

2\. Run 
```
python3 generatePatterns.py
```  
3\. After extraction, user will get training sets `Class1Patterns.pkl` and `Class1Patterns.pkl`
<br>
<br>

## Train CNN Model 
 > Skip this step if user wants to use the default `progNonprogModel.h5` to try validation set
 
1\. Run
```
python3 trainModel.py
```  
2\. After training, user will get `cnnModel.h5`
<br>
<br>

## Validate CNN Model
1\. Put two groups of songs into `./validation songs/class1` and `./validation songs/class2`
> Songs for validation should be different from the ones for training

2\. Run
```
python3 evaluateModel.py
```
3\. See results in console
<br>
<br>

## Visualize CNN Kernels
This feature is for learning purpose. This part of the code is referenced from an amazing post [Visualization of Filters with Keras (Yumi's Blog)](https://fairyonice.github.io/Visualization%20of%20Filters%20with%20Keras.html)
```
python3 visualizeKernel.py
```
<br>
<br>

Feel free to contact me if you find a bug or have any question :D
