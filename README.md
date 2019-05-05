# Music Genre CNN Classifier
A two-class music genre classifier based on CNN (Convolution Neuron Network). 
<br>
<br>

## What does this project do?
1\. Extract Mel-spectrogram as the training set from two groups of songs with different genre 

2\. Train the CNN model (on your PC or Google TPU)

3\. Evaluate the CNN model using the validation set

4\. Visualize each kernel (For learning purpose)

> NOTE that this project takes the classification of prograssive rock and non-prograssive rock musics as an example. Users can feel free to change them to other music genres.

```
code
  |
  |-- generatePatterns.py
  |-- trainModel.py
  |-- evaluateModel.py
  |-- toolbox
        |-- featureExtractTool.py (helper functions)
        |-- evaluateModelTool.py  (helper functions)
        |-- trainModelTPU.py   (optional)
        |-- generateTestset.py (optional)
        |-- visualizeKernel.py (optional)
 
training songs
  |
  |-- class1
  |-- class2

validation songs
  |
  |-- class1
  |-- class2
```


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

2\. Feature extraction tool
```
pip3 install librosa
```

3\. Model training tools
```
pip3 install keras tensorflow
```
<br>


## Extract Mel-spectograms from Songs as Training set
> Skip this step by downloading default training sets (Class1: [Prop songs](https://drive.google.com/file/d/1ggK2dHxsiVPdFeDtgIUEir0KDj82HOgF/view?usp=sharing) and Class2: [Non-prog songs](https://drive.google.com/file/d/1ZiCTaWlbeo702_4A1TjgWVc_grOu7sml/view?usp=sharing))

1\. Put two groups of songs (.mp3) into `./training song/class1` and `./training song/class2`

2\. Run 
```
python3 generatePatterns.py
```  
3\. After extraction, user will get training sets `Class1Patterns.pkl` and `Class2Patterns.pkl`

<br>


## Train CNN Model 
 > Skip this step by using the default trained [model](https://drive.google.com/file/d/11GtfNa6Lzm09ifd6Shu6CUv61kW6wJXB/view?usp=sharing) to try validation set
 
1\. Run
```
python3 trainModel.py
```  
2\. After training, user will get `cnnModel.h5` and `Scalers.sav`
> Scalers are used for training set normalization (make values in each channel between -1 and +1). It is necessary to use the same scalers to normalize validation songs.
<br>


## Train CNN Model using TPU (Optional)
1\. Create a notebook `.ipynb` on [Colab](https://colab.research.google.com/)

2\. Copy the code in `trainModelTPU.py` to notebook

3\. Upload `class1Patterns.pkl` and `class2Patterns.pkl` to `./Colab Notebooks` on your Google drive

4\. Run and get `cnnModel.h5` on the Google drive
<br>
<br>


## Validate CNN Model
1\. Put two groups of songs (.mp3) into `./validation songs/class1` and `./validation songs/class2`
> Songs for validation should be different from the ones for training

2\. Run
```
python3 evaluateModel.py
```
3\. See results in console
<br>
<br>

## Visualize CNN Kernels (Optional)
This feature is for learning purpose. This part of the code is referenced from an amazing post [Visualization of Filters with Keras (Yumi's Blog)](https://fairyonice.github.io/Visualization%20of%20Filters%20with%20Keras.html)
```
python3 visualizeKernel.py
```
<br>
<br>

Feel free to contact me if you find a bug or have any question :D
