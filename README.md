# AGD-Project
## Objective :
To build a python project of gender and age detector that can approximately  guess the gender and age of the person in the image using OpenCV on the  Adience dataset.

## 1. Abstract :
In this Python Project, we will approximately identify the gender and age of a 
person from the image. We will use the models trained by Tal Hassner and Gil 
Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the 
predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 
– 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax 
layer). It is very difficult to accurately guess an exact age from the image 
because of factors like makeup, lighting, obstructions, and facial expressions. 
And so, we make this as a classification problem instead of making it one of 
regression problem.

## 2. Terminologies Used :
### 2.1 Computer Vision :
Computer Vision is the field of study that enables computers to see and 
identify digital images and videos as a human would. The challenges it faces 
largely follow from the limited understanding of biological vision. Computer 
Vision involves acquiring, processing, analyzing, and understanding digital 
images to extract high-dimensional data from the real world in order to 
generate symbolic or numerical information which can then be used to make 
decisions. The process often includes practices like object recognition, video 
tracking, motion estimation, and image restoration.
### 2.2 OpenCV :
OpenCV is short for Open Source Computer Vision. Intuitively by the name, it 
is an open-source Computer Vision and Machine Learning library. This library 
is capable of processing real-time image and video while also boasting 
analytical capabilities. It supports the Deep Learning frameworks, Tensor 
flow, Caffe, and PyTorch.
To go about the python project, we’ll:
* Detect faces
* Classify into Male/Female
* Classify into one of the 8 age ranges
* Put the results on the image and display it

## 3. Prerequisites :
Additional python libraries required for this project are as below :
* OpenCV (cv2) 
Other packages we’ll be needing come as part of the standard Python library.

## 4. Contents of the Project :
* opencv_face_detector.pbtxt
* opencv_face_detector_uint8.pb
* age_deploy.prototxt
* age_net.caffemodel
* gender_deploy.prototxt
* gender_net.caffemodel
* a few pictures to try the project on
For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it 
holds the graph definition and the trained weights of the model. We can use 
this to run the trained model. And while a .pb file holds the protobuf in binary 
format, one with the .pbtxt extension holds it in text format. These are 
TensorFlow files. For age and gender, the .prototxt files describe the network 
configuration and the .caffemodel file defines the internal states of the 
parameters of the layers.

## 5. Steps for the Gender Age Detection Project :
* First we have to clone our repository for the files and images we need.
* Using ‘cd’ command we change our directory to AGD folder.
* Download pretrained data and unzip it to make availability of some 
caffe models and deploy prototxts in it.
* Then we need to import python libraries like OpenCV, math etc..,
* For face, age, and gender, initialize protocol buffer and model.
* Initialize the mean values for the model and the lists of age ranges and 
genders to classify from.
* Let’s make a call to the getFaceBox() function with the faceNet and 
frame parameters, and what this returns, we will store in the names face
and bboxes. 
Here, net is faceNet- this model is the DNN Face Detector .
* Create a shallow copy of frame and get its height and width.
* Create a blob from the shallow copy.
* Set the input and make a forward pass to the network.
* bboxes is an empty list now.Define the confidence (between 0 and 
1). Wherever we find the confidence greater than the confidence 
threshold, which is 0.7, we get the x1, y1, x2, and y2 coordinates 
and append a list of those to bboxes.
* Then, we put up rectangles on the image for each such list of 
coordinates and return two things: the shallow copy and the list of 
bboxes.
* But if there are indeed bboxes, for each of those, we define the face, 
create a 4-dimensional blob from the image. In doing this, we scale it, 
resize it, and pass in the mean values.
* We feed the input and give the network a forward pass to get the 
confidence of the two class. Whichever is higher, that is the gender of 
the person in the picture.
* Then, we do the same thing for age.
* At last the input image is read, sends to the age gender detector function 
and display it with imshow().

## 6. Project File Structure :
* Uploading the Data
* Importing and downloading the pre-trained models
* Importing required python libraries
* Functions for Detecting Faces
* Defining the variables of weights and architectures for face, age, and 
gender detection models
* Loading the Models
* Framing , Age-Gender detecting functions
* Function to read , detect , and output the result

## Conclusion: 
We tackled the estimation of age and gender detection of real-world face 
images. We posed the task as a detecting problem and as such, train the model 
with a estimation-based loss function as training targets. Our proposed model 
is originally pretrained on age and gender labelled large-scale Adience 
dataset. Finally, we use the original dataset (Adience benchmark of unfiltered 
faces for age and gender detection) to fine-tune this model. The robust image 
preprocessing algorithm, handles some of the variability observed in typical 
unfiltered real-world faces, and this confirms the model applicability for age 
and gender detection. Finally, we successfully done with the detecting 
accuracy on Adience dataset for age and gender; our proposed method 
achieves the state-of-the-art performance, in both age and gender detection, 
significantly outperforming the existing models.
For future works, we will consider a deeper CNN architecture and a more 
robust image processing algorithm for exact age and gender estimation. Also, 
the apparent age estimation of human’s face will be interesting research to 
investigate in the future
