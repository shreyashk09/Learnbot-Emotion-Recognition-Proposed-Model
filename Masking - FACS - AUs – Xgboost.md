# 
<pre> <h1>           Learnbot – Emotion Recognition       

                   (Idea Proposal)</h1> </pre>
status: model is set. writeup completed. model image onsertion left
## <pre>      MASKING - FACS - AUs – XGBOOST (2-fold and 1-full classifier)</pre>

### Aim Of Model:

Our aim is to recognize the emotions of humans examining their facial expressions and classifying them into various emotion classes.

The model should recognize the emotion in real – time and should be very fast.

The model should recognize emotion by performing moderate computations (no complex computations like DNNs) but should be very accurate.

Emotion of the person is to be recognized can possess various poses of the face and the architecture of face may differ from one another.

So, now our vision is clear and we focus on how to fulfill them.

### Features:

- **--** Derives face pose angle.
- **--** Frontal face view to Profile face view recognition.
- **--** Least preprocessing requirement.
- **--** Real time recognition.
- **--** Moderate computations performed.
- **--** Face skin texture examined
- **--** Fastest and Accurate Classifier: Xgboost.
- **--** 2-fold and 1-full classifier increase prediction accuracy.



### Hardware Requirements:

**Camera Module:**
- **--** 30 FPS
- **--** 3 MPixels

**CPU System:**
- **--** Processor 1.6 GHz base frequency.
- **--** 2 GB RAM
- **--** Quad Core
- **--** 32 GB Memory
- **--** Ports for camera module.

Minimum of above should be met for trained model to be executed.

### Software Requirements for Model:  

Opencv, Python, Xgboost, Numpy, Matplotlib, imutils, Sckilite,  Qt

### Runtime:

### Technologies and algorithms used:

HOG LB cascades, facial landmarks AAM, face Golden Ratio modeling, Action Units (AUs), Facial Action Coding System(FACS), Xgboost, etc.

##

### Action Units (AUs) and Facial Action Coding System (FACS): 

Emotion expressions on face is possible due to movement of various facial muscles. The Facial Action Coding System (FACS) refers to a set of facial muscle movements that correspond to a displayed emotion. It is an anatomical system for describing all observable facial movements into small units known as Action Units (AUs). AUs determines activeness of different sets of facial muscles at a time.

### Xgboost:

XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting algorithm. Algorithm is designed to work with numeric values only.

It is also known as &#39; **regularized boosting**&#39; technique because of its techniques to reduce overfitting problem automatically even with less training data.

The performance of xgboost is recorded to be more than other famous algorithms like DNN, random forest, etc. It is faster than other algorithms as it implements parallel processing.It helps to correct errors made by previously trained tree thus. Moreover, it allows to define custom optimization objectives and evaluation criteria. User can start training an XGBoost model from its last iteration of previous run, which makes training simpler.

## Flowchart:

  <div align='center'>
  <img src='M_Images/FACS_XGBOOST.jpg'  width='400px'>
  </div>
  
## Overview:

The model can be implemented for emotion recognition of human faces captured by Learnbot&#39;s camera module. The model computes moderate operations and performs in real time with sufficient frames per seconds. The model tries to attain the accuracy and focuses on only required essential parameters.  It is designed such that it performs the suitable processes in parallel in order to reduce the runtime.

During training phase, the model takes the inputs in the form of sequence of frames of each expression. This helps to classify the emotions accurately even at less expression features intensity. In runtime, the model is feed with captured images by Learnbot&#39;s camera module in real time.

The model is designed, as such don&#39;t require any sort of **pre-processing** and **face –alignment** as such. In order to reduce run-time and increase accuracy of further processes the image is grayscale and enhanced.

As the classification is carried into 5 prime emotions (and neutral), extracting key facial features is sufficient. To determine the feature vectors form the detected face frame, we plot **facial landmarks** over the face using AAM along with Lucas – Kanade (LK) algorithms. The facial landmarks are used to develop a **wireframe** across the face called as **mesh**. The details are extracted from the mesh as facial feature vectors. The mesh scientifically designed to fulfil multiple challenges.

The most important phase which gives model a new shape by increasing its accuracy, reducing runtime and makes it valid across different test cases, exceptions, etc. is facial feature extraction. It eliminates the need of pre-processing and facial alignment at vast level and brings uniformity across the images. For facial features angles between wireframes are considered, which don&#39;t change with change in face alignment, shape, size, rotation(3D), etc.

To learn from facial skin textures (wrinkles) we implement a simple **HOG-LB cascades** near key landmark points.

An approximate but accurate enough, neutral expression mesh is generated for the same face detected using the &quot; **Golden Ratios**&quot;. The driving forces are calculated between the &quot;expressive mask&quot; and &quot;neutral mask&quot;. The driving forces as well as the independent facial feature of expressive frame is combined in a list to form Feature Vectors List. The list is traversed across the following Xgboost Classifier to get the predicted emotion.

These feature vectors can be clustered into groups representing various unique face actions called as &quot; **Action units**&quot;. The combinations of these Action Units predict different emotion emotion.

 The feature vectors are transformed to the **first level Xgboost** to determine the Action Units (AUs) of the corresponding expression sequences. Finally, the detected AUs are inputed into the **second level Xgboost** for facial expressions classification.

In parallel, another **Xgboost** model will classify emotion directly from feature vectors. The fused result gives us accurate predication.

The model is trained with sequence of expression frames starting with less expressive to most expressive. Thus, face with any intensity of expression can be classified easily.

## Methodology:

1. Image frame is captured by Learnbot camera in real – time.
2. Cascades are used for detecting faces and cropping (into 64 x 64).
    (frontal to profile view in 3D)
3. The face is to grayscale and applying cv.createCLAHE tuner.
4. Facial Landmarks are positioned across the face using AAM and further dynamically re - localized using Lucas – Kanade method.
5. Face is masked with a wireframes and feature vectors are extracted.
6. Parallel to step 5, a mask for neutral expression is generated based on &quot;Golden Formula&quot; and feature vector are extracted. (Pose angle is estimated)
7. Multi – Cascade detectors are implemented in parallel to step 6 and 7
8. Force vectors are extracted from the masks and appended to the feature vectors list.
9. 1st fold Xgboost AU classifiers model (FACS) is used to determine probabilities of active Action Units.
10. 2nd fold Xgboost emotion classifier model is implemented to classify probability of emotion predicted based upon AUs activation probability.
11. Parallel, another Full-Emotion classifier is implemented to directly compute feature vectors to classify predicted emotions probability.
12. Probability results from both the models are combined to give a precise predicted emotion probability and label.
13. The result is then dynamically represented on screen graphically with necessary parameters.

### Face Detection and Ranking:

The captured images using Learnbot&#39;s camera may contain one or more faces at a time. The whole frame is investigated using Cascades for face detection from front view to profile view. The face – rectangles are then cropped are reshaped into 64 x 64 squares.

The face - rectangles are of different sizes and located at different location across the frame based upon the position of human in front of Learnbot in 3D. These faces are assigned ranks as follows:

- **--** Ranking based upon size of face-rectangle (measured diagonally) in descending order i.e., larger rectangle is ranked prior to smaller ones. (seems to be near and more interactive)
- **--** Ranking based upon displacement form center. Person at center should be prioritized.
  
  <div align='center'>
  <img src='M_Images/rank.png'  width='450px'>
  </div>

## Pre - Processing:

The face frame of 64 X 64 is then converted into grayscale from RGB form. It is then normalized and edges are sharpened by tuning opencv inbuilt function createCLAHE.

This is performed so that it reduces the runtime during skin texture detection, and positioning of facial landmarks.

The model shows no requirement of any heavy or additional preprocessing and face alignments, it works deals with the challenges more efficiently. (discussed in Masking).
Preprocessing: original image-->gray image-->clahe image
 <div align='center'>
  <img src='M_Images/pp/original.png'  width='250px'>
  <img src='M_Images/pp/gray.png'  width='250px'>
  <img src='M_Images/pp/clahe.png'  width='250px'>
  </div>
  
## Facial Landmarks:

Upon minimum pre-processing, the image becomes suitable to derive facial landmarks which could be further used to extract facial features.

The facial landmarks are accumulated based upon pose of face in XZ – plane and YZ – plane. So, we define a **threshold angle (alpha)** about which the face is distinguished as &quot; **Full-Face**&quot; view and &quot; **Half- Face**&quot; view. In case of Half – Face view only one half of face is used to determine the expression. The other half is either incapable of determining the exact landmark positions and distinguish among them or is not visible.

The initial frame landmarks are estimated using Active Appearance Model (AAM). An AAM face model consists a shape model and a texture model. The fitting procedure iteratively adjust the model until satisfy.

Further in consecutive frames Lucas–Kanade (LK) optical flow tracker can be used by estimating the displacements of the feature points.
  <div align='center'>
  <p><img src='M_Images/alp2.png'  width='275px'>
  </p>
 <p> <img src='M_Images/alp1.png'  width='500px'>
  </p>
  <p>
  <img src='M_Images/alp3.png'  width='275px'>
  </p>
  </div>

## Face Masking:

**(MOST IMPOTANT PHASE) (motivation:**  **Candide Wireframe Model)**

The facial landmarks are very sensitive to the movements of facial muscles. Each person has unique face structure and gives vivid poses at different point of time. To study the pattern followed by vector displacement of these facial landmarks for any given expression, we generate mask over the face such that details extracted from them can be compared across the frames.

The architecture of the mask is motivated from **Candide Wireframe Model**. The driving forces exerted during change in expression intensities are the distance between nodes and **angles at intersections in wireframe**.

The angles at intersections don't change with change orientation of face about z-axis or with any change in size (aspect ratio is constant) of the face frame for any given expression intensity.

Inorder to find the change in these driving forces we subtract the present frame mask&#39;s details with mask&#39;s details of neutral expression. The result is change in angles and displacements of nodes.

To make it possible we first need to normalize the face within the frame and normalize faces across the frames.

**Normalization of feature vectors within the face frame:**

Face pose at different angles in 3D, this leads to compression and expansion of angles within a face according to degree of inclination.
  <div align='center'>
  <img src='Images/database.png'  width='400px'>
  </div>
Example:

Let, a person looks diagonally at an angle towards right-bottom. The angles between wireframes of left eyebrow will be relatively greater than right eyebrow&#39;s for same expression as compared to normal pose angles of same expression.

To avoid such variation within a frame, we can apply face-alignment to center. But, face-alignment of each frame will consume time (3D matrix) and don&#39;t we gives required solution. **So, we take relative variation of angles within a feature for each feature within a frame.** This solves problem of any pose angles.

**Normalization of feature vector across the faces frames:** 

Every person has variations in the structural positioning, shape and size of the features. To determine the expression on face of each individual we should know their neutral expression dimensions, so that we can subtract expressive frame from neutral frame of each person to direction of driving forces.

The neutral frame is generated using Golden Formula. The whole mask generation for neutral expression is not required. We just need inner approximate aspect ratio(angles) of each feature of a face. And this can be computed very fast

using dimensions of face in the frame i.e, distance between an eye center and mid-point of both the eyes, distance between mid-point of eyes and nose center and pose angle.

Only, the landmarks for eyebrows and lips are plotted using Golden Ratio, that too separately. So, there is no constrain of overlapping neutral mask generated with the actual person&#39;s mask.

Thus, masks are appropriate for classification into different expression classes.

**FULL MASK:**

If the front view of face in the frame is visible clearly i.e., pose angle is less than or equal **threshold angle (alpha),** we use our full mask.

**HALF MASK:**

If the front view of face in the frame is not visible clearly i.e., pose angle is greater than **threshold angle (alpha),** we use our half mask. The feature extracted from one half mask (suppose left view of face) are replicated same as for the other half. We calculate only for visible side and consider the same for both sides.

**Golden Ratio:**

Golden Ratio is the most perfect fitting number, i.e, it fits shapes reduced with this ratio upto infinity within a single frame. An ideal face is said to have feature positioned relatively based on this Golden Ratio and it derivatives.

Even with a perfectly proportioned face though, there are endless variations in the shapes and sizes of each facial feature (eyes, eyebrows, lips, nose, etc.) that gives rise to the distinctive appearance of each person and provide for endless variations.

But, in our case **we don&#39;t need any exact shape, size or even relative position of features.** We just need a approximate constant value that can subtracted from the expression mask to normalise across frames which varies negligibly with across the frames.

It is just to bring extreme different face architecture into comparable range.    The relative position is not required between them, we just require inner approximate aspect ratio(angles) of a feature. Inter features are not compared. And moreover, these angles of a feature are taken as probability (angle/average(angles)), so they are again normalised within a frame itself.



## Multi - Cascade Detectors:

Some of the most important facial muscles show their movements in form of skin texture. These are the case where landmarks may not show drastic movements but the skin texture is enough to determine face expression features.

To determine these skin textures at different regions of the face, various &quot;Cascades&quot; can be used at different regions over the face.

The cascades are trained using opencv functions **opencv\_haartraining and opencv\_traincascade.** They apply **HOG – LB** filters.
  <div align='center'>
  <img src='M_Images/cas_n.jpg'  width='200px'>
  </div>
Each Cascade's region of implementation is determined by nearest landmarks&#39; locations respectively.

The outcome of cascading is defined as expression feature intensity and represented in form of binary {0,1}.
 
  <div align='center'>
  <img src='M_Images/csd_ha.jpg'  width='200px'>
  <img src='M_Images/csd_sa.png'  width='200px'>
  <img src='M_Images/csd_fe.jpg'  width='200px'>
  </div>
  <div align='center'>
  <img src='M_Images/csd_an.jpg'  width='200px'>
  <img src='M_Images/csd_di.jpg'  width='200px'>
  </div>

## AUs Xgboost (xgb\_1\_1):

The clusters of various face expression feature vectors can be formed to define a cumulative muscle actions called as Action Units. The classification is computed using xgboost algorithm. 17 AUs are recognized, they are AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU12, AU14, AU15, AU17, AU20, AU23, AU24, AU25, AU27 and AU38

A xgboost model is created where inputs are the face feature vectors, and the output is classified Action Units. The model tree is trained such that the combinations of definite range of varying feature intensities are clustered under each Action Units. The model is called FACS.

We get required active AUs probabilities, classified based upon given feature vectors.

## Xgboost Classifier(xgb\_1\_2):

The second fold Xgboost model is based upon activeness of various AUs to classify an emotion. The classification is performed, by feeding probabilities of AUs active in a frame and computed output is the probabilities of emotions possibly shown.

## Full – Xgboost (xgb\_2):

Due to less accurate prediction of activeness of AUs, the final classification may lead to wrong predictions. Thus to avoid wrong prediction and get accurate probability distribution across emotions, we implement a model computing emotion classification directly based upon feature vectors. The model runs in parallel to 2-fold Xgboost model.

## Fused Result:

The output, probabilities of emotions from both the model are combined based upon strength of each emotion relative to others.

## Result Presentation:
  
  <div align='center'>
  <video src='M_Images/output.mov'  width='400px'>
  </div>


## Steps towards accuracy:

- **--** Similar face alignment across frames
- **--** Proper selection of feature vectors
- **--** Perfect masking
- **--** Accuracy can be gained further by tuning of the Xgboost models

**Thank You**

**References:**



** **
