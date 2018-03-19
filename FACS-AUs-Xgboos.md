## Learnbot – Emotion Recognition

## (Idea Proposal)

##

### FACS - AUs – Xgboost (2-fold and 1-full classifier)

##

**Action Units (AUs) and** **Facial Action Coding System (FACS): **

Emotion expressions on face is possible due to movement of various facial muscles. The Facial Action Coding System (FACS) refers to a set of facial muscle movements that correspond to a displayed emotion. It is an anatomical system for describing all observable facial movements into small units known as Action Units (AUs). AUs determines activeness of different sets of facial muscles at a time.

**Xgboost:**

XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting algorithm. Algorithm is designed to work with numeric values only.

It is also known as &#39; **regularized boosting**&#39; technique because of it&#39;s techniques to reduce overfitting problem automatically even with less training data.

The performance of xgboost is recorded to be more than other famous algorithms like DNN, random forest, etc. It is faster than other algorithms as it implements parallel processing.It helps to correct errors made by previously trained tree thus. Moreover, it allows to define custom optimization objectives and evaluation criteria. User can start training an XGBoost model from its last iteration of previous run, which makes training simpler.

**Overview:**

The FACS - AUs – Xgboost algorithms can be implemented for emotion recognition over camera captured facial images. Initially, the images are pre-processed using various techniques such that all the face images with various orientations and poses are arranged with a similar face - orientation.

As the classification is carried into 5 prime emotions ( and neutral), extracting key facial features are sufficient. These facial features are in form of face-landmarks(Masks) and skin texture representing the various facial muscles movements. Mask over face is created to vectorise the feature defined by the landmarks.  All these facial features are vectorised to gain similar representations across the image frames. These feature vectors can be clustered into groups representing various unique face actions called as &quot;Action units&quot;. The combinations of these Action Units predict different emotion emotion.

 The feature vectors are transformed to the **first level Xgboost** to determine the Action Units (AUs) of the corresponding expression sequences. Finally, the detected AUs are inputed into the **second level Xgboost** for facial expressions classification.

In parallel, another **Xgboost** model will classify emotion directly from feature vectors. The fused results gives us accurate predication.

**Advantages:**

- **--**

**Hardware Requirements:**

**Flowchart:**

**Methodology:**

1. Image frame is captured by Learnbot camera in real – time.
2. Haar Cascade xmls are used for detecting faces and cropping (into 64 x 64 x 3). (haarcascade\_frontalface\_default.xml)
3. The face pose in the image is aligned about x,y and z – axis and pivoted about fixed eye positions
4. Facial Landmarks are positioned across the face using AAM and further dynamically re - localized using Lucas – Kanade method.
5. Multi – Cascade detectors are implemented
6. Face is masked with feature vectors.
7. 1
# st
 fold Xgboost AU classifiers model (FACS) is used to determine probabilities of active Action Units.
8. 2
# nd
 fold Xgboost emotion classifier model is implemented to classify probability of emotion predicted based upon AUs activation probability.
9. Parallel, another Full emotion classifier is implemented to directly compute feature vectors to classify predicted emotions probability.
10. Probability results from both the models are combined to give a precise predicted emotion probability and label.
11. The predicted emotion and its probability is represented on-screen graphically.

**Face Detection and Ranking:**

The captured images using Learnbot&#39;s camera may contain one or more faces at a time. The whole frame is investigated using Haarcascade cascade for face detection. The detected face angle lies between -30 to 30 degree approximately. The face – rectangles are then cropped are reshaped into 64 x 64 square.

The face - rectangles are of different sizes and located at different location across the frame based upon the position of human in 3D in front of Learnbot. These faces are ranked for representation as follows:

Ranked based on size of rectangle in descending order i.e., larger rectangle is ranked prior to smaller ones.

**Face Alignment:**

Face alignment is to be done in a manner such that faces with different pose are forced aligned to a common pose(centered). This is done by getting angle of pose using Eye Cascades. The face is rotated across x, y as well as z – axis. (using opencv affine operations)

Moreover, the face is pivoted about its eyes at a fixed distance i.e., the face is resized such that distance between eyes of all face images are constant.

This done so that measuring distances between landmarks can used as a feature to detect AUs and hence classify emotion.

**Facial Landmarks:**

Upon aligning the face pivoted across eyes, the image becomes suitable to derive facial landmarks which could be further compared across multiple frames.

The initial frame landmarks are estimated using Active Appearance Model (AAM). An AAM face model consists a shape model and a texture model. The fitting procedure iteratively adjust the model until satisfy.

Further in consecutive Lucas–Kanade (LK) optical flow tracker can be used by estimating the displacements of the feature points.

**Face Masking:**

The detected landmarks are positioned uniquely without any relevance between frames as face structure varies from person to person and face pose. So, these face-landmarks are vectorised such that can be compared across the frames.



**Multi - Cascade Detectors:**

Some of the most important facial muscles show their movements in form of skin texture. These are the case where landmarks may not show drastic movements but the skin texture is enough to determine face expression features.

To determine these skin textures at different regions of the face, various &quot;Cascades&quot; are used at different regions over the face.



Each Cascade&#39;s region of implementation is determined by landmarks location.

The outcome of cascading is defined as expression feature intensity and represented as binary {0,1}.



**AUs Xgboost (xgb\_1\_1):**

The clusters of various face expression feature vectors can be formed to define a cumulative muscle actions called as Action Units. The classification is computed using xgboost algorithm.

A xgboost model is created where inputs are the face feature vectors, and the output is classified Action Units. The model tree is trained such that the combinations of definite range of varying feature intensities are clustered under each Action Units. The model is called FACS.

We get required active AUs probabilities, classified based upon given feature vectors.

**Xgboost Classifier(xgb\_1\_2):**

The second fold Xgboost model is based upon activeness of various AUs to classify an emotion. The classification is performed, by feeding probabilities of AUs active in a frame and computed output is the probabilities of emotions possibly shown.

**Full – Xgboost (xgb\_2):**

Due to less accurate prediction of activeness of AUs, the final classification may lead to wrong predictions. Thus to avoid wrong prediction and get accurate probability distribution across emotions, we implement a model computing emotion classification directly based upon feature vectors. The model runs in parallel to 2-fold Xgboost model.

**Fused Result:**

The output, probabilities of emotions from both the model are combined based upon strength of each emotion relative to others.



**Result Presentation:**

**Steps towards accuracy:**

- **--** Similar face alignment across frames
- **--** Proper selection of feature vectors
- **--**** Introduce relationship between landmarks across sequence frames.**
- **--** Accuracy can be gained further by tuning of the Xgboost models

**Thank You**

**References:**

** **
