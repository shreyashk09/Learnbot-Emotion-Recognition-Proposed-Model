## Learnbot – Emotion Recognition

## (Idea Proposal)

##

### FACS - AUs – Xgboost (2-fold and 1-full classifier)

##

**Action Units (AUs) and** **Facial Action Coding System (FACS): **

Emotion expressions on face is possible due to movement of various facial muscles. The Facial Action Coding System (FACS) refers to a set of facial muscle movements that correspond to a displayed emotion. It is an anatomical system for describing all observable facial movements into small units known as Action Units (AUs). AUs determines activeness of different sets of facial muscles at a time.

**Xgboost:**

XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting algorithm. Algorithm is designed to work with numeric values only.

It is also known as &#39; **regularized boosting**&#39; technique because of its techniques to reduce overfitting problem automatically even with less training data.

The performance of xgboost is recorded to be more than other famous algorithms like DNN, random forest, etc. It is faster than other algorithms as it implements parallel processing.It helps to correct errors made by previously trained tree thus. Moreover, it allows to define custom optimization objectives and evaluation criteria. User can start training an XGBoost model from its last iteration of previous run, which makes training simpler.

**Overview:**

The FACS - AUs – Xgboost algorithms can be implemented for emotion recognition over camera captured facial images. Initially, the images are pre-processed such that all the face images with various orientations and poses are aligned about z axis with fixed horizontal distance between eyes.

As the classification is carried into 5 prime emotions (and neutral), extracting key facial features is sufficient. To determine the feature vectors form the detected face frame, we plot facial landmarks over the face using AAM along with Lucas – Kanade (LK) algorithms. The facial landmarks are used to develop a wireframe across the face called as mesh. The details are extracted from the mesh as facial feature vectors. A neutral expression mesh is generated for the same face detected using the &quot;Golden Ratios&quot;. The driving forces are calculated between the &quot;expressive mask&quot; and &quot;neutral mask&quot;. The driving forces as well as the independent facial feature of expressive frame is combined in a list to form Feature Vectors List. The list is traversed across the following Xgboost Classifier to get the predicted emotion.

These feature vectors can be clustered into groups representing various unique face actions called as &quot;Action units&quot;. The combinations of these Action Units predict different emotion emotion.

 The feature vectors are transformed to the **first level Xgboost** to determine the Action Units (AUs) of the corresponding expression sequences. Finally, the detected AUs are inputed into the **second level Xgboost** for facial expressions classification.

In parallel, another **Xgboost** model will classify emotion directly from feature vectors. The fused result gives us accurate predication.

The model is trained with sequence of expression frames starting with less expressive to most expressive. Thus, face with any intensity of expression can be classified easily.

**Advantages:**

- **--**

**Hardware Requirements:**

**Flowchart:**

**Methodology:**

1. Image frame is captured by Learnbot camera in real – time.
2. Cascades (Haar cascade) are used for detecting faces and cropping (into 64 x 64).

(frontal and profile) and eye detector

1. The face pose in the image is aligned about z – axis and pivoted about fixed eye positions
2. Facial Landmarks are positioned across the face using AAM and further dynamically re - localized using Lucas – Kanade method.
3. Multi – Cascade detector are implemented
4. Face is masked with a wireframes and feature vectors are extracted.
5. Parallel to steps 5 and 6, a mask for neutral expression is generated based on &quot;Golden Formula&quot; and feature vector are extracted.
6. Force vectors are extracted from the masks and appended to the feature vectors list.
7.	Parallel to steps 5 and 6, a mask for neutral expression is generated based on “Golden Formula” and feature vector are extracted. 
8.	Force vectors are extracted from the masks and appended to the feature vectors list.
9.	1st fold Xgboost AU classifiers model (FACS) is used to determine probabilities of active Action Units.
10.	2nd fold Xgboost emotion classifier model is implemented to classify probability of emotion predicted based upon AUs activation probability.
11.	Parallel, another Full-Emotion classifier is implemented to directly compute feature vectors to classify predicted emotions probability.
12.	Probability results from both the models are combined to give a precise predicted emotion probability and label.
13.	The result is then dynamically represented on screen graphically with necessary parameters.


**Face Detection and Ranking:**

The captured images using Learnbot&#39;s camera may contain one or more faces at a time. The whole frame is investigated using Cascades (Haar Cascades) for face detection both front view and profile view. The face – rectangles are then cropped are reshaped into 64 x 64 square.

The face - rectangles are of different sizes and located at different location across the frame based upon the position of human in front of Learnbot in 3D. These faces are assigned ranks as follows:

- **--** Ranking based upon size of face-rectangle (measured diagonally) in descending order i.e., larger rectangle is ranked prior to smaller ones. (seems to be near and more interactive)
- **--** Ranking based upon displacement form center. Person at center should be prioritized.



**Face Alignment:**

Face alignment is to be implemented such that faces with different pose are forced aligned to a common pose(centered). This is done by getting angle of pose in xy-plane using **Eye Cascades**. The face is rotated only about **z – axis**. (using opencv affine operations).

Moreover, the face is pivoted about its eyes at a fixed distance i.e., the face is resized such that distance between eyes of all face images are constant.

This done so that, measuring distances between landmarks can used as a feature to detect AUs and hence classify emotion.

**Facial Landmarks:**

Upon aligning the face about Z-axis in XY - plane, the image becomes suitable to derive facial landmarks which could be further used to extract facial features.

The facial landmarks are accumulated based upon pose of face in XZ – plane and YZ – plane. So, we define a threshold angle about which the face is distinguished as &quot;Full-Face&quot; view and &quot;Half- Face&quot; view. In case of Half – Face view only one half of face is used to determine the expression. The other half is either incapable of determining the exact landmark positions and distinguish among them or is not visible.

The initial frame landmarks are estimated using Active Appearance Model (AAM). An AAM face model consists a shape model and a texture model. The fitting procedure iteratively adjust the model until satisfy.

Further in consecutive frames Lucas–Kanade (LK) optical flow tracker can be used by estimating the displacements of the feature points.

**Face Masking:**

**(motivation:**  **Candide Wireframe Model**** )**

The facial landmarks are very sensitive to the movements of facial muscles. Each person has unique face structure and gives vivid poses at different point of time. To study the pattern followed by vector displacement of these facial landmarks for any given expression, we generate mask over the face such that details extracted from them can be compared across the frames.

The architecture of the mask is motivated from Candide Wireframe Model. The driving forces exerted during change in expression intensities are the distance between nodes and angles at intersections in wireframe.

The angles at intersections don&#39;t change with change orientation of face about z-axis or with any change in size (aspect ratio is constant) of the face frame for any given expression intensity.

Inorder to find the change in these driving forces we subtract the present frame mask&#39;s details with mask&#39;s details of neutral expression. The result is change in angles and displacements of nodes.

The Neutral Mask is generated using fixed dimensions of face in the frame i.e, distance between an eye center and mid-point of both the eyes and distance between mid-point of eyes and nose center. The landmarks for eyebrows and lips are plotted using Golden Ratio and on the basis of neutral images in training classes.

Even if neutral mask doesn&#39;t coincide exactly with actual neutral expression of face, the error becomes negligible by training vast data.

Thus, masks are appropriate for classification into different expression classes.

**Golden Ratio:**

The human face abounds with examples of the Golden Ratio, also known as the Golden Section or Divine Proportion.

The head forms a golden rectangle with the eyes at its midpoint. The mouth and nose are each placed at golden sections of the distance between the eyes and the bottom of the chin.

Thus, we can resize the landmarks mask accordingly and would be proportionally vary with frames.



**Multi - Cascade Detectors:**

Some of the most important facial muscles show their movements in form of skin texture. These are the case where landmarks may not show drastic movements but the skin texture is enough to determine face expression features.

To determine these skin textures at different regions of the face, various &quot;Cascades&quot; can be used at different regions over the face.

Each Cascade&#39;s region of implementation is determined by nearest landmarks&#39; locations respectively.

The outcome of cascading is defined as expression feature intensity and represented in form of binary {0,1}.



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
- **--** Perfect masking
- **--** Accuracy can be gained further by tuning of the Xgboost models

**Thank You**

**References:**



** **
