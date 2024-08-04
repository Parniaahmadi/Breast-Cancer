# **Predicting Breast Cancer Using KNN**

In this project some of the basic components of machine learning methods are presented and examined. These components include feature selection based on correlation, feature extraction using principal component analysis (PCA) and hyperparameter tuning through cross-validation using packages available in scikit-learn. KNN is used in a k-fold cross-validation process to predict new cases of breast cancer. The breast cancer Wisconsin dataset is used for this purpose which is a classic and binary dataset available as one of the scikit-learn datasets. 

This project has five parts:

**Part 1: Exploratory Data Analysis**
A dataframe is created. Data is split into training and test sets and standardized in a way that there is no leakage from the test set. Then, only the training set is used for visualization.

![image](https://github.com/user-attachments/assets/6a13ef55-ca29-46e3-a9d8-15fefd3a9d83)


![image](https://github.com/user-attachments/assets/6868b977-72eb-407a-a693-dad1e82474cb)


![image](https://github.com/user-attachments/assets/2b9ee949-118b-459d-92de-c7d4077e3ee7)


![image](https://github.com/user-attachments/assets/007c286f-daff-449a-ac01-99bebfde4bfa)


![image](https://github.com/user-attachments/assets/cf05a40c-1552-48a1-a26a-2620138ad8e4)


## Visualizing two features of the standardized data ##
![image](https://github.com/user-attachments/assets/daa8461d-24f0-4732-be9c-6fde9c7e69c0)

## Visualizing pairplots for multiple features of the standardized data ##
![image](https://github.com/user-attachments/assets/55cbe3a5-5fe6-4a88-9de1-66ff5530d6d0)



**Part 2: Using All Features**
KNN with all the dataset features is used for predicting breast cancer. Each of part 2, 3 and 4 of this project has three sections. In the first section, for different number of neighbors, KNN is applied to the trainiing set without cross-validation, and the test score is reported for each number of neighbors. In the next section hyperparameter tuning is done to find the best number of neighbors in KNN using cross-validation for the training data. A loop over number of neighbors and cross_val_score is used in this section. The last step of part 2, 3 and 4 includes using GridSearchCV for cross-validation and hyperparameter tuning.


**Plot test scores for different number of neighbours without cross-validation**

![image](https://github.com/user-attachments/assets/84db9981-a6e9-4af0-b99c-d4f1d56c7212)


## KNN with cross-validation ##
## Plot cross-validation scores for different number of neighbours, CV=5 ##

![image](https://github.com/user-attachments/assets/3b1cc7bf-be08-4c9f-b734-7cd6947eb221)


## KNN with cross-validation using GridSearchCV##

![image](https://github.com/user-attachments/assets/ba869994-f0b5-4ba6-a72d-bf6e970fb065)



**Part 3: Feature Selection**
Based on the correlation between features and the target, and the correlation between features themselves, a function is designed to drop some of the features. This function accepts training dataset and order the features based on correlation with the target, then from each two highly correlated features the one which has a weaker correlation with the target is dropped. Then similar to part 2, KNN is used for predicting breast cancer without and with cross-validation based on the new set of features.

## Create a datafram (including only the training data) for feature section ##

![image](https://github.com/user-attachments/assets/e5fec26a-9760-4da1-b88e-c61d80f389c6)

Training data is used for feature selection based on correlation. As seen bellow, using threshold1=0.37 and threshold2=0.90, ten features are left in the dataframe which should make our model more efficient and enough accurate.

![image](https://github.com/user-attachments/assets/242de1b6-b7a4-416c-b898-24554157106e)

## Create a datafram (including only the training data) for feature section ##

Training data including only the new selected features is created. As seen in the heatmap of the new training dataset all the features have a minimum amount of correlation with the target and are not strongly correlated with each other (threshold1=0.37 and threshold2=0.90).

![image](https://github.com/user-attachments/assets/519ea0ed-45c4-4396-8309-03ed9a3b13d3)


**KNN without cross-validation after feature selection**
**Plot test scores for different number of neighbours after feature selection without cross-validation**

![image](https://github.com/user-attachments/assets/b8cc7cdd-8a15-47c8-96cf-e07ebd3bb2a2)


**KNN with cross-validation after feature selection**
## Plot cross-validation scores for different number of neighbours after feature selection##

![image](https://github.com/user-attachments/assets/5333d165-b083-430a-9507-28d7075ec358)


**KNN with cross-validation using GridSearchCV after feature selection**
## Plot cross-validation scores for different number of neighbours using GridSearchCV after feature selection ##

![image](https://github.com/user-attachments/assets/9e541358-b9d1-4bf3-8c2a-53836a271528)



**Part 4: Feature Extraction**
Principal component analysis (PCA) is applied to the training data to extract the most important components (eigenvectors) using singular value decomposition (SVD) of training data or eigendecomposition of the covariance matrix. Then similar to parts 2 and 3, KNN is used for predicting breast cancer without and with cross-validation based on the new set of extracted features (which are not the same as the original features).

**KNN without cross-validation after feature extraction**
## Plot test scores for different number of neighbours after feature extraction without cross-validation ##

![image](https://github.com/user-attachments/assets/d25723ac-e249-44ea-aac7-9952adcb2d78)

**KNN with cross-validation after feature extraction**
## Plot cross-validation scores for different number of neighbours after feature extraction ##

![image](https://github.com/user-attachments/assets/03fa9c8d-9da5-4d1c-af4f-db41ede7b3c3)

**KNN with cross-validation using GridSearchCV after feature extraction**
## Plot cross-validation scores for different number of neighbours using GridSearchCV after feature extraction ##

![image](https://github.com/user-attachments/assets/4d903077-7956-4330-a9e1-b7c62dec60b5)



**Part 5: Standardization, Feature Extraction, Cross-Validation, and Parameter Tuning Together Using Pipeline and GridSearchCV (NO DATA LEAKAGE)**
The correct approach to perform feature extraction is to incorporate it within the cross-validation process to prevent data leakage from both the validation set (the test set within the training data used in cross-validation) and the final test set. This is achieved by using a Pipeline for cross-validation. The Pipeline includes standardization, PCA, and KNN. By testing various combinations of PCA components and KNN neighbors, hyperparameter tuning is conducted using GridSearchCV. This ensures that the optimal parameters are identified without any data leakage.

## Contour plot for cross-validation score for a range of input features used in cross-validation ##

![image](https://github.com/user-attachments/assets/91c1ba40-4792-475a-9e97-426fb507c4a2)


**Summary**

This project introduces basic data science concepts, focusing on machine learning techniques such as feature selection, feature extraction, cross-validation, and preventing data leakage. Using the K-Nearest Neighbors (KNN) model, the project predicts breast cancer. Key techniques include:

**Exploratory Data Analysis:** Data from the scikit-learn Bunch object was split into training and test sets, standardized, and visualized. Strong correlations among features suggested that using all features could be inefficient.

**Initial Modeling:** The KNN model was initially used without cross-validation and then with cross-validation using cross_val_score and GridSearchCV.

**Feature Selection:** A function was developed for selecting features based on correlation.

**Feature Extraction with PCA:** Implementing PCA showed better cross-validation scores compared to correlation-based selection when extracting the same number of features.

**Avoiding Data Leakage:** The project demonstrated using Pipelines to properly implement standardization, feature extraction, and cross-validation without leaking data from validation and test sets.
