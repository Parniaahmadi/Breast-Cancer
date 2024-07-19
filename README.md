# **Predicting Breast Cancer Using KNN**

In this project some of the basic components of machine learning methods are presented and examined. These components include feature selection based on correlation, feature extraction using principal component analysis (PCA) and hyperparameter tuning through cross-validation using packages available in scikit-learn. KNN is used in a k-fold cross-validation process to predict new cases of breast cancer. The breast cancer Wisconsin dataset is used for this purpose which is a classic and binary dataset available as one of the scikit-learn datasets. 

This project has five parts:

**Part 1: Exploratory Data Analysis**
A dataframe is created. Data is split into training and test sets and standardized in a way that there is no leakage from the test set. Then, only the training set is used for visualization.


Check the codes in Spacecode

## Visualizing two features of the standardized data ##
![image](https://github.com/user-attachments/assets/daa8461d-24f0-4732-be9c-6fde9c7e69c0)

## Visualizing pairplots for multiple features of the standardized data ##
![image](https://github.com/user-attachments/assets/55cbe3a5-5fe6-4a88-9de1-66ff5530d6d0)

## KNN with cross-validation ##
## Plot cross-validation scores for different number of neighbours##
![image](https://github.com/user-attachments/assets/68d996af-f132-40be-a8c9-8a361580c977)

## KNN with cross-validation using GridSearchCV##
![image](https://github.com/user-attachments/assets/fddf06f9-8d42-4697-a780-01511b29e500)





**Part 2: Using All Features**
KNN with all the dataset features is used for predicting breast cancer. Each of part 2, 3 and 4 of this project has three sections. In the first section, for different number of neighbors, KNN is applied to the trainiing set without cross-validation, and the test score is reported for each number of neighbors. In the next section hyperparameter tuning is done to find the best number of neighbors in KNN using cross-validation for the training data. A loop over number of neighbors and cross_val_score is used in this section. The last step of part 2, 3 and 4 includes using GridSearchCV for cross-validation and hyperparameter tuning.

**Plot test scores for different number of neighbours without cross-validation**
![image](https://github.com/user-attachments/assets/ba4c52ef-4a9f-4955-81ad-826fb2ff0752)


**Part 3: Feature Selection**
Based on the correlation between features and the target, and the correlation between features themselves, a function is designed to drop some of the features. This function accepts training dataset and order the features based on correlation with the target, then from each two highly correlated features the one which has a weaker correlation with the target is dropped. Then similar to part 2, KNN is used for predicting breast cancer without and with cross-validation based on the new set of features.

**Part 4: Feature Extraction**
Principal component analysis (PCA) is applied to the training data to extract the most important components (eigenvectors) using singular value decomposition (SVD) of training data or eigendecomposition of the covariance matrix. Then similar to parts 2 and 3, KNN is used for predicting breast cancer without and with cross-validation based on the new set of extracted features (which are not the same as the original features).

**Part 5: Standardization, Feature Extraction, Cross-Validation, and Parameter Tuning Together Using Pipeline and GridSearchCV (NO DATA LEAKAGE)**
The correct approach to perform feature extraction is to incorporate it within the cross-validation process to prevent data leakage from both the validation set (the test set within the training data used in cross-validation) and the final test set. This is achieved by using a Pipeline for cross-validation. The Pipeline includes standardization, PCA, and KNN. By testing various combinations of PCA components and KNN neighbors, hyperparameter tuning is conducted using GridSearchCV. This ensures that the optimal parameters are identified without any data leakage.
