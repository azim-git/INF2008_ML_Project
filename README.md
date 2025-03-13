# ML_Project
## ver1:
Gaussian Naive Bayes Accuracy: 48.84362401867176
Decision Tree Accuracy: 98.61729966758611
Random Forest Accuracy: 51.86717589645661
K Nearest Neighbors Accuracy: 93.71596293938751
Xgboost Accuracy: 99.83379305467147

## ver3:
SVM:
Best hyperparameters for svm: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
svm Training Accuracy: 96.14%
svm Validation Accuracy: 80.17%

Gaussian Naive Bayes:
gaussian_naive_bayes Training Accuracy: 37.94%
gaussian_naive_bayes Validation Accuracy: 37.67%

Decision Tree:
Best hyperparameters for decision_tree: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2}
decision_tree Training Accuracy: 76.95%
decision_tree Validation Accuracy: 49.46%

Random Forest:
Best hyperparameters for random_forest: {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}
random_forest Training Accuracy: 77.87%
random_forest Validation Accuracy: 68.57%

K Nearest Neighbours:
Best hyperparameters for k_nearest_neighbors: {'metric': 'euclidean', 'n_neighbors': 7}
k_nearest_neighbors Training Accuracy: 79.40%
k_nearest_neighbors Validation Accuracy: 71.48%

Stochastic Gradient Descent:
Best hyperparameters for sgd: {'loss': 'hinge', 'max_iter': 100, 'penalty': 'l2'}
sgd Training Accuracy: 58.75%
sgd Validation Accuracy: 56.48%

XGBoost:
Best hyperparameters for xgboost: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
xgboost Training Accuracy: 84.08%
xgboost Validation Accuracy: 74.67%

Runtime to train all Models: ~3h 30mins