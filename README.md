# Task2_Regression
## Task 2: Real Estate Housing Price Prediction (California Housing Dataset)
Objective:Built regularized linear regression models (Ridge and Lasso) to predict California median housing values based on structural, geographical, and demographic features.
  Methodology:Imputed missing feature values (207 missing entries in `total_bedrooms`) using column median, converted the categorical `ocean_proximity` feature into numeric indicators via One-Hot Encoding (`pd.get_dummies`), scaled features using `StandardScaler`, and performed an 80/20 train-test split.
  Results: Achieved consistent generalization with an $R^2$ score of ~0.6255 and an RMSE of ~$70,057 across test predictions, effectively preventing overfitting using $L_1$ and $L_2$ regularization.
