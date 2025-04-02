import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Toy regression dataset
n_samples = 5000
n_features = 5
n_targets = 3
random_state = 42

X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_targets=n_targets,
    random_state=random_state
)
# DataFrame
features = [f"Feature_{i + 1}" for i in range(n_features)]
targets = [f"Target_{i + 1}" for i in range(n_targets)]

df = pd.DataFrame(X, columns=features)
targets_df = pd.DataFrame(y, columns=targets)

#Merge data
full_data = pd.concat([df, targets_df], axis=1)

# Missing value and random selection
missing_percentage = 0.02
n_missing = int(n_samples * missing_percentage)

feature_with_missing = "Feature_1"
missing_indices = random.sample(range(n_samples), n_missing)

# add NaN
full_data.loc[missing_indices, feature_with_missing] = np.nan

# Random Imputation??
feature_mean = full_data[feature_with_missing].mean(skipna=True)
feature_std = full_data[feature_with_missing].std(skipna=True)

random_imputed_data = full_data.copy()
random_imputed_data.loc[missing_indices, feature_with_missing] = np.random.normal(
    loc=feature_mean, scale=feature_std, size=n_missing
)

# Separating missing and non-missing data
data_with_missing = full_data[full_data[feature_with_missing].isnull()]
data_without_missing = full_data[~full_data[feature_with_missing].isnull()]

# Training regression model (ridge regression
regression_features = [col for col in full_data.columns if col != feature_with_missing]
regressor = Ridge(alpha=1.0)

# features
X_train = data_without_missing[regression_features]
y_train = data_without_missing[feature_with_missing]

# estimate
regressor.fit(X_train, y_train)

# Estimating missing values
X_test = data_with_missing[regression_features]
predicted_values = regressor.predict(X_test)


regression_imputed_data = full_data.copy()
regression_imputed_data.loc[data_with_missing.index, feature_with_missing] = predicted_values

# Feature and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# normalize data
X_scaled = scaler_X.fit_transform(df)
y_scaled = scaler_y.fit_transform(targets_df)

# Compare the original data and the two imputed data
datasets = {
    "Original": full_data.dropna(),
    "Random Imputation": random_imputed_data,
    "Regression Imputation": regression_imputed_data
}

# MSE calculate
mse_scores = {name: {target: 0 for target in targets} for name in datasets}

for name, dataset in datasets.items():
    train, test = train_test_split(dataset, test_size=0.3, random_state=42)

    X_train = train[features]
    y_train = train[targets]
    X_test = test[features]
    y_test = test[targets]

    # Neural network
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        random_state=42,
        max_iter=2000,
        alpha=0.0001,
        learning_rate_init=0.001,
        solver='adam'
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    for i, target in enumerate(targets):
        mse = mean_squared_error(y_test.iloc[:, i], predictions[:, i])
        mse_scores[name][target] = mse

#output format
for target in targets:
    target_mse = mse_scores["Regression Imputation"][target]
    print(f"Mean Squared Error ({target}): {target_mse:.18f}")


plt.figure(figsize=(15, 5))
for i, target in enumerate(targets):
    plt.subplot(1, 3, i + 1)
    sample_size = min(200, len(data_with_missing))
    sample_data_without_missing = data_without_missing.sample(n=sample_size, random_state=42)
    sample_data_with_missing = data_with_missing.sample(n=sample_size, random_state=42, replace=True)

    plt.scatter(sample_data_without_missing[target], sample_data_without_missing[feature_with_missing], color="red",
                label="Actual value", marker="*")
    plt.scatter(sample_data_with_missing[target], predicted_values[:sample_size], color="blue", label="Predicted value",
                marker="*")
    plt.ylabel("Feature")
    plt.xlabel(target)
    plt.legend()

    mse_value = mse_scores["Regression Imputation"][target]

plt.suptitle("Missing Value Prediction through Regression")
plt.show()

# Dataset MSE Scores
print("\nDataset MSE Scores:")
for name, target_scores in mse_scores.items():
    dataset_mse = np.mean(list(target_scores.values()))
    print(f"{name:20} {dataset_mse:.3f}")
