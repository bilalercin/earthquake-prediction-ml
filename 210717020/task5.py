import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline  #

# Load and preprocess data
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
df['message'] = df['message'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),  # Use unigrams, bigrams, and trigrams
    ('smote', SMOTE(random_state=42)),  # SMOTE to balance classes
    ('clf', MultinomialNB())  # Default classifier is Multinomial Naive Bayes
])

# Hyperparameter tuning for MultinomialNB
nb_param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__alpha': [0.1, 1.0, 10.0]  # Naive Bayes alpha
}

# Hyperparameter tuning for SVC
svc_param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__C': [0.1, 1, 10],  # Regularization parameter for SVM
    'clf__gamma': ['scale', 'auto']  # Kernel coefficient
}

# Hyperparameter tuning for RandomForest
rf_param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 10, 20, 30]
}

# Try multiple models: Naive Bayes, SVC, and Random Forest
models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Perform grid search for each model
for model_name, model in models.items():

    # Setup pipeline with current model
    pipeline.set_params(clf=model)

    if model_name == 'Multinomial Naive Bayes':
        param_grid = nb_param_grid
    elif model_name == 'SVM':
        param_grid = svc_param_grid
    elif model_name == 'Random Forest':
        param_grid = rf_param_grid

    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline  #

# Load and preprocess data
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
df['message'] = df['message'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),  # Use unigrams, bigrams, and trigrams
    ('smote', SMOTE(random_state=42)),  # SMOTE to balance classes
    ('clf', MultinomialNB())  # Default classifier is Multinomial Naive Bayes
])

# Hyperparameter tuning for MultinomialNB
nb_param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__alpha': [0.1, 1.0, 10.0]  # Naive Bayes alpha
}

# Hyperparameter tuning for SVC
svc_param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__C': [0.1, 1, 10],  # Regularization parameter for SVM
    'clf__gamma': ['scale', 'auto']  # Kernel coefficient
}

# Hyperparameter tuning for RandomForest
rf_param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 10, 20, 30]
}

# Try multiple models: Naive Bayes, SVC, and Random Forest
models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Perform grid search for each model
for model_name, model in models.items():

    # Setup pipeline with current model
    pipeline.set_params(clf=model)

    if model_name == 'Multinomial Naive Bayes':
        param_grid = nb_param_grid
    elif model_name == 'SVM':
        param_grid = svc_param_grid
    elif model_name == 'Random Forest':
        param_grid = rf_param_grid

    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
