# Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Veri setini yükleme
df = pd.read_csv('earthquake_data.csv')

# Veri setini inceleme
print("Veri Seti Genel Bilgi:")
print(df.info())
print("\nİlk 5 Satır:")
print(df.head())

# Özellik ve hedef değişken tanımları
features = ["magnitude", "depth", "cdi", "mmi", "sig"]
target = "alert"

# Veri temizleme ve ön işleme
df = df[features + [target]]
print("\nEksik değerler analizi:")
print(df.isnull().sum())

# Numerik eksik değerleri KNN ile doldurma
knn_imputer = KNNImputer(n_neighbors=5)
df[features] = knn_imputer.fit_transform(df[features])

# Kategorik hedef değişken için en sık değerle doldurma
if df[target].isnull().sum() > 0:
    df[target].fillna(df[target].mode()[0], inplace=True)

print("\nEksik değerler doldurulduktan sonra:")
print(df.isnull().sum())

# Özellik mühendisliği
df['impact_score'] = df['magnitude'] * df['depth']
df['log_depth'] = np.log1p(df['depth'])

# Hedef değişkeni kategorik tipe dönüştürme
df[target] = df[target].astype('category')

# Görselleştirme: Hedef değişken dağılımı
df[target].value_counts().plot(kind='bar', title='Alert Levels Distribution', color=['green', 'yellow', 'orange', 'red'])
plt.show()

# SMOTE ile veri dengesini sağlama
X = df.drop(columns=target)
y = df[target]

# Mevcut sınıf dağılımını analiz etme
class_counts = y.value_counts()
print("\nMevcut sınıf dağılımı:")
print(class_counts)

# SMOTE Uygulaması
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Artırılmış veri setinin sınıf dağılımını kontrol et
print("\nArtırılmış veri setinin sınıf dağılımı:")
print(pd.Series(y_resampled).value_counts())

# Eğitim ve test verisinin ayrılması
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Özellik ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model tanımlama ve eğitim
models = []
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

models.append(('DecisionTree', dt))
models.append(('RandomForest', rf))
models.append(('GradientBoosting', gb))

# Hiperparametre optimizasyonu için parametre ızgarası
param_grid = {
    'DecisionTree': {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5]},
    'RandomForest': {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5]},
    'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5], 'min_samples_split': [2, 5]}
}

# Hiperparametre optimizasyonu ve model eğitimi
best_models = {}
for name, model in models:
    print(f"Training {name} using cross-validation...")
    grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best Parameters for {name}: {grid_search.best_params_}")

# Performans değerlendirme
f1_scores = []
for name, model in best_models.items():
    print(f"Evaluating {name}...")
    # Çapraz doğrulama ile modelin F1 skorunu hesapla
    f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted').mean()
    f1_scores.append((name, f1))
    print(f"F1 Score for {name}: {f1}")


# En İyi Modelin Test Verisi Üzerindeki Performansını Değerlendirme
best_model_name, best_model_score = f1_scores[0]
best_model = best_models[best_model_name]
y_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for {best_model_name}')
plt.show()

# ROC Curve
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test.cat.codes, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.title(f'ROC Curve for {best_model_name}')
    plt.legend()
    plt.show()
