import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')


def main():
    print("Sepsis prediction - training two models")
    print("=" * 60)

    print("Loading data...")
    df = pd.read_csv('data/sepsis_emr_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Sepsis cases: {df['sepsis_risk'].sum()}/{len(df)} ({df['sepsis_risk'].mean()*100:.1f}%)")

    X = df.drop(['patient_id', 'sepsis_risk'], axis=1)
    y = df['sepsis_risk']

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Model A: Gradient Boosting
    print("\nBuilding Model A: GradientBoostingClassifier")
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.8, 1.0]
    }
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb_model, gb_params, cv=5, scoring='roc_auc', n_jobs=-1)
    gb_grid.fit(X_train_scaled, y_train)

    model_a = gb_grid.best_estimator_
    model_a_score = gb_grid.best_score_

    print(f"Model A Best AUC: {model_a_score:.4f}")
    print(f"Model A Best params: {gb_grid.best_params_}")

    feature_names = X.columns
    importances = model_a.feature_importances_
    print("Model A feature importance:")
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]:
        print(f"   {name}: {importance:.3f}")

    # Model B: Neural Network (MLPClassifier)
    print("\nBuilding Model B: MLPClassifier")
    nn_params = {
        'hidden_layer_sizes': [(50,), (100,), (50, 25)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001, 0.001]
    }
    nn_model = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)
    nn_grid = GridSearchCV(nn_model, nn_params, cv=5, scoring='roc_auc', n_jobs=-1)
    nn_grid.fit(X_train_scaled, y_train)

    model_b = nn_grid.best_estimator_
    model_b_score = nn_grid.best_score_

    print(f"Model B Best AUC: {model_b_score:.4f}")
    print(f"Model B Best params: {nn_grid.best_params_}")
    print(f"Model B Architecture: {nn_grid.best_params_['hidden_layer_sizes']}")

    # Ensemble: weighted average based on training AUC
    print("\nBuilding ensemble (model A + model B)")
    model_a_pred = model_a.predict_proba(X_train_scaled)[:, 1]
    model_b_pred = model_b.predict_proba(X_train_scaled)[:, 1]

    model_a_auc = roc_auc_score(y_train, model_a_pred)
    model_b_auc = roc_auc_score(y_train, model_b_pred)

    total_auc = model_a_auc + model_b_auc
    weight_a = model_a_auc / total_auc
    weight_b = model_b_auc / total_auc

    print(f"Model A Weight: {weight_a:.3f}")
    print(f"Model B Weight: {weight_b:.3f}")

    # Evaluation
    print("\nEvaluating models...")
    model_a_test_pred = model_a.predict_proba(X_test_scaled)[:, 1]
    model_a_test_auc = roc_auc_score(y_test, model_a_test_pred)

    model_b_test_pred = model_b.predict_proba(X_test_scaled)[:, 1]
    model_b_test_auc = roc_auc_score(y_test, model_b_test_pred)

    ensemble_pred = (model_a_test_pred * weight_a) + (model_b_test_pred * weight_b)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    ensemble_class_pred = (ensemble_pred >= 0.5).astype(int)

    print("\nFinal results:")
    print(f"   Model A (Tree-based) Test AUC: {model_a_test_auc:.4f}")
    print(f"   Model B (Neural Network) Test AUC: {model_b_test_auc:.4f}")
    print(f"   Ensemble (A+B) Test AUC: {ensemble_auc:.4f}")

    print("\nEnsemble classification report:")
    report = classification_report(y_test, ensemble_class_pred)
    print(report)

    # Save artifacts
    print("\nSaving models...")
    joblib.dump(model_a, 'models/model_a_tree_based_model.pkl')
    joblib.dump(model_b, 'models/model_b_neural_network_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')

    ensemble_weights = {
        'model_a_tree_based': weight_a,
        'model_b_neural_network': weight_b
    }
    joblib.dump(ensemble_weights, 'models/ensemble_weights.pkl')

    print("All models saved.")

    # Simple clinical scenario checks
    high_risk = np.array([[125, 28, 38.8, 16.5, 4.2, 72, 3]])
    high_risk_scaled = scaler.transform(high_risk)
    hr_pred_a = model_a.predict_proba(high_risk_scaled)[0, 1]
    hr_pred_b = model_b.predict_proba(high_risk_scaled)[0, 1]
    hr_ensemble = (hr_pred_a * weight_a) + (hr_pred_b * weight_b)

    low_risk = np.array([[85, 16, 36.8, 7.2, 1.1, 35, 0]])
    low_risk_scaled = scaler.transform(low_risk)
    lr_pred_a = model_a.predict_proba(low_risk_scaled)[0, 1]
    lr_pred_b = model_b.predict_proba(low_risk_scaled)[0, 1]
    lr_ensemble = (lr_pred_a * weight_a) + (lr_pred_b * weight_b)

    print("\nHigh risk patient:")
    print(f"   Model A: {hr_pred_a:.1%}, Model B: {hr_pred_b:.1%}")
    print(f"   Ensemble: {hr_ensemble:.1%}")

    print("\nLow risk patient:")
    print(f"   Model A: {lr_pred_a:.1%}, Model B: {lr_pred_b:.1%}")
    print(f"   Ensemble: {lr_ensemble:.1%}")

    print("\n" + "=" * 60)
    print("Training completed.")
    print("=" * 60)

    return {
        'model_a_auc': model_a_test_auc,
        'model_b_auc': model_b_test_auc,
        'ensemble_auc': ensemble_auc,
        'model_a': model_a,
        'model_b': model_b,
        'weights': ensemble_weights
    }


if __name__ == "__main__":
    results = main()