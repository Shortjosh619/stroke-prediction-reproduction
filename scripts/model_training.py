import numpy as np 
import pandas as pd 
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC 
from imblearn.pipeline import Pipeline #i had the sklearn version at first, but i got an error when i ran it because the sklearn pipeline doesnt handle fit_resample from ros, it only handles fit_transform
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, confusion_matrix, f1_score, recall_score, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.pyplot as plt 


DATA_PATH = Path(r"C:\stroke-prediction-reproduction\data\processed")
RESULTS_PATH = Path(r"C:\stroke-prediction-reproduction\results")
RANDOM_STATE = 9


#load data
def load_data(directory):

    X_train = pd.read_csv(directory / "X_train.csv")
    X_test = pd.read_csv(directory / "X_test.csv")
    y_train = pd.read_csv(directory / "y_train.csv").squeeze(axis=1)
    y_test = pd.read_csv(directory / "y_test.csv").squeeze(axis=1)

    return X_train, X_test, y_train, y_test

def get_models():
     return {
    "LogReg": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=RANDOM_STATE, probability=True),
    "GBoost": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "RF": RandomForestClassifier(random_state=RANDOM_STATE)
    }


#initally did not have this helper function and ran without it. Leakage occured and figured out why. wrote to pass into loop in train_models instead of inital model variable i had
def build_pipeline(model):
    return Pipeline([("oversampler", RandomOverSampler(random_state=RANDOM_STATE)), 
                     ("model", model), 
                     ])


#define models
def train_models(X_train, y_train): 
    models_dict = get_models()

    #cross validation 
    cv = StratifiedKFold(n_splits=5, shuffle=True,  random_state=RANDOM_STATE) #nsplits is 5 because thats what the paper did
    scoring = ["accuracy", "f1", "recall", "roc_auc"]

    results = []

    for name, model in models_dict.items():
         pipe = build_pipeline(model)
         scores = cross_validate(pipe, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
        
         row = {"Model": name}
         
         for metrics in scoring:
             metric_key = "test_" + metrics

             row[metrics + "_mean"] = np.nanmean(scores[metric_key])
             row[metrics + "_std"] = np.nanstd(scores[metric_key])
             
         results.append(row)

    return pd.DataFrame(results)
        
#sklearn does not compute ECE itself, but gives what is needed via calibration_curve which is fraction of positives and mean predicted probability
# so i just made a wrapper here and then called it in the final eval func
def compute_ece(y_test, y_proba, n_bins=10):
     frac_of_pos, mean_pred_proba = calibration_curve(y_test, y_proba, n_bins=n_bins)
     hist, bin_edges = np.histogram(y_proba, n_bins, density=False)
     ece = np.sum((hist[:len(frac_of_pos)]/len(y_proba)) * np.abs((frac_of_pos - mean_pred_proba)))

     return ece 

def evaluate_final_model(X_train ,X_test, y_train, y_test):
        models_dict = get_models()
        results = []
    
        for name, model in models_dict.items():
             pipe = build_pipeline(model)
             
             pipe.fit(X_train, y_train)
             y_preds= pipe.predict(X_test)
             y_proba = pipe.predict_proba(X_test)[:,1]

             row = {
                  "model": name, 
                  "accuracy": accuracy_score(y_test, y_preds),
                  "roc_auc": roc_auc_score(y_test, y_proba),
                  "f1": f1_score(y_test, y_preds),
                  "recall": recall_score(y_test, y_preds),
                  "brier_score": brier_score_loss(y_test, y_proba),
                  "expected_callibration_error": compute_ece(y_test, y_proba)
                  }
             

             results.append(row)
         
             display = ConfusionMatrixDisplay.from_predictions(y_test, y_preds)
             display.plot()
             plt.tight_layout()
             plt.savefig(RESULTS_PATH / "confusion_matrices" / f"{name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
             print(f"\n Saved as {name}_correlation_matrix.png")
             plt.close()
             

        return pd.DataFrame(results)
             



def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

    print("\nRunning cross-validation...")
    cv_results = train_models(X_train, y_train)
    print("\n=== CV Results ===")
    print(cv_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))#did this to stop truncation in the console
    cv_folder = RESULTS_PATH / "cross_validation"
    cv_folder.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(cv_folder / "cv_results.csv", index=False)

    print("\nRunning final evaluation on held-out test set...")
    test_results = evaluate_final_model(X_train, X_test, y_train, y_test)
    print("\n=== Test Set Results ===")
    print(test_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    test_folder = RESULTS_PATH / "test_evaluation"
    test_folder.mkdir(parents=True, exist_ok=True)
    test_results.to_csv(test_folder / "test_results.csv", index=False)

    print("\nDone. Results saved to:", RESULTS_PATH.resolve())


if __name__ == "__main__":
    main()
