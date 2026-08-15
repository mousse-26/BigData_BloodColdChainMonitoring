"""
Diagnose why the classifier returns constant probabilities.
Run from the repo root with the venv active:  python diagnose.py
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from utils.preprocessing import engineer_features

model = joblib.load(Path("models/blood_bag_classifier.pkl"))
label_encoders = joblib.load(Path("artifacts/label_encoders.pkl"))

FEATURE_ORDER = [
    "route", "blood_type", "product_type",
    "temp_mean", "temp_min", "temp_max", "temp_std",
    "frac_temp_above_6", "frac_temp_above_8",
    "hum_mean", "hum_std", "door_count",
    "light_mean_abs", "accel_rms", "handling_stress",
    "temp_range", "temp_cv", "temp_deviation",
    "humidity_per_temp", "handling_per_door",
]

BASE = dict(
    temp_mean=4.0, temp_min=3.2, temp_max=5.5, temp_std=0.4,
    frac_temp_above_6=0.0, frac_temp_above_8=0.0,
    hum_mean=52.4, hum_std=4.1, door_count=1,
    light_mean_abs=12.5, accel_rms=0.42, handling_stress=1.35,
    route="Hospital_1", blood_type="O+", product_type="RBC",
)

CASES = {
    "pristine   ": dict(BASE),
    "borderline ": {**BASE, "temp_mean": 4.2, "temp_min": 2.8, "temp_max": 6.4,
                    "temp_std": 0.85, "frac_temp_above_6": 0.08, "door_count": 3},
    "bad        ": {**BASE, "temp_mean": 7.9, "temp_min": 3.1, "temp_max": 13.6,
                    "temp_std": 2.94, "frac_temp_above_6": 0.61,
                    "frac_temp_above_8": 0.34, "door_count": 14,
                    "accel_rms": 1.87, "handling_stress": 4.62},
    "catastrophe": {**BASE, "temp_mean": 19.0, "temp_min": 11.0, "temp_max": 28.0,
                    "temp_std": 6.1, "frac_temp_above_6": 1.0,
                    "frac_temp_above_8": 1.0, "door_count": 40,
                    "accel_rms": 5.0, "handling_stress": 12.0},
}


def prepare(case):
    df = pd.DataFrame([case])
    df = engineer_features(df)
    for col in ["route", "blood_type", "product_type"]:
        df[col] = label_encoders[col].transform(df[col])
    return df[FEATURE_ORDER]


print("=" * 68)
print("1. WHAT THE MODEL EXPECTS")
print("=" * 68)
names = getattr(model, "feature_names_in_", None)
print("n_features_in_ :", getattr(model, "n_features_in_", "?"))
print("classes_       :", getattr(model, "classes_", "?"))
if names is not None:
    print("trained order  :", list(names))
    if list(names) != FEATURE_ORDER:
        print("\n*** MISMATCH between trained order and FEATURE_ORDER in app.py ***")
        print("    missing from app.py:", set(names) - set(FEATURE_ORDER))
        print("    extra in app.py    :", set(FEATURE_ORDER) - set(names))
else:
    print("trained order  : (model has no feature_names_in_ — trained on a numpy array)")

print()
print("=" * 68)
print("2. ENGINEERED FEATURES PER CASE  (do they actually vary?)")
print("=" * 68)
frames = {}
for label, case in CASES.items():
    X = prepare(case)
    frames[label] = X.iloc[0]
    bad = X.columns[X.isna().any() | np.isinf(
        X.select_dtypes(float)).any()].tolist()
    if bad:
        print(f"{label} -> NaN/inf in: {bad}")

table = pd.DataFrame(frames).T
pd.set_option("display.width", 200, "display.max_columns", 50)
print(table.to_string())

constant = [c for c in table.columns if table[c].nunique() == 1]
print("\ncolumns identical across ALL four cases:", constant)

print()
print("=" * 68)
print("3. PREDICTIONS")
print("=" * 68)
for label, case in CASES.items():
    X = prepare(case)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    print(f"{label} -> pred={pred}  proba={np.round(proba, 4)}")

print()
print("=" * 68)
print("4. FEATURE IMPORTANCES")
print("=" * 68)
if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_,
                    index=FEATURE_ORDER).sort_values(ascending=False)
    print(imp.to_string())
    if (imp.head(3).sum()) < 0.05:
        print("\n*** All importances are tiny — the forest barely uses any feature ***")

print()
print("=" * 68)
print("5. TREE SHAPE")
print("=" * 68)
if hasattr(model, "estimators_"):
    depths = [t.get_depth() for t in model.estimators_]
    leaves = [t.get_n_leaves() for t in model.estimators_]
    print(f"n_estimators : {len(model.estimators_)}")
    print(
        f"max depth    : min={min(depths)} max={max(depths)} mean={np.mean(depths):.1f}")
    print(
        f"leaves       : min={min(leaves)} max={max(leaves)} mean={np.mean(leaves):.1f}")
    if max(depths) <= 1:
        print("\n*** Trees are stumps — the model never learned to split ***")
