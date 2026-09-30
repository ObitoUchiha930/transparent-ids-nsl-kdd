"""مقارنة عدة نماذج كشف تسلل على NSL-KDD باستخدام نفس تحميل البيانات وترميز train.py."""
import os
import time

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from train import CATEGORICAL, OUT, load

SEED = 42
RESULTS_CSV = os.path.join(OUT, "model_comparison.csv")


def encode(X_train, X_test):
    """نفس ترميز train.py: get_dummies على التدريب والاختبار معًا لضمان تطابق الأعمدة."""
    both = pd.get_dummies(pd.concat([X_train, X_test]), columns=CATEGORICAL)
    return both.iloc[: len(X_train)], both.iloc[len(X_train):]


def evaluate(name, model, X_test, y_test):
    """قياسات فئة الهجوم (1) مع أعدادTP/FP/FN/TN."""
    pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "accuracy": round(float((pred == y_test).mean()), 4),
        "precision_attack": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall_attack": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_attack": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "fn": int(fn),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
    }


def main():
    X_train, y_train = load("KDDTrain+.txt")
    X_test, y_test = load("KDDTest+.txt")
    X_train, X_test = encode(X_train, X_test)
    print(f"بيانات التدريب: {X_train.shape} | الاختبار: {X_test.shape}\n")

    models = [
        ("RandomForest(200)", RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=SEED)),
        ("RandomForest(200,balanced)", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED)),
        ("ExtraTrees(200)", ExtraTreesClassifier(
            n_estimators=200, n_jobs=-1, random_state=SEED)),
        ("HistGradientBoosting", HistGradientBoostingClassifier(
            random_state=SEED)),
    ]

    rows = []
    for name, model in models:
        print(f"تدريب {name} ...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        row = evaluate(name, model, X_test, y_test)
        row["fit_seconds"] = round(fit_s, 1)
        rows.append(row)
        print(f"  انتهى خلال {fit_s:.1f}s -> "
              f"acc={row['accuracy']} f1={row['f1_attack']} fn={row['fn']}\n")

    df = pd.DataFrame(rows).sort_values("f1_attack", ascending=False).reset_index(drop=True)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

    print("=== مقارنة النماذج على KDDTest+.txt (مرتبة حسب F1 لفئة الهجوم) ===")
    print(df.to_string(index=False))
    print(f"\nالأفضل: {df.iloc[0]['model']} بـ F1={df.iloc[0]['f1_attack']} "
          f"(FN={df.iloc[0]['fn']})")
    print(f"تم الحفظ في: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
