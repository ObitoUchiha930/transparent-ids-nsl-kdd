"""تفسير قرارات HistGradientBoosting باستخدام SHAP على بيانات NSL-KDD."""
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier

from train import CATEGORICAL, OUT, load

SEED = 42
SAMPLE_SIZE = 300
BACKGROUND_SIZE = 20
CLASS_ATTACK = 1
TOP_N = 5

MODEL_PATH = os.path.join(OUT, "best_model.joblib")
SUMMARY_PNG = os.path.join(OUT, "shap_summary.png")
ONE_PNG = os.path.join(OUT, "shap_one_attack.png")


def encode(X_train, X_test):
    """نفس ترميز train.py: get_dummies على التدريب والاختبار معًا."""
    both = pd.get_dummies(pd.concat([X_train, X_test]), columns=CATEGORICAL)
    return both.iloc[: len(X_train)], both.iloc[len(X_train):]


def build_explainer(model, background):
    """يجرّب TreeExplainer أولًا، وإن لم يدعم النموذجfellback إلى Explainer على predict_proba."""
    try:
        return shap.TreeExplainer(model), "TreeExplainer"
    except Exception as exc:  # HistGradientBoosting غير مدعوم في TreeExplainer
        print(f"TreeExplainer غير مدعوم لهذا النموذج: {type(exc).__name__}: {exc}")
        print("→ التحويل إلى shap.Explainer مع predict_proba وخلفية صغيرة.")
        try:
            exp = shap.Explainer(model.predict_proba, background,
                                 algorithm="permutation")
            return exp, "Explainer(predict_proba) - permutation"
        except Exception as exc2:
            print(f"فشل predict_proba الكامل ({exc2}) → attempt على عمود الهجوم فقط.")
            exp = shap.Explainer(
                lambda X: model.predict_proba(X)[:, CLASS_ATTACK], background,
                algorithm="permutation")
            return exp, "Explainer(predict_proba[:,1]) - permutation"


def attack_column(shap_values):
    """يرجّع قيم SHAP وقيم الأساس لفئة الهجوم فقط، أي مهما كان عدد أبعاد المخرجات."""
    if isinstance(shap_values, shap.Explanation):
        values = np.asarray(shap_values.values)
        base = np.asarray(shap_values.base_values)
    else:
        values = np.asarray(shap_values)
        base = None
    if values.ndim == 3:  # (n, features, classes)
        values = values[:, :, CLASS_ATTACK]
        if base is not None and base.ndim == 2:
            base = base[:, CLASS_ATTACK]
    if base is None:
        base = np.zeros(values.shape[0])
    return values, base


def main():
    X_train, y_train = load("KDDTrain+.txt")
    X_test, y_test = load("KDDTest+.txt")
    X_train, X_test = encode(X_train, X_test)
    print(f"بيانات التدريب: {X_train.shape} | الاختبار: {X_test.shape}")

    # 1) تدريب النموذج بإعداداته الافتراضية
    model = HistGradientBoostingClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    joblib.dump({"model": model, "columns": list(X_train.columns)}, MODEL_PATH)
    print(f"تم حفظ النموذج في: {MODEL_PATH}")

    # 2) عينة من 300 اتصال من ملف الاختبار + خلفية صغيرة
    X_sample = X_test.sample(n=SAMPLE_SIZE, random_state=SEED)
    y_sample = y_test.loc[X_sample.index]
    background = shap.sample(X_test, BACKGROUND_SIZE, random_state=SEED)
    print(f"عينة: {X_sample.shape} | خلفية: {background.shape}")

    explainer, how = build_explainer(model, background)
    print(f"المفسِّر المستخدم: {how}")
    raw_values = explainer(X_sample)
    values, base = attack_column(raw_values)
    print(f"قيم SHAP لفئة الهجوم: {values.shape}")

    cols = list(X_sample.columns)

    # 3) الرسم العام لأهم الميزات
    summary = shap.Explanation(
        values=values, base_values=base, data=X_sample.to_numpy(),
        feature_names=cols,
    )
    shap.plots.beeswarm(summary, max_display=15, show=False)
    plt = matplotlib.pyplot
    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"تم حفظ الرسم العام في: {SUMMARY_PNG}")

    # 4) اتصال واحد صُنّف هجومًا بشكل صحيح
    pred = model.predict(X_sample)
    hits = [i for i, p in enumerate(pred) if p == 1 and y_sample.iloc[i] == 1]
    if not hits:
        raise SystemExit("لا يوجد في العينة اتصال صُنّف هجومًا بشكل صحيح.")
    idx = hits[0]
    proba = model.predict_proba(X_sample.iloc[[idx]])[0, CLASS_ATTACK]

    one = shap.Explanation(
        values=values[idx], base_values=base[idx],
        data=X_sample.iloc[idx].to_numpy(), feature_names=cols,
    )
    shap.plots.waterfall(one, max_display=10, show=False)
    plt.gcf().set_size_inches(11, 6)
    plt.tight_layout()
    plt.savefig(ONE_PNG, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"تم حفظ تفسير الاتصال في: {ONE_PNG}")

    # 5) أهم 5 ميزات دفعت القرار نحو "هجوم"
    row = pd.Series(values[idx], index=cols).sort_values(ascending=False)
    print("\n" + "=" * 60)
    print("تفسير اتصال واحد: attack = 1 / normal = 0")
    print(f"  predict_proba(attack) = {proba:.4f}")
    print(f"  القيمة الأساسية (base) = {base[idx]:.4f}")
    print(f"  الصف في ملف الاختبار  = {X_sample.index[idx]}")
    print("=" * 60)
    print(f"أهم {TOP_N} ميزات دفعت القرار نحو \"هجوم\":")
    for rank, (feat, val) in enumerate(row.head(TOP_N).items(), start=1):
        actual = X_sample.iloc[idx][feat]
        print(f"  {rank}. {feat:<32} SHAP={val:+.4f}   قيمة={actual}")
    print(f"  (مجموع كل القيم = {values[idx].sum():+.4f} "
          f"≈ base + مجموع = {base[idx] + values[idx].sum():.4f})")


if __name__ == "__main__":
    main()
