"""
تدريب نموذج تصنيف متعدد الفئات على عيّنة CIC-IDS2017.

المسار:
1) قراءة data/cicids2017/sample.csv
2) فحص التكرار في "Fwd Header Length" و "Fwd Header Length.1" وحذف المكرر إن تطابقا
3) حذف الصفوف المكررة تمامًا (duplicates)
4) تقسيم 70/30 stratify مع random_state=42
5) تدريب HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
6) accuracy و macro-F1 و weighted-F1 + جدول كل فئة + مصفوفة الالتباس
7) حفظ التقرير في output/cicids_report.txt
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(ROOT, "data", "cicids2017", "sample.csv")
OUT_DIR = os.path.join(ROOT, "output")
REPORT_PATH = os.path.join(OUT_DIR, "cicids_report.txt")

LABEL_COL = "Label"
DUP_A = "Fwd Header Length"
DUP_B = "Fwd Header Length.1"


def rel(path):
    """مسار نسبي بالنسبة لجذر المشروع (يمنع طباعة اسم المستخدم والمسار الكامل)."""
    return os.path.relpath(path, ROOT).replace("\\", "/")

TEST_SIZE = 0.30
RANDOM_STATE = 42

_lines = []


def log(text=""):
    """يطبع ويخزّن السطر في التقرير."""
    print(text)
    _lines.append(str(text))


def rule(char="=", width=100):
    return char * width


# ---------------------------------------------------------------------------
# 1) القراءة
# ---------------------------------------------------------------------------
def load():
    log(rule())
    log("1) قراءة العيّنة")
    log(rule())
    if not os.path.isfile(SAMPLE_PATH):
        raise SystemExit(f"الملف غير موجود: {rel(SAMPLE_PATH)}")
    log(f"المسار: {rel(SAMPLE_PATH)}")
    log(f"الحجم: {os.path.getsize(SAMPLE_PATH):,} بايت")

    df = pd.read_csv(SAMPLE_PATH, encoding="utf-8")
    log(f"عدد الصفوف : {len(df):,}")
    log(f"عدد الأعمدة: {df.shape[1]}")
    log(f"عدد الفئات : {df[LABEL_COL].nunique()}")

    types = df.dtypes.value_counts().to_dict()
    log(f"أنواع الأعمدة: {types}")

    numeric = df.drop(columns=[LABEL_COL])
    non_numeric = [c for c in numeric.columns
                   if not pd.api.types.is_numeric_dtype(numeric[c])]
    log(f"أعمدة غير رقمية داخل X: {non_numeric if non_numeric else 'لا يوجد'}")
    log(f"NaN في X: {int(numeric.isna().sum().sum())}   "
        f"NaN في Label: {int(df[LABEL_COL].isna().sum())}")
    log(f"inf في X: {int(np.isinf(numeric.select_dtypes(include=[np.number])).sum().sum())}")
    log("")
    return df


# ---------------------------------------------------------------------------
# 2) فحص تكرار العمودين
# ---------------------------------------------------------------------------
def check_duplicate_column(df):
    log(rule())
    log("2) فحص تكرار العمودين")
    log(rule())

    has_a, has_b = DUP_A in df.columns, DUP_B in df.columns
    log(f"«{DUP_A}» موجود: {has_a}")
    log(f"«{DUP_B}» موجود: {has_b}")

    if not (has_a and has_b):
        log("")
        log("لا يمكن المقارنة: أحد العمودين غائب. لم يُحذف شيء.")
        log("")
        return df

    a, b = df[DUP_A], df[DUP_B]
    same_mask = a == b
    n_same = int(same_mask.sum())
    n_diff = int((~same_mask).sum())

    log("")
    log(f"عدد الصفوف        : {len(df):,}")
    log(f"متساويان تمامًا   : {n_same:,}  ({n_same / len(df):.2%})")
    log(f"مختلفان           : {n_diff:,}  ({n_diff / len(df):.2%})")

    if n_diff == 0:
        log(f"النتيجة: متطابقان في كل الصفوف ← حُذف «{DUP_B}»")
        df = df.drop(columns=[DUP_B])
        log(f"عدد الأعمدة الآن  : {df.shape[1]}")
    else:
        log(f"النتيجة: غير متطابقين في {n_diff:,} صف ← أبقِني كليهما")
        diff = df.loc[~same_mask, [DUP_A, DUP_B]].drop_duplicates()
        log(f"قيم مختلفة (أزواج فريدة): {len(diff)}")
        log("")
        log(f"{DUP_A:<22}{DUP_B:<22}{'عدد الصفوف':>14}")
        log("-" * 58)
        pair_counts = (df.loc[~same_mask]
                       .groupby([DUP_A, DUP_B])
                       .size()
                       .sort_values(ascending=False)
                       .head(15))
        for (va, vb), n in pair_counts.items():
            log(f"{va:<22}{vb:<22}{n:>14,}")
    log("")
    return df


# ---------------------------------------------------------------------------
# 3) حذف المتكررات
# ---------------------------------------------------------------------------
def drop_duplicates(df):
    log(rule())
    log("3) حذف الصفوف المكررة تمامًا (duplicates)")
    log(rule())

    before_per_class = df[LABEL_COL].value_counts().to_dict()
    n_before = len(df)
    n_dup = int(df.duplicated().sum())

    before_table = "\n".join(
        f"  {label:<36}{count:>10,}" for label, count in sorted(
            before_per_class.items(), key=lambda kv: -kv[1])
    )

    df = df.drop_duplicates().reset_index(drop=True)
    after_per_class = df[LABEL_COL].value_counts().to_dict()

    log(f"صفوف قبل الحذف : {n_before:,}")
    log(f"صفوف مكررة      : {n_dup:,}  ({n_dup / n_before:.2%})")
    log(f"صفوف بعد الحذف  : {len(df):,}")
    log("")

    labels = sorted(set(before_per_class) | set(after_per_class),
                    key=lambda k: (-before_per_class.get(k, 0), k))
    log(f"{'الفئة':<36}{'قبل':>12}{'بعد':>12}{'المحذوف':>12}")
    log("-" * 72)
    for label in labels:
        b = before_per_class.get(label, 0)
        a = after_per_class.get(label, 0)
        log(f"{label:<36}{b:>12,}{a:>12,}{b - a:>12,}")
    log("-" * 72)
    log(f"{'المجموع':<36}{n_before:>12,}{len(df):>12,}{n_dup:>12,}")
    log("")
    return df


# ---------------------------------------------------------------------------
# 4) التقسيم
# ---------------------------------------------------------------------------
def split(df):
    log(rule())
    log("4) التقسيم 70/30 stratify")
    log(rule())

    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    log(f"X: {X.shape}   |   y: {y.shape}")
    log(f"random_state={RANDOM_STATE}   |   test_size={TEST_SIZE}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    log("")
    log(f"تدريب : {len(X_train):,} صف × {X_train.shape[1]} عمود")
    log(f"اختبار: {len(X_test):,} صف × {X_test.shape[1]} عمود")
    log(f"نسبة التدريب الفعلية: {len(X_train) / len(df):.2%}")
    log("")
    log(f"توزيع الفئات في التدريب ({y_train.nunique()} فئة):")
    log(f"  {'الفئة':<36}{'عدد':>10}{'النسبة':>10}")
    log("  " + "-" * 56)
    for label, count in y_train.value_counts().items():
        log(f"  {label:<36}{count:>10,}{count / len(y_train):>9.2%}")
    log("")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5-6) التدريب والتقييم
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    log(rule())
    log("5) التدريب")
    log(rule())
    log("HistGradientBoostingClassifier")
    log("  class_weight = 'balanced'")
    log(f"  random_state = {RANDOM_STATE}")

    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    log("اكتمل التدريب.")

    log("")
    log(rule())
    log("6) التقييم على مجموعة الاختبار")
    log(rule())

    y_pred = model.predict(X_test)
    labels_sorted = sorted(y_test.unique())

    acc = accuracy_score(y_test, y_pred)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    log(f"دقة_accuracy : {acc:.4f}")
    log(f"macro-F1     : {macro:.4f}")
    log(f"weighted-F1  : {weighted:.4f}")
    log("")

    log(rule("-"))
    log("جدول precision / recall / F1 / support لكل فئة")
    log(rule("-"))
    report = classification_report(
        y_test, y_pred, labels=labels_sorted,
        digits=4, zero_division=0,
    )
    for line in report.splitlines():
        log("  " + line if line.strip() else "")
    log("")

    log(rule("-"))
    log("مصفوفة الالتباس (محور الصف = الحقيقي، محور العمود = المتوقَّع)")
    log(rule("-"))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    width = max(len(l) for l in labels_sorted)

    header = " " * (width + 6) + "".join(f"{i:>{width // 2 + 2}}" for i in range(len(labels_sorted)))
    log(header)
    log(f"{'الحقيقي \\ المتوقع':<{width + 6}}" + "".join(
        f"{l[:width // 2 + 1]:>{width // 2 + 2}}" for l in labels_sorted))
    log("-" * (width + 6 + (width // 2 + 2) * len(labels_sorted)))
    for label, row in zip(labels_sorted, cm):
        log(f"{label:<{width + 6}}" + "".join(
            f"{v:>{width // 2 + 2},}" for v in row))
    log("-" * (width + 6 + (width // 2 + 2) * len(labels_sorted)))

    log("")
    log(f"مجموع الصفوف في المصفوفة: {cm.sum():,}  (يجب أن يساوي {len(y_test):,})")
    log(f"القطر (التصنيفات الصحيحة): {np.trace(cm):,}  "
        f"({np.trace(cm) / cm.sum():.2%} من الاختبار)")
    log("")

    log("أكبر 10 مصادر للخطأ (FN: الفئة حقيقية وطُبِّقت كغيرها):")
    log(f"  {'الصنف الحقيقي':<36}{'FN':>10}")
    log("  " + "-" * 46)
    fns = []
    for i, label in enumerate(labels_sorted):
        fn = cm[i].sum() - cm[i][i]
        if fn:
            fns.append((label, int(fn)))
    for label, fn in sorted(fns, key=lambda kv: -kv[1])[:10]:
        log(f"  {label:<36}{fn:>10,}")
    log("")

    return model, y_pred, labels_sorted, cm


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("تقرير تدريب نموذج على عيّنة CIC-IDS2017")
    log(f"العيّنة: {rel(SAMPLE_PATH)}")
    log(f"التقسيم: {1 - TEST_SIZE:.0%} / {TEST_SIZE:.0%} stratify   |   random_state={RANDOM_STATE}")
    log("")

    df = load()
    df = check_duplicate_column(df)
    df = drop_duplicates(df)
    X_train, X_test, y_train, y_test = split(df)
    model, y_pred, labels_sorted, cm = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    log(rule())
    log(f"حُفظ التقرير: {rel(REPORT_PATH)}")
    log("لم يُحفظ أي نموذج (غير مطلوب). لم يُستخدم git.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nتم الحفظ: {rel(REPORT_PATH)} "
          f"({os.path.getsize(REPORT_PATH):,} بايت)")
    return 0


if __name__ == "__main__":
    sys.exit(main())