"""
تحقق متقاطع (5 طيات) + كشف تعارض التسميات في عيّنة CIC-IDS2017.

- يستورد تنظيف train_cicids.py نفسه (حذف العمود المكرر ثم حذف المتكررات) بدل إعادة كتابته.
- الجزء 1: StratifiedKFold بخمس طيات + تنبؤات خارج الطية + مقاييس مجمّعة + متوسط/انحراف F1 لكل فئة بين الطيات.
- الجزء 2: صفوف ميزاتها متطابقة تمامًا وتسمياتها مختلفة.
- لا يحفظ أي نموذج. لا يستخدم git.
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

# استيراد نفس تنظيف train_cicids.py (لا نعدّل ذلك الملف، بل نستورد منه)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_cicids as base  # noqa: E402

# ---------------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(ROOT, "data", "cicids2017", "sample.csv")
OUT_DIR = os.path.join(ROOT, "output")
REPORT_PATH = os.path.join(OUT_DIR, "cicids_cv_report.txt")

LABEL_COL = "Label"
DUP_COL = "Fwd Header Length.1"
N_SPLITS = 5
RANDOM_STATE = 42


def rel(path):
    """مسار نسبي بالنسبة لجذر المشروع (يمنع طباعة اسم المستخدم والمسار الكامل)."""
    return os.path.relpath(path, ROOT).replace("\\", "/")

_lines = []


def log(text=""):
    print(text)
    _lines.append(str(text))


def rule(char="=", width=100):
    return char * width


# نجعل دالة log داخل train_cicids.py تكتب في تقريرنا نحن
base.log = log


# ---------------------------------------------------------------------------
# التنظيف (مستورد من train_cicids.py)
# ---------------------------------------------------------------------------
def prepare_data():
    log(rule())
    log("0) التنظيف — نفس دوال train_cicids.py")
    log(rule())

    df = pd.read_csv(SAMPLE_PATH, encoding="utf-8")
    log(f"الملف     : {rel(SAMPLE_PATH)}")
    log(f"الحجم     : {os.path.getsize(SAMPLE_PATH):,} بايت")
    log(f"صفوف      : {len(df):,}")
    log(f"أعمدة     : {df.shape[1]}")
    log(f"فئات      : {df[LABEL_COL].nunique()}")

    # 1) العمود المكرر
    if DUP_COL in df.columns:
        log("")
        log(f"«{DUP_COL}» موجود ← حذفه")
        df = df.drop(columns=[DUP_COL])
    else:
        log("")
        log(f"«{DUP_COL}» غير موجود")
    log(f"أعمدة بعد حذف العمود: {df.shape[1]}")

    # 2) الصفوف المكررة — نفس دالة train_cicids.py
    df = base.check_duplicate_column(df)
    df = base.drop_duplicates(df)
    log("")
    return df


# ---------------------------------------------------------------------------
# الجزء 1: التحقق المتقاطع
# ---------------------------------------------------------------------------
def cross_validate(X, y):
    log("")
    log(rule())
    log(f"الجزء 1) StratifiedKFold — {N_SPLITS} طيات، shuffle=True, random_state={RANDOM_STATE}")
    log(rule())

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    labels = sorted(y.unique())
    log(f"X: {X.shape}   |   y: {y.shape}   |   عدد الفئات: {len(labels)}")
    log("")

    oof_pred = np.empty(len(y), dtype=object)
    fold_f1 = np.zeros((N_SPLITS, len(labels)))
    fold_sizes = []
    fold_times = []

    import time

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        elapsed = time.perf_counter() - t0

        oof_pred[test_idx] = pred
        scores = f1_score(y_te, pred, labels=labels, average=None, zero_division=0)
        fold_f1[fold - 1] = scores
        fold_sizes.append(len(test_idx))
        fold_times.append(elapsed)

        fold_acc = accuracy_score(y_te, pred)
        fold_macro = f1_score(y_te, pred, average="macro", zero_division=0)
        log(f"الطيّة {fold}: تدريب={len(X_tr):,}  اختبار={len(test_idx):,}"
            f"  accuracy={fold_acc:.4f}  macro-F1={fold_macro:.4f}"
            f"  الزمن={elapsed:.1f}s")

        del model, pred

    log("")
    log(f"حجم كل طيّة اختبار: {fold_sizes}   (مجموعها {sum(fold_sizes):,})")
    log(f"كل صف تغطّي طيّة واحدة بالضبط: {len(oof_pred) == len(y)}")

    # ---- مقاييس مجمّعة من التنبؤات خارج الطية ----
    log("")
    log(rule("-"))
    log("مقاييس مجمّعة على كل التنبؤات خارج الطية (out-of-fold)")
    log(rule("-"))
    acc = accuracy_score(y, oof_pred)
    macro = f1_score(y, oof_pred, average="macro", zero_division=0)
    weighted = f1_score(y, oof_pred, average="weighted", zero_division=0)
    log(f"دقة_accuracy : {acc:.4f}")
    log(f"macro-F1     : {macro:.4f}")
    log(f"weighted-F1  : {weighted:.4f}")
    log("")

    log(rule("-"))
    log("جدول precision / recall / F1 / support لكل فئة (من التنبؤات المجمّعة خارج الطية)")
    log(rule("-"))
    report = classification_report(
        y, oof_pred, labels=labels, digits=4, zero_division=0,
    )
    for line in report.splitlines():
        log("  " + line if line.strip() else "")
    log("")

    # ---- متوسط F1 وانحرافه المعياري بين الطيات لكل فئة ----
    log(rule("-"))
    log("متوسط F1 عبر الطيات مع الانحراف المعياري لكل فئة")
    log(rule("-"))
    log(f"{'الفئة':<30}{'المتوسط':>12}{'الانحراف':>12}{'الأدنى':>10}{'الأعلى':>10}")
    log("-" * 74)
    for i, label in enumerate(labels):
        col = fold_f1[:, i]
        log(f"{label:<30}{col.mean():>12.4f}{col.std(ddof=0):>12.4f}"
            f"{col.min():>10.4f}{col.max():>10.4f}")
    log("-" * 74)
    log(f"{'macro-F1 لكل طيّة':<30}"
        f"{fold_f1.mean(axis=1).mean():>12.4f}"
        f"{fold_f1.mean(axis=1).std(ddof=0):>12.4f}"
        f"{fold_f1.mean(axis=1).min():>10.4f}"
        f"{fold_f1.mean(axis=1).max():>10.4f}")
    log("")
    log("ملاحظة: الانحراف المعياري محسوب بـ ddof=0 (مجتمعات الطيات لا عيّنة عشوائية).")
    log("")

    return oof_pred, labels, fold_f1


# ---------------------------------------------------------------------------
# الجزء 2: تعارض التسميات
# ---------------------------------------------------------------------------
def find_label_conflicts(df):
    log(rule())
    log("الجزء 2) صفوف ميزاتها متطابقة تمامًا وتسمياتها مختلفة")
    log(rule())

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    log(f"عدد أعمدة الميزات: {len(feature_cols)} (كل الأعمدة ما عدا {LABEL_COL})")
    log(f"عدد الصفوف المفحوصة: {len(df):,}")
    log("")

    # تجميع سريع بتجزئة 64-بت، ثم تحقّق فعلي بالجدولة على القيم
    hashes = pd.util.hash_pandas_object(df[feature_cols], index=False)
    tmp = pd.DataFrame({"h": hashes.to_numpy(),
                        "y": df[LABEL_COL].to_numpy()})
    per_hash = tmp.groupby("h").agg(n=("y", "size"), k=("y", "nunique"))
    candidate_hashes = per_hash.index[per_hash["k"] > 1]
    log(f"صفوف ضمن تجزئة تحتوي أكثر من تسمية: "
        f"{int(per_hash.loc[candidate_hashes, 'n'].sum()) if len(candidate_hashes) else 0:,}")

    if len(candidate_hashes) == 0:
        log("")
        log("النتيجة: لا توجد مجموعات متعارضة.")
        return 0, Counter(), set()

    # تحقّق من التصادمات المحتملة بإعادة التجميع على القيم الفعلية
    sub = df.loc[hashes.isin(set(candidate_hashes))]
    keys = sub[feature_cols].astype(str).agg("|".join, axis=1)
    keys = keys.str.replace(r"\s+", " ", regex=True)

    grouped = pd.DataFrame({
        "key": keys.to_numpy(),
        "y": sub[LABEL_COL].to_numpy(),
    })
    counts = grouped.groupby(["key", "y"]).size().reset_index(name="count")
    n_labels_per_key = counts.groupby("key")["y"].nunique()

    conflicting_keys = n_labels_per_key[n_labels_per_key > 1].index
    conflict_counts = counts[counts["key"].isin(set(conflicting_keys))]

    # خريطة: لكل مجموعة متعارضة، {التسمية: عدد صفوفها}
    rows_by_label = {}
    for key, group in grouped[grouped["key"].isin(set(conflicting_keys))].groupby("key"):
        rows_by_label[key] = dict(zip(group["y"], group["count"]))

    # أزواج التسميات: لكل زوج، min(عدد كل تسمية) = أقل عدد يمكن إصلاحه بإعادة وسم المجموعة
    pair_rows = Counter()
    for d in rows_by_label.values():
        labs = sorted(d)
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                pair_rows[(labs[i], labs[j])] += min(d[labs[i]], d[labs[j]])

    n_groups = len(rows_by_label)
    n_conflict_rows = int(conflict_counts["count"].sum())

    log(f"عدد المجموعات المتعارضة (ميزاتها متطابقة وتسمياتها مختلفة): {n_groups:,}")
    log(f"عدد الصفوف داخل هذه المجموعات: {n_conflict_rows:,}")
    log("")

    log(rule("-"))
    log("أزواج التسميات المتعارضة مع عدد الصفوف المتأثرة في كل زوج")
    log(rule("-"))
    log(f"{'التسمية 1':<30}{'التسمية 2':<30}{'عدد الصفوف':>14}")
    log("-" * 74)
    for (a, b), n in sorted(pair_rows.items(), key=lambda kv: -kv[1]):
        log(f"{a:<30}{b:<30}{n:>14,}")
    log("-" * 74)
    log(f"{'المجموع':<60}{sum(pair_rows.values()):>14,}")
    log("")
    log("العدد لكل زوج = min(عدد التسمية الأولى، عدد التسمية الثانية) داخل المجموعة،")
    log("أي أصغر عدد يمكن إصلاحه بإعادة وسم المجموعة كلها على تسمية واحدة.")
    log("")

    # فحوص.required
    log(rule("-"))
    log("هل تتضمن التعارضات زوج Web Attack - Brute Force و Web Attack - XSS ؟")
    log(rule("-"))
    target = ("Web Attack - Brute Force", "Web Attack - XSS")
    n_target = pair_rows.get(target, 0)
    log("الزوج المطلوب: Brute Force ↔ XSS")
    log(f"عدد الصفوف المتأثرة في هذا الزوج: {n_target:,}")
    log(f"هل هو ضمن الأزواج المتعارضة؟ {'نعم' if n_target > 0 else 'لا'}")
    log("")

    labels_in_conflict = set()
    for labs in rows_by_label.values():
        labels_in_conflict.update(labs)
    log(f"عدد التسميات المتورطة في أي تعارض: {len(labels_in_conflict)}")
    log(f"{'التسمية':<40}{'هل متورطة'}")
    log("-" * 52)
    for label in sorted(labels_in_conflict):
        log(f"{label:<40}{'نعم':>10}")
    log("")
    return n_groups, pair_rows, labels_in_conflict


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("تقرير التحقق المتقاطع وتعارض التسميات — عيّنة CIC-IDS2017")
    log(f"العيّنة: {rel(SAMPLE_PATH)}")
    log(f"الطيّات: {N_SPLITS}   |   random_state={RANDOM_STATE}")
    log("النموذج: HistGradientBoostingClassifier(class_weight='balanced')")
    log("")

    df = prepare_data()
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    cross_validate(X, y)
    find_label_conflicts(df)

    log(rule())
    log(f"حُفظ التقرير: {rel(REPORT_PATH)}")
    log("لم يُحفظ أي نموذج. لم يُستخدم git.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nتم الحفظ: {rel(REPORT_PATH)} ({os.path.getsize(REPORT_PATH):,} بايت)")
    return 0


if __name__ == "__main__":
    sys.exit(main())