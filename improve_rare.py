"""تحسين أداء الفئات النادرة (R2L / U2R) في تصنيف هجمات NSL-KDD عبر اختيار إعدادات بـ StratifiedKFold.

منهج صارم: اختيار النموذج يتم على التحقق المتقاطع داخل بيانات التدريب فقط،
ثم يُقيَّم مرة واحدة على KDDTest+ إلى جانب خط الأساس الافتراضي.
"""
import os
from datetime import datetime

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from classify_attacks import CLASSES, OUT, encode, load_multiclass

SEED = 42
N_SPLITS = 5
REPORT_TXT = os.path.join(OUT, "improve_rare_report.txt")


def per_class_table(y_true, pred):
    """مقاييس لكل فئة من مصفوفة الارتباك: precision / recall / f1 / FN."""
    cm = confusion_matrix(y_true, pred, labels=list(range(len(CLASSES))))
    rows = []
    for i, cls in enumerate(CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "class": cls,
            "support": int(cm[i].sum()),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "fn": fn,
        })
    return rows, cm


def cross_validate(X, y, model, label):
    """macro-F1 عبر StratifiedKFold على بيانات التدريب فقط."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = []
    print(f"  [{label}]")
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        score = f1_score(y.iloc[va], pred, average="macro", zero_division=0)
        scores.append(score)
        print(f"    fold {fold}/{N_SPLITS}: macro-F1 = {score:.4f}")
    mean = sum(scores) / len(scores)
    spread = max(scores) - min(scores)
    print(f"    mean macro-F1 = {mean:.4f}  (spread {spread:.4f})")
    return mean, spread


def main():
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    X_train, y_train, _ = load_multiclass("KDDTrain+.txt")
    X_test, y_test, _ = load_multiclass("KDDTest+.txt")
    X_train, X_test = encode(X_train, X_test)

    out("=" * 100)
    out("تحسين الفئات النادرة في تصنيف هجمات NSL-KDD")
    out(f"التاريخ: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out("=" * 100)
    out(f"النموذج : HistGradientBoostingClassifier")
    out(f"الاختيار: StratifiedKFold(n_splits={N_SPLITS}, shuffle=True) على التدريب فقط")
    out(f"المقياس : macro-F1 (يعاقب الفئات الضعيفة DoS/Probe/R2L/U2R بالتساوي)")
    out(f"التدريب : {X_train.shape[0]:,} سطر × {X_train.shape[1]} ميزة")
    out(f"الاختبار: {X_test.shape[0]:,} سطر × {X_test.shape[1]} ميزة")
    out("قاعدة صارمة: KDDTest+ لم يُستخدم في أي قرار اختيار.")

    configs = [
        ("أ) الافتراضي", {}),
        ("ب) class_weight='balanced'", {"class_weight": "balanced"}),
        ("ج) balanced + min_samples_leaf=5",
         {"class_weight": "balanced", "min_samples_leaf": 5}),
    ]

    # 1) اختيار الإعدادات على بيانات التدريب فقط
    out("\n" + "=" * 100)
    out("المرحلة 1 — التحقق المتقاطع (StratifiedKFold 5) على بيانات التدريب فقط")
    out("=" * 100)
    cv_rows = []
    for label, params in configs:
        out(f"\nالإعدادات {label}")
        model = HistGradientBoostingClassifier(random_state=SEED, **params)
        mean, spread = cross_validate(X_train, y_train, model, label)
        cv_rows.append({"config": label, "params": params, "cv_macro_f1": round(mean, 4),
                        "fold_spread": round(spread, 4)})

    out("\n--- ملخص التحقق المتقاطع ---")
    out(f"{'الإعدادات':<38}{'macro-F1 (CV)':>16}{'الانتشار بين الطيات':>22}")
    out("-" * 100)
    for r in cv_rows:
        out(f"{r['config']:<38}{r['cv_macro_f1']:>16.4f}{r['fold_spread']:>22.4f}")

    best = max(cv_rows, key=lambda r: r["cv_macro_f1"])
    out(f"\nالأفضل بـ CV: {best['config']}  (macro-F1 = {best['cv_macro_f1']:.4f})")
    out(f"معاملات مختارة: {best['params'] if best['params'] else 'الافتراضي'}")

    # 2) التدريب على كامل التدريب: خط الأساس + المختار
    out("\n" + "=" * 100)
    out("المرحلة 2 — التقييم على KDDTest+ (مرة واحدة لكل نموذج)")
    out("=" * 100)

    baseline = HistGradientBoostingClassifier(random_state=SEED)
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    base_acc = accuracy_score(y_test, base_pred)
    base_rows, base_cm = per_class_table(y_test, base_pred)
    base_f1 = {r["class"]: r for r in base_rows}
    out(f"\nخط الأساس (الإعدادات الافتراضية) — accuracy = {base_acc:.4f}")
    out(classification_report(y_test, base_pred, labels=list(range(len(CLASSES))),
                              target_names=CLASSES, digits=4, zero_division=0))

    winner = HistGradientBoostingClassifier(random_state=SEED, **best["params"])
    winner.fit(X_train, y_train)
    win_pred = winner.predict(X_test)
    win_acc = accuracy_score(y_test, win_pred)
    win_rows, win_cm = per_class_table(y_test, win_pred)
    win_f1 = {r["class"]: r for r in win_rows}
    out(f"\nالمختار من CV — {best['config']} — accuracy = {win_acc:.4f}")
    out(classification_report(y_test, win_pred, labels=list(range(len(CLASSES))),
                              target_names=CLASSES, digits=4, zero_division=0))

    # 3) جدول المقارنة الواحد
    out("\n" + "=" * 100)
    out("الجدول المقارن — خط الأساس الافتراضي  vs  المختار من التحقق المتقاطع")
    out("=" * 100)
    out(f"{'الفئة':<9}{'دعم':>8}{'precision':>12}{'recall':>10}{'F1':>9}{'FN':>8}"
        f"{'│':>3}{'precision':>12}{'recall':>10}{'F1':>9}{'FN':>8}{'ΔF1':>9}")
    out("-" * 100)
    for cls in CLASSES:
        b, w = base_f1[cls], win_f1[cls]
        delta = w["f1"] - b["f1"]
        arrow = "+" if delta > 0 else ("=" if delta == 0 else "")
        out(f"{cls:<9}{w['support']:>8,}{b['precision']:>12.4f}{b['recall']:>10.4f}"
            f"{b['f1']:>9.4f}{b['fn']:>8,}{'│':>3}{w['precision']:>12.4f}"
            f"{w['recall']:>10.4f}{w['f1']:>9.4f}{w['fn']:>8,}{arrow}{delta:>+8.4f}")

    base_macro = sum(r["f1"] for r in base_rows) / len(base_rows)
    win_macro = sum(r["f1"] for r in win_rows) / len(win_rows)
    out("-" * 100)
    out(f"{'macro':<9}{X_test.shape[0]:>8,}{'':>12}{'':>10}{base_macro:>9.4f}{'':>8}"
        f"{'│':>3}{'':>12}{'':>10}{win_macro:>9.4f}{'':>8}{'+' if win_macro > base_macro else ''}"
        f"{win_macro - base_macro:>+8.4f}")
    out(f"{'accuracy':<9}{'':>8}{'':>12}{'':>10}{base_acc:>9.4f}{'':>8}{'│':>3}"
        f"{'':>12}{'':>10}{win_acc:>9.4f}{'':>8}"
        f"{'+' if win_acc > base_acc else ''}{win_acc - base_acc:>+8.4f}")

    # 4) مصفوفتا الارتباك
    out("\n--- مصفوفة الارتباك: خط الأساس (يسار) ---")
    out("        " + "".join(f"{c:>9}" for c in CLASSES) + f"{'total':>11}")
    for i, cls in enumerate(CLASSES):
        out(f"{cls:<8}" + "".join(f"{base_cm[i, j]:>9,}" for j in range(len(CLASSES)))
            + f"{base_cm[i].sum():>11,}")
    out("\n--- مصفوفة الارتباك: المختار (يمين) ---")
    out("        " + "".join(f"{c:>9}" for c in CLASSES) + f"{'total':>11}")
    for i, cls in enumerate(CLASSES):
        out(f"{cls:<8}" + "".join(f"{win_cm[i, j]:>9,}" for j in range(len(CLASSES)))
            + f"{win_cm[i].sum():>11,}")

    # 5) الحكم
    out("\n" + "=" * 100)
    out("الحكم على الفئات النادرة")
    out("=" * 100)
    for cls in ["R2L", "U2R"]:
        b, w = base_f1[cls], win_f1[cls]
        if w["f1"] > b["f1"]:
            verdict = f"تحسّن (+{w['f1'] - b['f1']:.4f}): FN {b['fn']:,} → {w['fn']:,}"
        elif w["f1"] < b["f1"]:
            verdict = f"تراجع ({w['f1'] - b['f1']:+.4f}): FN {b['fn']:,} → {w['fn']:,}"
        else:
            verdict = f"بلا تغيير: FN {b['fn']:,}"
        out(f"  {cls:<6} F1 {b['f1']:.4f} → {w['f1']:.4f}   {verdict}")
    out(f"\n  accuracy: {base_acc:.4f} → {win_acc:.4f}")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "=" * 100)
    print(f"تم الحفظ: {REPORT_TXT}")


if __name__ == "__main__":
    main()
