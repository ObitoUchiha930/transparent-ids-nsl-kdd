"""تحليل أخطاء كشف هجمات R2L في NSL-KDD: لماذا تفوت النموذج رغم أنه مدرب عليها؟"""
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier

from classify_attacks import (
    CATEGORICAL,
    CLASS_INDEX,
    CLASSES,
    COLUMNS,
    LABEL_TO_CLASS,
    OUT,
    encode,
    load_multiclass,
)

SEED = 42
SHAP_SAMPLE = 200
TOP_FEATURES = 10
TOP_SHAP = 5

REPORT_TXT = os.path.join(OUT, "r2l_error_analysis.txt")
COMPARE_PNG = os.path.join(OUT, "shap_r2l_missed_vs_caught.png")

ROOT = os.path.dirname(os.path.abspath(__file__))


def rel(path):
    """مسار نسبي بالنسبة لجذر المشروع (يمنع طباعة اسم المستخدم والمسار الكامل)."""
    return os.path.relpath(path, ROOT).replace("\\", "/")

NORMAL_I = CLASS_INDEX["normal"]
R2L_I = CLASS_INDEX["R2L"]
NUMERIC = [c for c in COLUMNS[:41] if c not in CATEGORICAL]
HEADER = "feature"


def to_scalar(value):
    """يحوّل قيمة عدادية قد تكون مصفوفة طولها 1 إلى float."""
    arr = np.asarray(value, dtype=float).ravel()
    return float(arr[0]) if arr.size == 1 else float(arr.mean())


def cohen_d(a, b):
    """حجم الفارق المعياري بين متوسطين بمقياس موحّد لا يتأثر بوحدات الميزة."""
    na, nb = len(a), len(b)
    pooled = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def main():
    X_train, y_train, raw_train = load_multiclass("KDDTrain+.txt")
    X_test, y_test, raw_test = load_multiclass("KDDTest+.txt")
    X_train, X_test = encode(X_train, X_test)

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    out("=" * 104)
    out("تحليل أخطاء كشف هجمات R2L على NSL-KDD")
    out(f"التاريخ: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out("=" * 104)
    out("النموذج: HistGradientBoostingClassifier(class_weight='balanced', "
        f"random_state={SEED})")
    out(f"التدريب: {X_train.shape[0]:,} سطر | الاختبار: {X_test.shape[0]:,} سطر")

    # 1) التدريب بنفس إعدادات improve_rare.py
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
    model.fit(X_train, y_train)
    out("تم التدريب بنفس إعدادات improve_rare.py (class_weight='balanced').")

    # 2) تقسيم عيّنات R2L الحقيقية إلى ملتقطة وفائتة
    pred = model.predict(X_test)
    is_r2l = y_test.to_numpy() == R2L_I
    is_normal_pred = pred == NORMAL_I
    is_r2l_pred = pred == R2L_I
    other = is_r2l & ~(is_normal_pred | is_r2l_pred)

    idx_r2l = np.flatnonzero(is_r2l)
    idx_caught = idx_r2l[is_r2l_pred[idx_r2l]]
    idx_missed = idx_r2l[is_normal_pred[idx_r2l]]
    idx_other = idx_r2l[other[idx_r2l]]
    idx_normal_true = np.flatnonzero(y_test.to_numpy() == NORMAL_I)

    out("\n" + "=" * 104)
    out("1) تقسيم عينات R2L الحقيقية")
    out("=" * 104)
    out(f"  إجمالي R2L في الاختبار        : {len(idx_r2l):,}")
    out(f"  ملتقطة (تنبأ بها R2L)          : {len(idx_caught):,}  "
        f"({len(idx_caught) / len(idx_r2l):.1%})")
    out(f"  فائتة (تنبأ بها normal)        : {len(idx_missed):,}  "
        f"({len(idx_missed) / len(idx_r2l):.1%})")
    out(f"  مصنفة فئات أخرى (DoS/Probe/U2R): {len(idx_other):,}")
    out(f"  (للمرجع: عينات normal الحقيقية = {len(idx_normal_true):,})")

    X_missed = X_test.iloc[idx_missed]
    X_caught = X_test.iloc[idx_caught]
    X_normal = X_test.iloc[idx_normal_true]

    # 3) أكثر 10 ميزات اختلافًا بين الفائتة والملتقطة (توحيد المقياس)
    out("\n" + "=" * 104)
    out("2) أكثر 10 ميزات رقمية اختلافًا بين الفائتة والملتقطة (بمقياس Cohen's d)")
    out("=" * 104)
    out(f"{'#':>3}  {HEADER:<28}{'normal':>13}{'ملتقطة':>13}{'فائتة':>13}"
        f"{'فائتة-ملتقطة':>16}{'d':>12}")
    out("-" * 104)
    rows = []
    for feat in NUMERIC:
        m_miss = X_missed[feat].to_numpy()
        m_caught = X_caught[feat].to_numpy()
        d = cohen_d(m_miss, m_caught)
        rows.append({
            HEADER: feat,
            "mean_normal": float(X_normal[feat].mean()),
            "mean_caught": float(m_caught.mean()),
            "mean_missed": float(m_miss.mean()),
            "diff": float(m_miss.mean() - m_caught.mean()),
            "cohen_d": d,
            "abs_d": abs(d),
        })
    rows.sort(key=lambda r: r["abs_d"], reverse=True)
    for rank, r in enumerate(rows[:TOP_FEATURES], start=1):
        out(f"{rank:>3}  {r[HEADER]:<28}{r['mean_normal']:>13.4f}{r['mean_caught']:>13.4f}"
            f"{r['mean_missed']:>13.4f}{r['diff']:>16.4f}{r['cohen_d']:>12.4f}")

    zero_missed = float((X_missed["src_bytes"] == 0).mean())
    zero_caught = float((X_caught["src_bytes"] == 0).mean())
    zero_normal = float((X_normal["src_bytes"] == 0).mean())
    out("-" * 104)
    out(f"نسبة src_bytes == 0  ->  فائتة: {zero_missed:.1%} | "
        f"ملتقطة: {zero_caught:.1%} | normal: {zero_normal:.1%}")

    # 4) SHAP على 200 فائتة + 200 ملتقطة
    out("\n" + "=" * 104)
    out("3) تحليل SHAP (200 فائتة + 200 ملتقطة)")
    out("=" * 104)
    rng = np.random.RandomState(SEED)
    sel_missed = rng.choice(idx_missed, size=min(SHAP_SAMPLE, len(idx_missed)), replace=False)
    sel_caught = rng.choice(idx_caught, size=min(SHAP_SAMPLE, len(idx_caught)), replace=False)
    Xs_missed = X_test.iloc[sel_missed]
    Xs_caught = X_test.iloc[sel_caught]
    out(f"  عينة الفائتة: {len(sel_missed)} | عينة الملتقطة: {len(sel_caught)}")

    explainer = shap.TreeExplainer(model)
    res_missed = explainer(Xs_missed)
    res_caught = explainer(Xs_caught)
    sv_m = np.asarray(res_missed.values)
    sv_c = np.asarray(res_caught.values)
    base_m = np.asarray(res_missed.base_values)

    if sv_m.ndim == 3:  # متعدد الأصناف: (n, features, classes)
        dir_missed = sv_m[:, :, NORMAL_I]
        dir_caught = sv_c[:, :, NORMAL_I]
        base_normal = float(base_m[0, NORMAL_I])
        raw = model.decision_function(Xs_missed)[:, NORMAL_I]
        recon = base_m[:, NORMAL_I] + dir_missed.sum(axis=1)
        err = float(np.abs(recon - raw).max())
        out(f"  النموذج متعدد الأصناف ({sv_m.shape[2]} أصناف) -> قِسنا فئة normal "
            f"(فهرس {NORMAL_I}).")
        out(f"  فحص الجمع (additivity): خطأ أقصاه = {err:.2e} — أي أن base + مجموع SHAP "
            f"= decision_function(normal) بدقة الآلة.")
        out(f"  ملاحظة: القيم في فضاء log-odds وليس احتمالات؛ موجب = يدفع نحو normal.")
    else:  # ثنائي الأخراج
        base = to_scalar(explainer.expected_value)
        proba = model.predict_proba(Xs_missed.iloc[[0]])[0]
        logit_attack = float(np.log(proba[1] / proba[0]))
        recon0 = base + float(sv_m[0].sum())
        values_are_attack = abs(recon0 - logit_attack) <= abs(recon0 + logit_attack)
        dir_missed = -sv_m if values_are_attack else sv_m
        dir_caught = -sv_c if values_are_attack else sv_c
        base_normal = -base if values_are_attack else base
        out("  النموذج ثنائي الأخراج -> تم تحويل القيم إلى اتجاه normal.")

    cols = list(X_test.columns)
    mean_missed = pd.Series(dir_missed.mean(axis=0), index=cols).sort_values(ascending=False)
    mean_caught = pd.Series(dir_caught.mean(axis=0), index=cols).sort_values(ascending=False)

    out(f"\n  متوسط قيمة SHAP لفئة normal، قيمة الأساس = {base_normal:+.4f}")
    out(f"\n  {'#':>3}  {'أهم ميزات أبقت الفائتة على normal':<34}{'فائتة':>12}"
        f"{'ملتقطة':>12}{'الفارق':>12}")
    out("  " + "-" * 100)
    for rank, (feat, val) in enumerate(mean_missed.head(TOP_SHAP).items(), start=1):
        out(f"  {rank:>3}  {feat:<34}{val:>+12.4f}{mean_caught[feat]:>+12.4f}"
            f"{val - mean_caught[feat]:>+12.4f}")

    out("\n  العكس — ما الذي جعل الملتقطة هجومًا (سالب = يدفع بعيدًا عن normal):")
    for rank, feat in enumerate(mean_caught.head(TOP_SHAP).index, start=1):
        out(f"  {rank:>3}  {feat:<34}{mean_caught[feat]:>+12.4f}{mean_missed[feat]:>+12.4f}"
            f"{mean_caught[feat] - mean_missed[feat]:>+12.4f}")

    # الرسم المقارن
    top_union = list(dict.fromkeys(list(mean_missed.head(12).index)
                                  + list(mean_caught.head(12).index)))
    y_pos = np.arange(len(top_union))
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(17, max(7.0, len(top_union) * 0.45)),
        gridspec_kw={"width_ratios": [1.15, 1]})
    h = 0.38
    ax1.barh(y_pos + h / 2, [mean_missed[f] for f in top_union], height=h,
             label=f"missed -> normal (n={len(sel_missed)})", color="#d1495b")
    ax1.barh(y_pos - h / 2, [mean_caught[f] for f in top_union], height=h,
             label=f"caught -> R2L (n={len(sel_caught)})", color="#00798c")
    ax1.set_yticks(y_pos, top_union)
    ax1.invert_yaxis()
    ax1.axvline(0, color="black", lw=1)
    ax1.set_xlabel("mean SHAP (positive = pushes toward normal)")
    ax1.set_title("R2L: why do some attacks slip through as normal?")
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", alpha=0.3)

    shap.plots.beeswarm(
        shap.Explanation(
            values=dir_missed,
            base_values=np.full(len(sel_missed), base_normal),
            data=Xs_missed.to_numpy(),
            feature_names=cols),
        max_display=12, show=False, plot_size=None, ax=ax2)
    ax2.set_title(f"{len(sel_missed)} missed R2L — SHAP of class 'normal'")
    fig.tight_layout()
    fig.savefig(COMPARE_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out(f"\n  تم حفظ الرسم المقارن: {rel(COMPARE_PNG)}")

    # 5) أنواع هجمات R2L التي تتركز فيها الفائتة
    out("\n" + "=" * 104)
    out("4) أنواع هجمات R2L التي تتركز فيها الفائتة (من العمود label الأصلي)")
    out("=" * 104)
    # LABEL_TO_CLASS: اسم الهجوم -> اسم العائلة، ثم العائلة -> فهرس الصنف
    class_index = {c: i for i, c in enumerate(CLASSES)}
    train_r2l_types = set(
        raw_train[raw_train.map(LABEL_TO_CLASS).map(class_index) == R2L_I].unique())
    train_r2l_counts = raw_train[raw_train.map(LABEL_TO_CLASS).map(class_index) == R2L_I] \
        .value_counts().to_dict()
    out(f"  أنواع R2L الموجودة في التدريب: {len(train_r2l_types)} ({', '.join(sorted(train_r2l_types))})")
    out("  أعدادها في التدريب: " + ", ".join(
        f"{k}={train_r2l_counts.get(k, 0):,}" for k in sorted(train_r2l_types)))
    r2l_types = raw_test.iloc[idx_r2l]
    missed_types = raw_test.iloc[idx_missed]
    type_rows = []
    for attack in sorted(r2l_types.unique()):
        tot = int((r2l_types == attack).sum())
        mis = int((missed_types == attack).sum())
        type_rows.append({
            "attack": attack,
            "train_count": int(train_r2l_counts.get(attack, 0)),
            "total": tot,
            "missed": mis,
            "caught": tot - mis,
            "miss_rate": mis / tot if tot else 0.0,
            "in_train": attack in train_r2l_types,
        })
    type_rows.sort(key=lambda r: (-r["train_count"], -r["missed"]))

    out(f"{'#':>3}  {'نوع الهجوم':<20}{'في التدريب؟':<14}{'تدريب':>9}{'اختبار':>10}"
        f"{'ملتقطة':>10}{'فائتة':>10}{'نسبة الفائت':>13}")
    out("-" * 104)
    for rank, r in enumerate(type_rows, start=1):
        mark = "نعم" if r["in_train"] else "* لا (جديد)"
        out(f"{rank:>3}  {r['attack']:<20}{mark:<14}{r['train_count']:>9,}{r['total']:>10,}"
            f"{r['caught']:>10,}{r['missed']:>10,}{r['miss_rate']:>12.1%}")

    novel = [r for r in type_rows if not r["in_train"]]
    seen = [r for r in type_rows if r["in_train"]]
    novel_missed = sum(r["missed"] for r in novel)
    novel_total = sum(r["total"] for r in novel)
    seen_missed = sum(r["missed"] for r in seen)
    seen_total = sum(r["total"] for r in seen)
    tot_all = sum(r["total"] for r in type_rows)
    mis_all = sum(r["missed"] for r in type_rows)

    out("-" * 104)
    out(f"{'المجموع':<24}{'':<14}"
        f"{sum(r['train_count'] for r in type_rows):>9,}{tot_all:>10,}"
        f"{tot_all - mis_all:>10,}{mis_all:>10,}{mis_all / tot_all:>12.1%}")

    out(f"\n  أنواع موجودة في التدريب: {seen_total:,} عينة، فائتة {seen_missed:,} "
        f"({seen_missed / seen_total:.1%})" if seen_total else
        "\n  أنواع موجودة في التدريب: 0 عينة")
    out(f"  أنواع جديدة على النموذج: {novel_total:,} عينة، فائتة {novel_missed:,} "
        f"({novel_missed / novel_total:.1%})" if novel_total else
        "  أنواع جديدة على النموذج: 0 عينة")
    if novel:
        out("  الأنواع الجديدة: " + ", ".join(r["attack"] for r in novel))

    out("\n" + "=" * 104)
    out("5) الخلاصة")
    out("=" * 104)
    worst = type_rows[0]
    out(f"  • أسوأ نوع: {worst['attack']} بفائتة {worst['miss_rate']:.1%} "
        f"({worst['missed']:,} من {worst['total']:,}).")
    if novel_missed > seen_missed:
        out(f"  • الفائتة تتركز في الأنواع الجديدة ({novel_missed:,} مقابل {seen_missed:,} "
            f"للأنواع المألوفة) فالسبب تعميم لا ضعف سعة.")
    else:
        out(f"  • الفائتة تتركز في الأنواع المألوفة ({seen_missed:,} مقابل {novel_missed:,}) "
            f"فالسبب ضعف تمثيل الميزات لا غياب بيانات الأنواع.")
    out(f"  • src_bytes == 0 في {zero_missed:.1%} من الفائتة مقابل {zero_caught:.1%} "
        f"من الملتقطة.")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "=" * 104)
    print(f"تم الحفظ: {REPORT_TXT}")
    print(f"تم حفظ الرسم: {COMPARE_PNG}")


if __name__ == "__main__":
    main()
