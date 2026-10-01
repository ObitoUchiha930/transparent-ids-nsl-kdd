"""
تحليل الفئتين Web Attack - Brute Force و Web Attack - XSS في عيّنة CIC-IDS2017.

المدخل: data/cicids2017/sample.csv بعمود التسمية Label، موحَّداً كما في make_cicids_sample.py.

الأجزاء:
1) حجم كل فئة في العيّنة (قبل وبعد حذف الصفوف المكرّرة تمامًا).
2) تصنيف ثنائي لهاتين الفئتين: HistGradientBoostingClassifier(class_weight='balanced')
   مع StratifiedKFold(5, shuffle=True)، ومقاييس مجمّعة من التنبؤات خارج الطيّة + مصفوفة 2x2.
3) NearestNeighbors بعد StandardScaler يُضبط داخل كل طيّة على بيانات التدريب فقط.
4) أكثر 10 ميزات فصلاً: |Cohen's d| على البيانات كلها مع متوسطي الفئتين.
5) عدد الصفوف المتطابقة تماماً في الميزات بين الفئتين (محسوبة على الصفوف قبل حذف المكرّر).

البذرة 42 في كل مكان. لا يحفظ أي نموذج. لا يستخدم git.
المخرج: output/bf_xss_report.txt بترميز UTF-8.
"""

import os
import platform
import re
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(ROOT, "data", "cicids2017", "sample.csv")
OUT_DIR = os.path.join(ROOT, "output")
REPORT_PATH = os.path.join(OUT_DIR, "bf_xss_report.txt")

LABEL_COL = "Label"
DUP_COL = "Fwd Header Length.1"

CLASS_BF = "Web Attack - Brute Force"
CLASS_XSS = "Web Attack - XSS"
CLASSES = (CLASS_BF, CLASS_XSS)

N_SPLITS = 5
RANDOM_STATE = 42
TOP_K = 10
WIDTH = 100

_lines = []


def rel(path):
    """مسار نسبي بالنسبة لجذر المشروع (يمنع طباعة اسم المستخدم والمسار الكامل)."""
    return os.path.relpath(path, ROOT).replace("\\", "/")


def log(text=""):
    """يطبع السطر ويخزّنه في التقرير."""
    print(text)
    _lines.append(str(text))


def rule(char="=", width=WIDTH):
    return char * width


def clean_label(value):
    """نفس توحيد تسمية Web Attack المستخدَم في make_cicids_sample.py."""
    s = re.sub(r"\s+", " ", str(value)).strip()
    if s.startswith("Web Attack"):
        s = re.sub(r"[^\x00-\x7F]+", "-", s)   # أي محرف غير ASCII -> شرطة عادية
        s = re.sub(r"\s*-\s*", " - ", s)       # توحيد الفراغات حول الشرطة
        s = re.sub(r"\s+", " ", s).strip()
    return s


def table(header_cols, rows, widths, aligns=None):
    """يطبع جدولاً محاذياً؛ aligns قائمة '<' أو '>' لكل عمود، ومسافة فاصلة بين الخلايا."""
    aligns = aligns or ["<"] * len(widths)
    sep = " "
    lines = []
    lines.append("  " + sep.join(
        f"{h:{a}{w}}" for h, a, w in zip(header_cols, aligns, widths)))
    lines.append("  " + "".join("-" * w for w in widths))
    for row in rows:
        lines.append("  " + sep.join(
            f"{v:{a}{w}}" for v, a, w in zip(row, aligns, widths)))
    lines.append("  " + "".join("-" * w for w in widths))
    return lines


def confusion_2x2(cm):
    """مصفوفة 2x2 بمفتاح رقمي لتفادي أطوال التسميات."""
    out = []
    out.append(f"  مفتاح التسميات:  1 = {CLASS_BF}   |   2 = {CLASS_XSS}")
    out.append("")
    out.append(f"  {'الحقيقي \\ المتوقَّع':<28}{'1':>12}{'2':>12}{'المجموع':>12}")
    out.append("  " + "-" * 64)
    for i, label in enumerate(CLASSES, start=1):
        out.append(f"  {str(i) + '  ' + label:<28}"
                   + f"{cm[i - 1, 0]:>12,}{cm[i - 1, 1]:>12,}{cm[i - 1].sum():>12,}")
    out.append("  " + "-" * 64)
    out.append(f"  {'المجموع':<28}"
               + f"{cm[:, 0].sum():>12,}{cm[:, 1].sum():>12,}{cm.sum():>12,}")
    return out


# ---------------------------------------------------------------------------
# 0) التحميل والتنظيف
# ---------------------------------------------------------------------------
def load_and_filter():
    df = pd.read_csv(SAMPLE_PATH, encoding="utf-8")
    n_file = len(df)          # عدد صفوف الملف كما قُرئ (يُستخدم في السطر 9 وفي قسم الحقائق)

    log(rule())
    log("0) التحميل والتنظيف")
    log(rule())
    log(f"المصدر              : {rel(SAMPLE_PATH)}")
    log(f"الحجم بالبايت        : {os.path.getsize(SAMPLE_PATH):,}")
    log(f"صفوف الملف           : {n_file:,}")
    log(f"أعمدة الملف          : {df.shape[1]}")
    log(f"عمود التسمية         : {LABEL_COL}")
    log(f"عمود مكرر مُزال      : {DUP_COL}   (مطابق تمامًا لعمود Fwd Header Length)")
    log(f"عدد الفئات في الملف  : {df[LABEL_COL].nunique()}")
    log(f"البذرة (random_state): {RANDOM_STATE}")
    log(f"الطيّات              : {N_SPLITS}   (StratifiedKFold, shuffle=True)")
    log("النموذج              : HistGradientBoostingClassifier(class_weight='balanced')")
    log("مقياس المسافات       : StandardScaler يُضبط داخل كل طيّة على بيانات التدريب فقط")
    log("مقياس الجوار        : NearestNeighbors على كامل صفوف التدريب")
    log("مقياس الفصل          : Cohen's d  (التباين المشترك بين الفئتين، ddof=1)")
    log(f"إصدار بايثون         : {platform.python_version()}")
    log(f"إصدار pandas         : {pd.__version__}")
    log(f"إصدار numpy          : {np.__version__}")
    log(f"إصدار scikit-learn   : {sklearn.__version__}")
    log(f"نظام التشغيل         : {platform.system()}")
    log("")

    df = df.drop(columns=[DUP_COL])

    # توحيد التسمية قبل الترشيح، تماماً كما في make_cicids_sample.py
    df[LABEL_COL] = df[LABEL_COL].map(clean_label)
    found = sorted(l for l in df[LABEL_COL].unique() if l in CLASSES)
    log(f"الفئتان المطلوبتان بعد التوحيد: {found}")
    log(f"تطابقهما مع الاسم المطلوب      : {found == sorted(CLASSES)}")
    log("")

    # الترشيح على الصفوف قبل حذف المكرّر (تُستخدم في الجزء 5)
    raw = df.loc[df[LABEL_COL].isin(CLASSES)].reset_index(drop=True)
    log(f"صفوف الملف بعد إزالة «{DUP_COL}»: {len(df):,}   أعمدة: {df.shape[1]}")
    log(f"صفوف الفئتين قبل حذف المكرّر    : {len(raw):,}")
    log("")

    # حذف الصفوف المكرّرة تمامًا كما في train_cicids.py (للأجزاء 2 و3 و4)
    dedup = raw.drop_duplicates().reset_index(drop=True)
    log(f"صفوف مكرّرة تمامًا داخل الفئتين  : {len(raw) - len(dedup):,}")
    log(f"صفوف الفئتين بعد حذف المكرّر     : {len(dedup):,}")
    log("")

    if not os.path.isfile(SAMPLE_PATH):
        raise SystemExit(f"الملف غير موجود: {rel(SAMPLE_PATH)}")

    return raw, dedup, n_file


# ---------------------------------------------------------------------------
# 1) حجم كل فئة
# ---------------------------------------------------------------------------
def part1_counts(raw, dedup):
    log(rule())
    log("1) حجم كل فئة في العيّنة")
    log(rule())
    log("1.1) قبل حذف الصفوف المكرّرة")
    log("")
    rows = []
    for label in CLASSES:
        n = int((raw[LABEL_COL] == label).sum())
        rows.append([label, f"{n:,}", f"{n / len(raw):.2%}"])
    rows.append(["المجموع", f"{len(raw):,}", "100.00%"])
    for line in table(["الفئة", "عدد الصفوف", "النسبة"], rows, [30, 14, 12], ["<", ">", ">"]):
        log(line)
    log("")

    log("1.2) بعد حذف الصفوف المكرّرة (المستخدم في الأجزاء 2 و3 و4)")
    log("")
    rows = []
    for label in CLASSES:
        n = int((dedup[LABEL_COL] == label).sum())
        rows.append([label, f"{n:,}", f"{n / len(dedup):.2%}"])
    rows.append(["المجموع", f"{len(dedup):,}", "100.00%"])
    for line in table(["الفئة", "عدد الصفوف", "النسبة"], rows, [30, 14, 12], ["<", ">", ">"]):
        log(line)
    log("")
    log(f"عدد أعمدة الميزات المستخدمة: {dedup.shape[1] - 1}")
    num = dedup.drop(columns=[LABEL_COL])
    log(f"أعمدة غير رقمية داخل X     : "
        f"{[c for c in num.columns if not pd.api.types.is_numeric_dtype(num[c])] or 'لا يوجد'}")
    log(f"قيم NaN داخل X             : {int(num.isna().sum().sum())}")
    log(f"قيم inf داخل X             : "
        f"{int(np.isinf(num.select_dtypes(include=[np.number])).sum().sum())}")
    log("")


# ---------------------------------------------------------------------------
# 2) التصنيف الثنائي بخمس طيات
# ---------------------------------------------------------------------------
def part2_binary_cv(dedup):
    log(rule())
    log(f"2) تصنيف ثنائي — HistGradientBoostingClassifier، StratifiedKFold({N_SPLITS})")
    log(rule())

    X = dedup.drop(columns=[LABEL_COL])
    y = dedup[LABEL_COL].reset_index(drop=True)
    labels = list(CLASSES)

    log(f"X: {X.shape}   |   y: {y.shape}   |   عدد الفئات: {len(labels)}")
    log("")

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_pred = np.empty(len(y), dtype=object)
    fold_rows = []

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
        p, r, f, _s = precision_recall_fscore_support(
            y_te, pred, labels=labels, zero_division=0
        )
        fold_rows.append([
            str(fold),
            f"{len(train_idx):,}",
            f"{len(test_idx):,}",
            f"{accuracy_score(y_te, pred):.4f}",
            f"{p[0]:.4f}", f"{r[0]:.4f}", f"{f[0]:.4f}",
            f"{p[1]:.4f}", f"{r[1]:.4f}", f"{f[1]:.4f}",
            f"{elapsed:.1f}",
        ])
        log(f"الطيّة {fold}: تدريب={len(train_idx):,}  اختبار={len(test_idx):,}"
            f"  accuracy={accuracy_score(y_te, pred):.4f}  الزمن={elapsed:.1f}s")
        del model, pred

    log("")
    log("2.1) مقاييس كل طيّة على بيانات اختبارها")
    log("")
    header = (["الطيّة", "تدريب", "اختبار", "accuracy",
               "P:1", "R:1", "F1:1", "P:2", "R:2", "F1:2", "زمن/ث"])
    widths = [6, 9, 9, 10, 9, 9, 9, 9, 9, 9, 8]
    aligns = ["<", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">"]
    for line in table(header, fold_rows, widths, aligns):
        log(line)
    log("")
    log(f"مفتاح التسميات:  1 = {CLASS_BF}   |   2 = {CLASS_XSS}")
    log("")

    log("2.2) المقاييس المجمّعة من كل التنبؤات خارج الطيّة")
    log("")
    p, r, f, sup = precision_recall_fscore_support(
        y, oof_pred, labels=labels, zero_division=0
    )
    acc = accuracy_score(y, oof_pred)
    macro = f1_score(y, oof_pred, average="macro", zero_division=0)
    weighted = f1_score(y, oof_pred, average="weighted", zero_division=0)

    log(f"دقة_accuracy : {acc:.4f}")
    log(f"macro-F1     : {macro:.4f}")
    log(f"weighted-F1  : {weighted:.4f}")
    log(f"عدد الصفوف المغطّاة خارج الطيّة: {len(oof_pred):,}  (من أصل {len(y):,})")
    log("")

    rows = []
    for i, label in enumerate(labels):
        rows.append([
            label, f"{p[i]:.4f}", f"{r[i]:.4f}", f"{f[i]:.4f}", f"{sup[i]:,}",
        ])
    log(f"  {'الفئة':<30}{'precision':>12}{'recall':>12}{'F1':>12}{'support':>12}")
    log("  " + "-" * 78)
    for row in rows:
        log(f"  {row[0]:<30}{row[1]:>12}{row[2]:>12}{row[3]:>12}{row[4]:>12}")
    log("  " + "-" * 78)
    log("")

    log("2.3) مصفوفة الالتباس المجمّعة خارج الطيّة (محور الصف = الحقيقي، محور العمود = المتوقَّع)")
    log("")
    cm = confusion_matrix(y, oof_pred, labels=labels)
    for line in confusion_2x2(cm):
        log(line)
    log("")
    log(f"مجموع المصفوفة : {cm.sum():,}")
    log(f"القطر         : {np.trace(cm):,}")
    log("")


# ---------------------------------------------------------------------------
# 3) أقرب الجيران بعد StandardScaler داخل كل طيّة
# ---------------------------------------------------------------------------
def part3_nearest_neighbors(dedup):
    log(rule())
    log("3) مسافات أقرب جار — StandardScaler داخل كل طيّة")
    log(rule())

    X = dedup.drop(columns=[LABEL_COL])
    y = dedup[LABEL_COL].reset_index(drop=True)
    labels = list(CLASSES)

    # أحجام الفئتين بعد حذف المكرّر — أساس حساب نسبة الصدفة المتوقعة
    sizes = {label: int((y == label).sum()) for label in labels}
    total = int(len(y))

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []
    mean_rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y.iloc[train_idx].to_numpy()
        y_te = y.iloc[test_idx].to_numpy()

        scaler = StandardScaler()
        Z_tr = scaler.fit_transform(X_tr)      # الضبط على التدريب فقط
        Z_te = scaler.transform(X_te)          # تطبيق على الاختبار بلا إعادة ضبط

        nn = NearestNeighbors(n_neighbors=len(Z_tr))
        nn.fit(Z_tr)
        dist, idx = nn.kneighbors(Z_te)

        same = y_tr[idx] == y_te[:, None]                    # (n_test, n_train)
        d_same = np.where(same, dist, np.inf).min(axis=1)
        d_other = np.where(~same, dist, np.inf).min(axis=1)
        nearest_is_other = ~same[np.arange(len(y_te)), dist.argmin(axis=1)]

        log(f"الطيّة {fold}: تدريب={len(train_idx):,}  اختبار={len(test_idx):,}"
            f"  أبعاد بعد التطبيع={Z_tr.shape[1]}")

        for label in labels:
            m = y_te == label
            n_test = int(m.sum())
            n_other = int(nearest_is_other[m].sum())
            rows.append([
                str(fold), label, f"{n_test:,}", f"{n_other:,}",
                f"{n_other / n_test:.2%}",
            ])
            mean_rows.append([
                str(fold), label, n_test,
                f"{d_same[m].mean():.4f}", f"{d_other[m].mean():.4f}",
            ])
        del Z_tr, Z_te, nn, dist, idx

    log("")
    log("3.1) لكل طيّة وكل فئة (عبر المسافات بعد التطبيع)")
    log("")
    header = ["الطيّة", "الفئة", "اختبار", "أقرب جار من الأخرى", "النسبة"]
    widths = [9, 26, 9, 22, 9]
    aligns = ["<", "<", ">", ">", ">"]
    for line in table(header, rows, widths, aligns):
        log(line)
    log("")
    log(f"«أقرب جار من الأخرى» = عدد عيّنات الاختبار التي كان فيها d(الفئة الأخرى) < d(نفس الفئة).")
    log("")

    log("3.1ب) متوسط المسافات لكل طيّة وكل فئة — سطران نصّيان لكل فئة")
    log("")
    for fold, label, n_test, d_same_mean, d_other_mean in mean_rows:
        log(f"  الطيّة {fold} — الفئة «{label}» (عدد صفوف الاختبار = {n_test:,})")
        log(f"      متوسط المسافة إلى أقرب عينة من نفس الفئة     : {d_same_mean}")
        log(f"      متوسط المسافة إلى أقرب عينة من الفئة الأخرى  : {d_other_mean}")
        log("")

    log("3.2) المجموع عبر الطيات الخمس")
    log("")
    rows2 = []
    agg = []
    grand_t = grand_o = 0
    for label in labels:
        sel = [r for r in rows if r[1] == label]
        weights = [int(r[2].replace(",", "")) for r in sel]
        n_test = sum(weights)
        n_other = sum(int(r[3].replace(",", "")) for r in sel)
        rows2.append([label, f"{n_test:,}", f"{n_other:,}", f"{n_other / n_test:.2%}"])

        msel = [r for r in mean_rows if r[1] == label]
        mw = [r[2] for r in msel]
        d_same_w = float(np.average([float(r[3]) for r in msel], weights=mw))
        d_other_w = float(np.average([float(r[4]) for r in msel], weights=mw))
        agg.append([label, n_test, n_other, d_same_w, d_other_w])

        grand_t += n_test
        grand_o += n_other

    rows2.append(["المجموع", f"{grand_t:,}", f"{grand_o:,}",
                  f"{grand_o / grand_t:.2%}"])
    for line in table(["الفئة", "اختبار", "أقرب جار من الأخرى", "النسبة"],
                      rows2, [30, 10, 22, 9], ["<", ">", ">", ">"]):
        log(line)
    log("")

    log("3.2ب) نسبة الصدفة المتوقعة — حجم الفئة الأخرى ÷ المجموع، لكل فئة")
    log("")
    for label in labels:
        other = next(l for l in labels if l != label)
        log(f"  الفئة «{label}»: "
            f"حجم الفئة الأخرى «{other}» = {sizes[other]:,} ÷ {total:,} "
            f"= {sizes[other] / total:.2%}")
    log("")
    log(f"  المقاسات المرصودة مقابل الصدفة المتوقعة: "
        f"«{labels[0]}» {agg[0][2]:,} من {agg[0][1]:,} = {agg[0][2] / agg[0][1]:.2%}"
        f"   |   «{labels[1]}» {agg[1][2]:,} من {agg[1][1]:,} = {agg[1][2] / agg[1][1]:.2%}")
    log("")

    log("3.2ج) متوسط المسافات عبر الطيات الخمس — سطران نصّيان لكل فئة")
    log("")
    for label, n_test, n_other, d_same_w, d_other_w in agg:
        log(f"  الفئة «{label}» (عدد صفوف الاختبار = {n_test:,})")
        log(f"      متوسط المسافة إلى أقرب عينة من نفس الفئة     : {d_same_w:.4f}")
        log(f"      متوسط المسافة إلى أقرب عينة من الفئة الأخرى  : {d_other_w:.4f}")
        log("")

    log(f"مجموع صفوف اختبار الطيّات: {grand_t:,}  (من أصل {total:,})")
    log(f"عدد الصفات المُطبَّعة: {X.shape[1]}")
    log("")


# ---------------------------------------------------------------------------
# 4) Cohen's d
# ---------------------------------------------------------------------------
def part4_cohen_d(dedup):
    log(rule())
    log("4) أكثر ميزات فصلاً — |Cohen's d| على البيانات كلها")
    log(rule())

    feature_cols = [c for c in dedup.columns if c != LABEL_COL]
    X = dedup[feature_cols].to_numpy(dtype=float)
    y = dedup[LABEL_COL].to_numpy()

    m_a = y == CLASS_BF
    m_b = y == CLASS_XSS
    X_a, X_b = X[m_a], X[m_b]
    n_a, n_b = len(X_a), len(X_b)

    mean_a, mean_b = X_a.mean(axis=0), X_b.mean(axis=0)
    var_a, var_b = X_a.var(axis=0, ddof=1), X_b.var(axis=0, ddof=1)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    sd_pooled = np.sqrt(pooled)

    with np.errstate(divide="ignore", invalid="ignore"):
        d = (mean_a - mean_b) / sd_pooled
    abs_d = np.abs(d)

    n_zero_sd = int((sd_pooled == 0).sum())
    n_nan = int(np.isnan(abs_d).sum())
    n_inf = int(np.isinf(abs_d).sum())

    log(f"الفئة 1 = {CLASS_BF}   عدد صفوفها = {n_a:,}")
    log(f"الفئة 2 = {CLASS_XSS}   عدد صفوفها = {n_b:,}")
    log(f"عدد الميزات المفحوصة = {len(feature_cols)}")
    log(f"عدد الميزات ذات الانحراف المشترك = 0 : {n_zero_sd}")
    log(f"عدد |d| غير رقمية (NaN)              : {n_nan}")
    log(f"عدد |d| لا نهائية (inf)             : {n_inf}")
    log("")

    ranking = np.argsort(-np.nan_to_num(abs_d, nan=-1.0, posinf=np.inf, neginf=np.inf),
                         kind="stable")
    rows = []
    for rank, j in enumerate(ranking[:TOP_K], start=1):
        rows.append([
            str(rank),
            feature_cols[j],
            f"{abs_d[j]:.4f}",
            f"{d[j]:+.4f}",
            f"{mean_a[j]:.4f}",
            f"{mean_b[j]:.4f}",
            f"{sd_pooled[j]:.4f}",
        ])
    log(f"4.1) أعلى {TOP_K} ميزات")
    log("")
    log(f"مفتاح الأعمدة:  mean class 1 = متوسط «{CLASS_BF}»   |   "
        f"mean class 2 = متوسط «{CLASS_XSS}»")
    log("«pooled sd» = الانحراف المعياري المشترك بين الفئتين، و«|Cohen's d|» = القيمة المطلقة.")
    log("")
    header = ["rank", "feature", "|Cohen's d|", "Cohen's d",
              "mean class 1", "mean class 2", "pooled sd"]
    widths = [7, 34, 13, 13, 15, 15, 16]
    aligns = [">", "<", ">", ">", ">", ">", ">"]
    for line in table(header, rows, widths, aligns):
        log(line)
    log("")
    log(f"الترتيب تنازلياً بـ |Cohen's d| من 1 إلى {TOP_K}.")
    log(f"عدد الميزات المرتبطة بـ |d| > 0 : {int((abs_d > 0).sum())} من {len(feature_cols)}")
    log(f"الحد الأدنى لـ |d| : {np.nanmin(abs_d):.6f}")
    log(f"الحد الأعلى لـ |d| : {np.nanmax(abs_d):.6f}")
    log(f"متوسط |d| عبر كل الميزات : {np.nanmean(abs_d):.6f}")
    log("")

    tie_value = abs_d[ranking[TOP_K - 1]]
    n_at_tie = int((abs_d == tie_value).sum())
    log(f"قيمة |d| عند الحدّ الفاصل للترتيب {TOP_K} : {tie_value:.6f}")
    log(f"عدد الميزات التي لها نفس هذه القيمة بالضبط    : {n_at_tie}")
    log("")


# ---------------------------------------------------------------------------
# 5) الصفوف المتطابقة تماماً بين الفئتين
# ---------------------------------------------------------------------------
def part5_exact_matches(raw):
    log(rule())
    log("5) الصفوف المتطابقة تماماً في الميزات بين الفئتين")
    log(rule())
    log("يُحسب على الصفوف قبل حذف المكرّر.")
    log("")

    feature_cols = [c for c in raw.columns if c != LABEL_COL]
    y = raw[LABEL_COL].to_numpy()
    log(f"عدد الصفوف المفحوصة : {len(raw):,}")
    log(f"عدد أعمدة الميزات   : {len(feature_cols)}")
    for label in CLASSES:
        log(f"  صفوف «{label}» : {int((y == label).sum()):,}")
    log("")

    # تكرار داخل كل فئة على حدة (قبل الحذف)
    dup_all = raw.duplicated().sum()
    per_class_dup = []
    for label in CLASSES:
        sub = raw.loc[raw[LABEL_COL] == label]
        per_class_dup.append([label, f"{len(sub):,}", f"{int(sub.duplicated().sum()):,}"])
    log("5.1) الصفوف المكرّرة تماماً داخل كل فئة على حدة")
    log("")
    for line in table(["الفئة", "عدد الصفوف", "مكرّرة تماماً"],
                      per_class_dup, [30, 14, 16], ["<", ">", ">"]):
        log(line)
    log("")
    log(f"مجموع الصفوف المكرّرة داخل الفئات: {int(dup_all):,}")
    log("")

    # ترشيح مرشّحين بالتجزئة ثم تحقّق فعلي على القيم
    hashes = pd.util.hash_pandas_object(raw[feature_cols], index=False)
    tmp = pd.DataFrame({"h": hashes.to_numpy(), "y": y})
    per_hash = tmp.groupby("h").agg(n=("y", "size"), k=("y", "nunique"))
    candidates = per_hash.index[per_hash["k"] > 1]

    log(f"عدد تجزئات الميزات المستخدمة     : {len(per_hash):,}")
    log(f"عدد التجزئات التي فيها أكثر من تسمية : {len(candidates):,}")
    log(f"عدد الصفوف ضمن هذه التجزئات      : "
        f"{int(per_hash.loc[candidates, 'n'].sum()) if len(candidates) else 0:,}")
    log("")

    if len(candidates) == 0:
        log("5.2) عدد الصفوف المتطابقة تماماً عبر الفئتين : 0")
        log("عدد المجموعات المتطابقة عبر الفئتين        : 0")
        log("")
        return 0, 0

    sub = raw.loc[hashes.isin(set(candidates))]
    # الدمج في نص واحد: تفادياً لاختلاف سلوك agg بين إصدارات pandas
    arr = sub[feature_cols].astype(str).to_numpy(dtype=object)
    keys = np.array([re.sub(r"\s+", " ", "|".join(row)) for row in arr], dtype=object)
    grouped = pd.DataFrame({"key": keys, "y": sub[LABEL_COL].to_numpy()})
    counts = grouped.groupby(["key", "y"]).size().reset_index(name="count")
    labels_per_key = counts.groupby("key")["y"].nunique()
    conflicting = labels_per_key[labels_per_key > 1].index

    if len(conflicting) == 0:
        log("5.2) عدد الصفوف المتطابقة تماماً عبر الفئتين : 0")
        log("عدد المجموعات المتطابقة عبر الفئتين        : 0")
        log("")
        return 0, 0

    rows_by_key = {}
    for key, group in grouped[grouped["key"].isin(set(conflicting))].groupby("key"):
        rows_by_key[key] = dict(zip(group["y"], group["count"]))

    cross_rows = 0
    pair_totals = {}
    for mapping in rows_by_key.values():
        labs = sorted(mapping)
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                pair_totals[(labs[i], labs[j])] = (
                    pair_totals.get((labs[i], labs[j]), 0) + min(mapping[labs[i]], mapping[labs[j]])
                )
        cross_rows += sum(mapping.values())

    log(f"5.2) عدد المجموعات المتطابقة تماماً عبر الفئتين : {len(rows_by_key):,}")
    log(f"5.3) عدد الصفوف داخل هذه المجموعات              : {cross_rows:,}")
    log(f"    النسبة من صفوف الفئتين المفحوصة            : {cross_rows / len(raw):.4%}")
    log("")
    log("أزواج التسميات داخل المجموعات المتطابقة (العدد لكل زوج = min(عدد كل تسمية داخل المجموعة))")
    log("")
    rows = [[a, b, f"{n:,}"] for (a, b), n in sorted(pair_totals.items(), key=lambda kv: -kv[1])]
    for line in table(["التسمية 1", "التسمية 2", "عدد الصفوف"], rows, [30, 30, 14], ["<", "<", ">"]):
        log(line)
    log("")
    log(f"مجموع أزواج التسميات : {sum(pair_totals.values()):,}")
    log("")
    return len(rows_by_key), cross_rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log(rule())
    log("تقرير تحليل Web Attack - Brute Force مقابل Web Attack - XSS — عيّنة CIC-IDS2017")
    log(rule())

    raw, dedup, n_file = load_and_filter()

    part1_counts(raw, dedup)
    part2_binary_cv(dedup)
    part3_nearest_neighbors(dedup)
    part4_cohen_d(dedup)
    part5_exact_matches(raw)

    log(rule())
    log("الحقائق العامة")
    log(rule())
    log(f"المصدر                    : {rel(SAMPLE_PATH)}")
    log(f"صفوف الملف               : {n_file:,}")
    log(f"صفوف الفئتين قبل المكرّر  : {len(raw):,}")
    log(f"صفوف الفئتين بعد المكرّر  : {len(dedup):,}")
    log(f"البذرة                    : {RANDOM_STATE}")
    log(f"الطيّات                   : {N_SPLITS} (shuffle=True)")
    log(f"عمود مكرر مُزال            : {DUP_COL}")
    log(f"عدد الميزات المستخدمة      : {dedup.shape[1] - 1}")
    log(f"المخرج                    : {rel(REPORT_PATH)}")
    log("لم يُحفظ أي نموذج. لم يُستخدم git.")
    log("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nتم الحفظ: {rel(REPORT_PATH)} ({os.path.getsize(REPORT_PATH):,} بايت)")
    return 0


if __name__ == "__main__":
    sys.exit(main())