"""
بناء عيّنة CIC-IDS2017 من الملفات المحلية في data/cicids2017.

- قراءة على دفعات (chunks) حتى لا تمتلئ الذاكرة.
- تنظيف أسماء الأعمدة (المسافات الزائدة) وتوحيد تسميات "Web Attack".
- استبدال +inf و -inf بـ NaN ثم حذف الصفوف التي تحتوي NaN، مع طباعة عدد المحذوف.
- أخذ كل صفوف الفئات التي عددها أقل من CAP، وعيّنة عشوائية بحد أقصى CAP لبقية الفئات.
- لا تدريب أي نموذج، ولا كتابة خارج مجلد data/.
"""

import os
import re
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "cicids2017")
CSV_DIR = os.path.join(DATA_DIR, "MachineLearningCVE")
OUT_PATH = os.path.join(DATA_DIR, "sample.csv")
REPORT_PATH = os.path.join(DATA_DIR, "sample_report.txt")

CAP = 5_000          # الحد الأقصى لصفوف أي فئة
CHUNK = 100_000      # حجم الدفعة عند القراءة
RANDOM_STATE = 42
FLUSH_AT = 2 * CAP   # عند بلوغ ضعف الحد، نُنقّي الخزّان عشوائيًا (=2×CAP ذاكرة كحد أقصى للفئة)

LABEL_COL_CLEAN = "Label"


# ---------------------------------------------------------------------------
# تنظيف الأسماء والتسميات
# ---------------------------------------------------------------------------
def clean_columns(df):
    """يحذف المسافات الزائدة من أسماء الأعمدة ويوحّد شكلها."""
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def clean_label(value):
    """يوحّد تسمية Web Attack: أي محرف غير ASCII يُستبدل بشرطة عادية '-'."""
    s = re.sub(r"\s+", " ", str(value)).strip()
    if s.startswith("Web Attack"):
        s = re.sub(r"[^\x00-\x7F]+", "-", s)   # أي محرف غير ASCII -> شرطة
        s = re.sub(r"\s*-\s*", " - ", s)       # توحيد الفراغات حول الشرطة
        s = re.sub(r"\s+", " ", s).strip()
    return s


def prepare(chunk):
    """يطبّق قاعدة ±inf ثم يحذف الصفوف التي تحتوي NaN."""
    n_in = len(chunk)
    numeric = chunk.select_dtypes(include=[np.number])
    has_inf = np.isinf(numeric).any(axis=1) if not numeric.empty else pd.Series(False, index=chunk.index)
    had_nan = chunk.isna().any(axis=1)

    cleaned = chunk.replace([np.inf, -np.inf], np.nan)
    out = cleaned.loc[~cleaned.isna().any(axis=1)]

    return out, n_in, n_in - len(out), int((has_inf & ~had_nan).sum()), int(had_nan.sum())


def iter_chunks(csv_files):
    """يولّد دفعات نظيفة بالترتيب الأبجدي للملفات (ترتيب ثابت = نتيجة ثابتة)."""
    for path in csv_files:
        for chunk in pd.read_csv(
            path,
            chunksize=CHUNK,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
        ):
            chunk = clean_columns(chunk)
            if LABEL_COL_CLEAN not in chunk.columns:
                raise SystemExit(f"لا يوجد عمود '{LABEL_COL_CLEAN}' في {os.path.basename(path)}")
            chunk[LABEL_COL_CLEAN] = chunk[LABEL_COL_CLEAN].map(clean_label)
            yield os.path.basename(path), chunk


def csv_file_list():
    if not os.path.isdir(CSV_DIR):
        raise SystemExit(f"المجلد غير موجود: {CSV_DIR}")
    files = [
        os.path.join(CSV_DIR, n)
        for n in sorted(os.listdir(CSV_DIR))
        if n.lower().endswith(".csv")
    ]
    if not files:
        raise SystemExit(f"لا توجد ملفات CSV في {CSV_DIR}")
    return files


# ---------------------------------------------------------------------------
# المرحلة 1: الجرد
# ---------------------------------------------------------------------------
def survey(csv_files, log):
    log("=" * 100)
    log("المرحلة 1) الجرد على دفعات — قراءة فقط")
    log("=" * 100)
    log(f"عدد ملفات CSV: {len(csv_files)}   |   حجم الدفعة: {CHUNK:,} صف")

    counts = {}
    total_rows = dropped = inf_only = pre_nan = 0

    for path in csv_files:
        name = os.path.basename(path)
        f_rows = f_drop = f_inf = f_nan = 0
        for _name, chunk in iter_chunks([path]):
            clean, n_in, n_drop, n_inf, n_nan = prepare(chunk)
            f_rows += n_in
            f_drop += n_drop
            f_inf += n_inf
            f_nan += n_nan
            for label, c in clean[LABEL_COL_CLEAN].value_counts().items():
                counts[label] = counts.get(label, 0) + int(c)
            del clean, chunk
        total_rows += f_rows
        dropped += f_drop
        inf_only += f_inf
        pre_nan += f_nan
        log(f"  {name:<62} صفوف={f_rows:>9,}  محذوف(NaN)={f_drop:>8,}")

    log("")
    log(f"إجمالي الصفوف المقروءة        : {total_rows:,}")
    log(f"صفوف حُذفت (تحتوي NaN)       : {dropped:,}   ({dropped / total_rows:.2%} من المقروء)")
    log(f"  منها بسبب +inf/-inf وحده    : {inf_only:,}")
    log(f"صفوف كانت فيها NaN أصلًا     : {pre_nan:,}")
    log("")
    return counts, total_rows, dropped, inf_only, pre_nan


# ---------------------------------------------------------------------------
# المرحلة 2: العيّنة (تنقيح خزّان على مستوى الدفعة، بلا حلقة لكل صف)
# ---------------------------------------------------------------------------
def _subsample(frames, keep, rng):
    """يختار keep صفًا عشوائيًا من إطار مُكدَّس (يحافظ على أنواع البيانات)."""
    stacked = pd.concat(frames, ignore_index=True)
    idx = np.sort(rng.choice(len(stacked), size=keep, replace=False))
    return stacked.iloc[idx].reset_index(drop=True)


def build_sample(csv_files, counts, cap, seed, log):
    log("")
    log("=" * 100)
    log(f"المرحلة 2) أخذ العيّنة — حد أقصى {cap:,} صف لكل فئة")
    log("=" * 100)

    rng = np.random.default_rng(seed)
    small = {lab for lab, n in counts.items() if n < cap}

    buffers = {lab: [] for lab in counts if lab not in small}   # قوائم DataFrames
    buffered = {lab: 0 for lab in buffers}
    seen = {lab: 0 for lab in counts}
    small_rows = {lab: [] for lab in small}
    columns = None

    for _name, chunk in iter_chunks(csv_files):
        clean, _n, _d, _i, _p = prepare(chunk)
        if columns is None:
            columns = list(clean.columns)
        labels = clean[LABEL_COL_CLEAN].to_numpy()

        for label in np.unique(labels):
            rows = clean.loc[labels == label]
            seen[label] += len(rows)
            if label in small_rows:
                small_rows[label].append(rows)
                continue
            buffers[label].append(rows)
            buffered[label] += len(rows)
            if buffered[label] >= FLUSH_AT:
                kept = _subsample(buffers[label], cap, rng)
                buffers[label] = [kept]
                buffered[label] = len(kept)

        del clean, chunk, labels

    parts = []
    for label in sorted(small_rows):
        if small_rows[label]:
            parts.append(pd.concat(small_rows[label], ignore_index=True))
    for label in sorted(buffers):
        if not buffers[label]:
            continue
        if seen[label] > cap:
            parts.append(_subsample(buffers[label], cap, rng))
        else:
            parts.append(pd.concat(buffers[label], ignore_index=True))

    sample = pd.concat(parts, ignore_index=True)[columns]
    sample[LABEL_COL_CLEAN] = pd.Categorical(
        sample[LABEL_COL_CLEAN], categories=sorted(counts), ordered=True
    )
    sample = sample.sort_values(LABEL_COL_CLEAN, kind="stable").reset_index(drop=True)
    sample[LABEL_COL_CLEAN] = sample[LABEL_COL_CLEAN].astype(str)

    return sample, small


# ---------------------------------------------------------------------------
# التقرير
# ---------------------------------------------------------------------------
def report_table(before, after, small, cap):
    rows = []
    rows.append("")
    rows.append("=" * 100)
    rows.append("مقارنة الفئات: قبل العيّنة (بعد التنظيف وحذف NaN) مقابل بعد العيّنة")
    rows.append("=" * 100)
    rows.append(f"{'الفئة':<40}{'قبل':>12}{'بعد':>10}{'المأخوذ':>11}   {'القاعدة المطبَّقة'}")
    rows.append("-" * 100)
    for label in sorted(before, key=lambda k: (-before[k], k)):
        a = after.get(label, 0)
        rule = f"كل الصفوف (أقل من {cap:,})" if label in small else f"عشوائي بحد {cap:,}"
        rows.append(f"{label:<40}{before[label]:>12,}{a:>10,}{a:>10,}   {rule}")
    rows.append("-" * 100)
    tb, ta = sum(before.values()), sum(after.values())
    rows.append(f"{'المجموع':<40}{tb:>12,}{ta:>10,}{ta:>10,}")
    rows.append("=" * 100)
    rows.append(f"نسبة الاحتفاظ الإجمالية: {ta / tb:.2%}")
    return "\n".join(rows)


def main():
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    log("CIC-IDS2017 — بناء عيّنة")
    log(f"مجلد المصدر: {CSV_DIR}")
    log(f"ملف الوجهة : {OUT_PATH}")

    csv_files = csv_file_list()
    before, total_read, dropped, inf_only, pre_nan = survey(csv_files, log)

    log("توزيع الفئات بعد التنظيف وقبل العيّنة:")
    for label in sorted(before, key=lambda k: -before[k]):
        log(f"  {label:<40}{before[label]:>12,}{before[label] / total_read:>10.2%}")

    sample, small = build_sample(csv_files, before, CAP, RANDOM_STATE, log)
    after = sample[LABEL_COL_CLEAN].value_counts().to_dict()

    log("")
    log(f"فئات أُخذت بالكامل (عددها قبل العيّنة أقل من {CAP:,}): {len(small)}")
    for label in sorted(small, key=lambda k: -before[k]):
        log(f"  - {label:<38}{before[label]:>10,}")
    log(f"فئات أُخذ منها {CAP:,} صف كحد أقصى: {len(before) - len(small)}")

    log(report_table(before, after, small, CAP))

    num = sample.select_dtypes(include=[np.number])
    log("")
    log("تفاصيل العيّنة المحفوظة:")
    log(f"  عدد الصفوف : {len(sample):,}")
    log(f"  عدد الأعمدة: {sample.shape[1]}")
    log(f"  عدد الفئات : {sample[LABEL_COL_CLEAN].nunique()}")
    log(f"  NaN متبقية : {int(sample.isna().sum().sum())}")
    log(f"  inf متبقية : {int(np.isinf(num).sum().sum())}")
    log(f"  أنواع الأعمدة: {sample.dtypes.value_counts().to_dict()}")
    log(f"  أول 5 أعمدة: {list(sample.columns[:5])}")
    log(f"  آخر 5 أعمدة: {list(sample.columns[-5:])}")

    sample.to_csv(OUT_PATH, index=False, encoding="utf-8")
    log("")
    log(f"حُفظت العيّنة : {OUT_PATH}")
    log(f"الحجم         : {os.path.getsize(OUT_PATH):,} بايت "
        f"({os.path.getsize(OUT_PATH) / 1024 / 1024:.2f} MB)")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"حُفظ التقرير  : {REPORT_PATH} ({os.path.getsize(REPORT_PATH):,} بايت)")
    log("")
    log("لم يُدرَّب أي نموذج. لم يُستخدم git.")


if __name__ == "__main__":
    sys.exit(main())