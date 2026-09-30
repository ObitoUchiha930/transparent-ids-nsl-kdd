"""
تنزيل ملف MachineLearningCSV.zip من مستودع dataset على Hugging Face، مع فك الضغط.

سلوك مقصود:
- ينزّل بأمر hf_hub_download مع token=False (بلا تسجيل دخول وبلا أي بيانات شخصية).
- يتحقق أن الحجم بالضبط 235102953 بايت؛ عند الاختلاف يتوقف ويطبع خطأ ولا يحذف شيئًا.
- يتخطى التنزيل إن كان الملف موجودًا بالحجم الصحيح.
- قبل فك الأرشيف: يقبل الأعضاء ذات الامتداد .csv فقط، ويتأكد أن كل عضو يقع داخل
  المجلد الهدف (حماية من path traversal مثل ../../ ). أي عضو مخالف يُسقط الأرشيف كله.
- يتخطى الفك إن كانت ملفات CSV موجودة.
- يطبع قائمة الملفات المفكوكة وأحجامها و SHA256 للأرشيف.
- تنبيه: المصدر نسخة غير رسمية، وفحص الحجم يكشف التلف لا التلاعب.

كل المسارات المطبوعة نسبية لجذر المشروع.
"""

import hashlib
import os
import posixpath
import sys
import zipfile

# ---------------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "cicids2017")
TARGET_DIR = os.path.join(DATA_DIR, "csvs")   # موضع الأرشيف داخل data/cicids2017
# جذر الفك: نحافظ على البنية الداخلية للأرشيف، فـ "MachineLearningCVE/..."
# يستقرّ في data/cicids2017/MachineLearningCVE/... وهو ما يقرأه make_cicids_sample.py
EXTRACT_ROOT = DATA_DIR
ARCHIVE_TOP = "MachineLearningCVE"
EXPECTED_DIR = os.path.join(DATA_DIR, ARCHIVE_TOP)

REPO = "bencorn/CICIDS2017"
REPO_TYPE = "dataset"
FILENAME = "csvs/MachineLearningCSV.zip"
EXPECTED_BYTES = 235_102_953
ALLOWED_SUFFIX = ".csv"

ZIP_NAME = os.path.basename(FILENAME)


def rel(path):
    """مسار نسبي لجذر المشروع (يمنع طباعة اسم المستخدم والمسار المطلق)."""
    return os.path.relpath(path, ROOT).replace("\\", "/")


def rule(char="=", width=100):
    return char * width


# ---------------------------------------------------------------------------
# تنبيه المصدر
# ---------------------------------------------------------------------------
def print_notice():
    print(rule())
    print("تنبيه المصدر")
    print(rule())
    print(f"  المستودع: {REPO}  (repo_type={REPO_TYPE})")
    print("  هذا ليس المصدر الرسمي لـ Canadian Institute for Cybersecurity (CIC).")
    print("  صفحة UNB الرسمية تتطلب تعبئة نموذج ببيانات شخصية قبل التنزيل،")
    print("  لذلك هذا المستودع على Hugging Face نسخة غير رسمية من طرف ثالث.")
    print()
    print("  ما يكشفه فحص الحجم: أن الملف لم يتلف أثناء النقل (بت ناقص/زائد).")
    print("  ما لا يكشفه فحص الحجم: التلاعب المقصود في المحتوى. قيمة SHA256")
    print("  المطبوعة أدناه محسوبة من الملف نفسه، فهي تكشف التلف لا التلاعب.")
    print("  لإثبات المصدر يلزم مرجع خارجي موثوق: توقيع رقمي، أو بصمة (hash)")
    print("  معلنة من الناشر في قناة منفصلة عن الملف نفسه.")
    print("  استخدم هذا الملف للبحث والتعلّم، لا كمدخل حاسم أمنيًا.")
    print()


# ---------------------------------------------------------------------------
# فحص الحجم
# ---------------------------------------------------------------------------
def check_size(path):
    actual = os.path.getsize(path)
    print()
    print("فحص الحجم:")
    print(f"  المتوقع: {EXPECTED_BYTES:,} بايت")
    print(f"  الفعلي : {actual:,} بايت")
    print(f"  الفرق  : {actual - EXPECTED_BYTES:+,} بايت")
    ok = actual == EXPECTED_BYTES
    print(f"  النتيجة: {'مطابق تمامًا' if ok else 'غير مطابق'}")
    return ok


def sha256_of(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# (أ)(ب)(ج) التنزيل
# ---------------------------------------------------------------------------
def download():
    print(rule())
    print("(أ)-(ب)-(ج) التنزيل")
    print(rule())
    print(f"الوجهة: {rel(TARGET_DIR)}/{ZIP_NAME}")

    existing = os.path.join(TARGET_DIR, ZIP_NAME)
    if os.path.isfile(existing):
        print("الملف موجود مسبقًا — أتحقق من الحجم قبل أي تنزيل.")
        if check_size(existing):
            print("  تخطّي التنزيل: الحجم صحيح.")
            return existing, False
        print("  الحجم غير صحيح — سيُعاد التنزيل فوقه.")
        print("  (لن يُحذف أي شيء؛ الكتابة فوق الملف فقط.)")

    from huggingface_hub import hf_hub_download

    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"\nجارٍ التنزيل بـ token=False (بلا تسجيل دخول)...")
    path = hf_hub_download(
        repo_id=REPO,
        filename=FILENAME,
        repo_type=REPO_TYPE,
        local_dir=DATA_DIR,
        token=False,
    )
    print(f"اكتمل التنزيل: {rel(path)}")

    if not check_size(path):
        print()
        print("خطأ: الحجم لا يطابق 235102953 بايت.")
        print("توقّف التنفيذ. لم يُحذف أي ملف، ولم يُفكّ الأرشيف.")
        sys.exit(1)
    return path, True


# ---------------------------------------------------------------------------
# (د) تدقيق محتوى الأرشيف قبل فكّه
# ---------------------------------------------------------------------------
def audit_archive(path):
    print()
    print(rule())
    print("(د) تدقيق محتوى الأرشيف قبل فك الضغط")
    print(rule())

    target_abs = os.path.abspath(EXTRACT_ROOT)
    ok_members, seen, dirs = [], [], set()

    with zipfile.ZipFile(path) as zf:
        bad_syntax = zf.testzip()
        print(f"  سلامة بنية الأرشيف: {'سليمة' if bad_syntax is None else 'تالفة عند ' + bad_syntax}")

        for info in zf.infolist():
            name = info.filename

            if info.is_dir():
                dirs.add(name)
                continue

            ext = os.path.splitext(name)[1].lower()
            if ext != ALLOWED_SUFFIX:
                print(f"  رفض: «{name}» امتداده {ext or '(بلا امتداد)'} وليس {ALLOWED_SUFFIX}")
                return None, "عضو غير CSV"

            normalized = posixpath.normpath(name)
            if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
                print(f"  رفض: «{name}» مساره يخرج من المجلد الهدف")
                return None, "مسار خارج الهدف (path traversal)"

            if os.path.isabs(name) or (len(name) > 1 and name[1] == ":"):
                print(f"  رفض: «{name}» مسار مطلق")
                return None, "مسار مطلق"

            parts = normalized.split("/")
            if parts[0] != ARCHIVE_TOP:
                print(f"  رفض: «{name}» ليس داخل مجلد {ARCHIVE_TOP}/")
                return None, "بنية غير متوقعة (ليس تحت " + ARCHIVE_TOP + "/)"

            dest_abs = os.path.abspath(os.path.join(target_abs, *parts))
            if os.path.commonpath([target_abs, dest_abs]) != target_abs:
                print(f"  رفض: «{name}» حُلّ إلى خارج المجلد الهدف")
                return None, "مسار خارج الهدف بعد التطبيع"

            seen.append((normalized, info.file_size))
            ok_members.append(info)

    if not seen:
        return None, "لا يوجد أي ملف CSV"

    print(f"  أعضاء صالحون: {len(seen)} ملف CSV")
    for norm, size in seen:
        print(f"    {norm:<62}{size:>14,} بايت")
    print(f"  تم رفض الأرشيف: لا — كل الأعضاء داخل {rel(target_abs)}")
    return ok_members, None


# ---------------------------------------------------------------------------
# (هـ) فك الضغط مع تخطّي الموجود
# ---------------------------------------------------------------------------
def extract(members, path):
    print()
    print(rule())
    print("(هـ) فك الضغط")
    print(rule())
    os.makedirs(EXTRACT_ROOT, exist_ok=True)

    pending, skipped, all_dests = [], [], []
    for info in members:
        dest = os.path.join(EXTRACT_ROOT, *info.filename.split("/"))
        all_dests.append(dest)
        if os.path.isfile(dest):
            skipped.append(dest)
        else:
            pending.append(info)

    if not pending:
        print(f"لا شيء لفكّه: كل ملفات CSV موجودة ({len(skipped)} ملف).")
    else:
        print(f"فكّ {len(pending)} ملف، وتخطّي {len(skipped)} موجودة.")
        with zipfile.ZipFile(path) as zf:
            for info in pending:
                zf.extract(info, EXTRACT_ROOT)
        print("اكتمل فك الضغط.")

    print()
    print("(و) الملفات المفكوكة (المسارات كما يقرؤها make_cicids_sample.py):")
    print(f"  {'الملف':<62}{'بايت':>14}{'الحجم':>12}")
    print("  " + "-" * 90)
    total = 0
    for dest in sorted(all_dests):
        size = os.path.getsize(dest)
        total += size
        name = os.path.relpath(dest, ROOT).replace("\\", "/")
        print(f"  {name:<62}{size:>14,}{size / 1024 / 1024:>11.2f}M")
    print("  " + "-" * 90)
    print(f"  {'المجموع':<62}{total:>14,}{total / 1024 / 1024:>11.2f}M")
    return total


# ---------------------------------------------------------------------------
def main():
    print("تنزيل وفكّ عيّنة CIC-IDS2017 — نسخة غير رسمية")
    print(rule())

    print_notice()

    zip_path, downloaded = download()
    members, err = audit_archive(zip_path)
    if members is None:
        print()
        print(f"خطأ: رُفض الأرشيف ({err}).")
        print("لم يُفكّ أي ملف ولم يُحذف أي شيء. تنتهي العملية.")
        sys.exit(1)

    extract(members, zip_path)

    print()
    print(rule())
    print("(ز) ملخص")
    print(rule())
    print(f"  الأرشيف        : {rel(zip_path)}")
    print(f"  الحجم          : {os.path.getsize(zip_path):,} بايت (مطابق لـ {EXPECTED_BYTES:,})")
    print(f"  SHA256         : {sha256_of(zip_path)}")
    print(f"  نُزّل هذا Run   : {'نعم' if downloaded else 'لا — تخطّى لوجوده بالحجم الصحيح'}")
    print(f"  مجلد الفك      : {rel(EXPECTED_DIR)}")
    print()
    print("  تذكير: حجم ثابت + SHA256 من الملف نفسه لا يثبت سلامة المحتوى.")
    print("  يلزم مرجع خارجي موثوق (توقيع أو بصمة معلنة) إن لزم إثبات المصدر.")
    print("  والمصدر نسخة غير رسمية من طرف ثالث، لا من UNB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())