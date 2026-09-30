"""تصنيف هجمات NSL-KDD إلى 5 فئات معروفة (DoS/Probe/R2L/U2R) وتقييم النموذج على كل فئة."""
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

from train import CATEGORICAL, COLUMNS, DATA, OUT

SEED = 42
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# قاموس التصنيف الرسمي لأسماء هجمات NSL-KDD / KDD'99
ATTACK_FAMILIES = {
    "DoS": [
        "apache2", "back", "land", "mailbomb", "neptune", "named", "pod",
        "processtable", "sendmail", "smurf", "teardrop", "udpstorm", "worm",
        "xlock", "snmpgetattack", "httpxunnel",
    ],
    "Probe": [
        "ipsweep", "mscan", "nmap", "nmap1", "portsweep", "saint", "satan",
        "satan1",
    ],
    "R2L": [
        "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "phf",
        "snmpguess", "sqlattack", "spy", "warezclient", "warezmaster", "xsnoop",
    ],
    "U2R": [
        "buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "xterm",
    ],
}
LABEL_TO_CLASS = {"normal": "normal"}
for _family, _names in ATTACK_FAMILIES.items():
    for _name in _names:
        LABEL_TO_CLASS[_name] = _family

REPORT_TXT = os.path.join(OUT, "attack_types_report.txt")
CONFUSION_PNG = os.path.join(OUT, "attack_types_confusion.png")


def encode(X_train, X_test):
    """نفس ترميز train.py: get_dummies على التدريب والاختبار معًا."""
    both = pd.get_dummies(pd.concat([X_train, X_test]), columns=CATEGORICAL)
    return both.iloc[: len(X_train)], both.iloc[len(X_train):]


def load_multiclass(name):
    """قراءة الملف بنفس أعمدة train.py مع الإبقاء على عمود label للتصنيف إلى 5 فئات."""
    df = pd.read_csv(os.path.join(DATA, name), names=COLUMNS)
    raw = df["label"].astype(str).str.strip()

    unknown = sorted(set(raw.unique()) - set(LABEL_TO_CLASS))
    if unknown:
        print("!! أسماء هجمات غير موجودة في القاموس:")
        for u in unknown:
            print(f"   - {u}  ({int((raw == u).sum())} سطر)")
        raise SystemExit(
            "توقّف عمدًا: أضف هذه الأسماء إلى ATTACK_FAMILIES في classify_attacks.py")

    y = raw.map(LABEL_TO_CLASS).map(CLASS_INDEX).astype(int)
    X = df.drop(columns=["label", "difficulty"])
    return X, y, raw


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    with np.errstate(divide="ignore"):
        norm = mcolors.LogNorm(vmin=1, vmax=max(cm.max(), 2))
    im = ax.imshow(cm, cmap="Blues", norm=norm)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("المتوقَّع (predicted)")
    ax.set_ylabel("الحقيقي (actual)")
    ax.set_title("Confusion Matrix — NSL-KDD 5-class (log color scale)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            total = cm[i].sum()
            share = cm[i, j] / total if total else 0.0
            ax.text(j, i, f"{cm[i, j]}\n({share:.1%})", ha="center", va="center",
                    fontsize=9, color="white" if share > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="count (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    X_train, y_train, raw_train = load_multiclass("KDDTrain+.txt")
    X_test, y_test, raw_test = load_multiclass("KDDTest+.txt")
    X_train, X_test = encode(X_train, X_test)

    lines = []
    def out(text=""):
        print(text)
        lines.append(text)

    out("=" * 78)
    out("تقرير تصنيف هجمات NSL-KDD إلى 5 فئات")
    out(f"التاريخ: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out(f"النموذج: HistGradientBoostingClassifier (إعدادات افتراضية, random_state={SEED})")
    out("=" * 78)
    out(f"التدريب: {X_train.shape[0]:,} سطر × {X_train.shape[1]} ميزة")
    out(f"الاختبار: {X_test.shape[0]:,} سطر × {X_test.shape[1]} ميزة")

    # توزيع الفئات
    out("\n--- توزيع الفئات ---")
    out(f"{'الفئة':<10}{'تدريب':>12}{'نسبة':>10}{'اختبار':>12}{'نسبة':>10}{'هجمات فريدة':>16}")
    out("-" * 78)
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    n_atk_train = int((y_train != 0).sum())
    n_atk_test = int((y_test != 0).sum())
    for idx, cls in enumerate(CLASSES):
        tr = int(train_counts.get(idx, 0))
        te = int(test_counts.get(idx, 0))
        n_att = int(raw_train[raw_train.map(LABEL_TO_CLASS) == cls].nunique())
        denom_tr = n_atk_train if idx else train_counts.get(0, 0)
        denom_te = n_atk_test if idx else test_counts.get(0, 0)
        out(f"{cls:<10}{tr:>12,}{tr / denom_tr:>9.1%}{te:>12,}{te / denom_te:>9.1%}"
            f"{n_att:>16}")

    # التدريب
    model = HistGradientBoostingClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = float((pred == y_test).mean())

    # تقرير التصنيف
    report_txt = classification_report(
        y_test, pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
        digits=4, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASSES))))

    out(f"\n--- accuracy الكلي: {acc:.4f} ---")
    out("\n--- تقرير precision / recall / F1 لكل فئة ---")
    out(report_txt)

    out("--- Confusion Matrix (صف = الحقيقي، عمود = المتوقَّع) ---")
    out("classes: " + ", ".join(f"{i}={c}" for i, c in enumerate(CLASSES)))
    out("        " + "".join(f"{c:>10}" for c in CLASSES) + f"{'total':>12}")
    for i, cls in enumerate(CLASSES):
        row = f"{cls:<8}"
        for j in range(len(CLASSES)):
            row += f"{cm[i, j]:>10,}"
        row += f"{cm[i].sum():>12,}"
        out(row)

    # ملخص يركّز علىrecall لكل فئة (حساسية الكشف)
    out("\n--- ملخص detectability لكل فئة ---")
    out(f"{'الفئة':<10}{'دعم الاختبار':>14}{'precision':>12}{'recall':>10}{'f1':>10}{'مفقود FN':>12}")
    out("-" * 78)
    for i, cls in enumerate(CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out(f"{cls:<10}{cm[i].sum():>14,}{prec:>12.4f}{rec:>10.4f}{f1:>10.4f}{fn:>12,}")

    plot_confusion(cm, CONFUSION_PNG)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "=" * 78)
    print(f"تم الحفظ: {REPORT_TXT}")
    print(f"تم الحفظ: {CONFUSION_PNG}")


if __name__ == "__main__":
    main()
