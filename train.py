"""تدريب نظام كشف تسلل شفاف (Random Forest + Decision Tree قابل للقراءة) على NSL-KDD."""
import os
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]


def load(name):
    df = pd.read_csv(os.path.join(DATA, name), names=COLUMNS)
    y = (df["label"] != "normal").astype(int)  # 0 = طبيعي، 1 = هجوم
    X = df.drop(columns=["label", "difficulty"])
    return X, y


def main():
    X_train, y_train = load("KDDTrain+.txt")
    X_test, y_test = load("KDDTest+.txt")

    # ترميز الميزات النصية بنفس الأعمدة في التدريب والاختبار
    both = pd.get_dummies(pd.concat([X_train, X_test]), columns=CATEGORICAL)
    X_train = both.iloc[: len(X_train)]
    X_test = both.iloc[len(X_train):]

    # 1) Random Forest: الأقوى
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    print("=== Random Forest ===")
    print(classification_report(y_test, pred, target_names=["normal", "attack"]))
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred))

    # 2) Decision Tree بسيطة: قواعد يمكن قراءتها بالعين
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = (dt.predict(X_test) == y_test).mean()
    print(f"\n=== Decision Tree (depth=4) accuracy: {dt_acc:.3f} ===")
    rules = export_text(dt, feature_names=list(X_train.columns))
    with open(os.path.join(OUT, "rules.txt"), "w", encoding="utf-8") as f:
        f.write(rules)
    print(rules)

    # 3) أهمية الميزات: لماذا يقرر النموذج هكذا
    imp = (
        pd.Series(rf.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    print("\nأهم 15 ميزة:")
    print(imp.round(4))
    imp[::-1].plot(kind="barh", figsize=(8, 6), title="Top 15 feature importances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=120)

    joblib.dump({"model": rf, "columns": list(X_train.columns)},
                os.path.join(OUT, "ids_model.joblib"))
    print("\nتم حفظ النموذج والقواعد والرسم في مجلد output/")


if __name__ == "__main__":
    main()
