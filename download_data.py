"""تنزيل مجموعة بيانات NSL-KDD (مفتوحة المصدر) من GitHub."""
import os
import requests

BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"
FILES = ["KDDTrain+.txt", "KDDTest+.txt"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in FILES:
        path = os.path.join(OUT_DIR, name)
        if os.path.exists(path):
            print(f"موجود مسبقًا: {name}")
            continue
        print(f"تنزيل {name} ...")
        r = requests.get(f"{BASE}/{name}", timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    print("تم.")


if __name__ == "__main__":
    main()
