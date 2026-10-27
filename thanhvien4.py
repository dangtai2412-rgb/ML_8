import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ==========================================
# 1. NHẬN DỮ LIỆU ĐẦU VÀO (X_train, y_train từ Bước 3)
# ==========================================
# Nếu chưa có file dữ liệu từ Thành viên 3, code dùng mock data để chạy test.
# Sau khi Thành viên 3 đẩy code lên GitHub, cậu chỉ cần thay đường dẫn đọc file thật vào đây.
print("Đang tải dữ liệu đầu vào cho Thành viên 4...")
X_train = csr_matrix(
    np.random.rand(1000, 3003)
)  # Thay bằng X_train thực tế nếu có
y_train = np.random.choice([0, 1], size=1000, p=[0.85, 0.15])  # Nhãn Spam/Ham


# ==========================================
# 2. KHỞI TẠO CÁC MÔ HÌNH (BASELINE & 4 MÔ HÌNH)
# ==========================================
models = {
    "Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Linear SVM": LinearSVC(random_state=42, max_iter=2000),
}


# ==========================================
# 3. ĐÁNH GIÁ MÔ HÌNH BẰNG 5-FOLD CROSS-VALIDATION
# ==========================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
evaluation_results = []

print(
    "\n================================================================"
)
print("ĐANG HUẤN LUYỆN VÀ ĐÁNH GIÁ 5-FOLD CROSS-VALIDATION...")
print(
    "================================================================\n"
)

for name, model in models.items():
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring={
            "f1": "f1",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
        },
        return_train_score=False,
    )

    f1_mean = cv_results["test_f1"].mean()
    f1_std = cv_results["test_std"] if "test_std" in cv_results else 0.0
    acc_mean = cv_results["test_accuracy"].mean()
    prec_mean = cv_results["test_precision"].mean()
    rec_mean = cv_results["test_recall"].mean()

    evaluation_results.append({
        "Model": name,
        "F1-Score (Mean)": f1_mean,
        "Accuracy": acc_mean,
        "Precision": prec_mean,
        "Recall": rec_mean,
    })

    print(
        f"✔️ {name:<22} | F1-Score: {f1_mean:.4f} | Accuracy: {acc_mean:.4f}"
    )

# Tổng kết kết quả thành bảng DataFrame
df_results = pd.DataFrame(evaluation_results).sort_values(
    by="F1-Score (Mean)", ascending=False
)

print("\n--- BẢNG TỔNG KẾT KẾT QUẢ SO SÁNH ---")
print(df_results.to_string(index=False))


# ==========================================
# 4. TRỰC QUAN HÓA BẰNG BIỂU ĐỒ CỘT
# ==========================================
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

ax = sns.barplot(
    x="F1-Score (Mean)", y="Model", data=df_results, palette="mako"
)

for p in ax.patches:
    width = p.get_width()
    ax.annotate(
        f"{width:.4f}",
        (width, p.get_y() + p.get_height() / 2.0),
        ha="left",
        va="center",
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

plt.title(
    "So Sánh Chỉ Số F1-Score Giữa Các Mô Hình (5-Fold Cross-Validation)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.xlabel("F1-Score Trung Bình", fontsize=11)
plt.ylabel("Mô Hình", fontsize=11)
plt.xlim(0, 1.15)
plt.tight_layout()
plt.show()


# ==========================================
# 5. KẾT LUẬN & BÀN GIAO CHO THÀNH VIÊN 5
# ==========================================
best_model_name = df_results.iloc[0]["Model"]
best_f1_score = df_results.iloc[0]["F1-Score (Mean)"]

print(
    "\n================================================================"
)
print(f"🎯 MÔ HÌNH TỐT NHẤT: {best_model_name}")
print(f"👉 Đạt F1-Score trung bình cao nhất: {best_f1_score:.4f}")
print("📦 Đã hoàn thành phần Thành viên 4 và sẵn sàng bàn giao cho Thành viên 5!")
print(
    "================================================================\n"
)