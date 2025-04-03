"""
TODO:
    1. Nhập dữ liệu:
        a. Nhập file bằng pandas.
        b. Loại bỏ các cột thừa bằng pd.drop()
        c. Tách input / output
        d. (Generate html report với ydata_profiling).
        e. Tách x_train, x_test, y_train, y_test bằng train_test_split với random_state = 42

    2. Tiền xử lý:
        a. Xử lý zero / outliner của bộ train
        b. Với dữ liệu số, dùng StandardScaler.
            - fit_transform với x_train.
            - transform với x_test.
        c. Với dữ liệu category, dùng ???

    3. Chọn mô hình:
        a. Dùng LazyPredict để tìm nhanh 5 - 10 mô hình phù hợp nhất.
        b. GridSearchCV để tìm bộ siêu tham số phù hợp nhất với từng mô hình.
        c. Kết luận về bộ mô hình / siêu tham số phù hợp.

    4. Xây dựng mô hình và lưu lại với pickle.

    5. Chạy thử mô hình.
"""
from os import remove

import pandas as pd
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

# from lazypredict.Supervised import LazyRegressor


##1.##


file_dir = "data"           # Thư mục chứa file cần dọc.
file_name = "dataset.csv"   # Tên file cần đọc
os.makedirs(file_dir, exist_ok=True)    # Tạo đường dẫn theo hệ điều hành
data = pd.read_csv(os.path.join(file_dir, file_name))  # đọc nội dung file đưa vào data

                # #Generate report ONCE
                # from ydata_profiling import ProfileReport
                #
                # profile = ProfileReport(data, title="Overview report", explorative=True)
                # profile.to_file("Overview_Report.html")


target = "MedHouseVal"   # Cột chứa kết quả cần dự đoán
rm_cols = []         # Cột chứa các thông tin không cần thiết.
rm_cols.append(target)



x = data.drop(rm_cols, axis='columns')      # Drop các cột không nằm trong INPUT
y = data[target]                            # Lấy cột OUTPUT

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    # Tách bộ train / test theo tỉ lệ 80 / 20. Randon seed = 42

## Nên dùng pipeline để chuẩn hóa
# scaler = StandardScaler()               # Chuẩn hóa dữ liệu dạng số với Standard Scaler
# x_train = scaler.fit_transform(x_train)   # Các mô hình tree sẽ không cần chuẩn hóa
# x_test = scaler.transform(x_test)


            # ## lazypredict ###  Result >> LGBMRegressor
            #
            # lazy_x_train, lazy_x_test, lazy_y_train, lazy_y_test = train_test_split(x_test, y_test, test_size=0.4, random_state=42)
            # lazy_regressor = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
            # models, predictions = lazy_regressor.fit(lazy_x_train, lazy_x_test, lazy_y_train, lazy_y_test)


### Chạy GridSearchCV với LGBMRegressor


# 1. Định nghĩa lưới các siêu tham số để thử
param_grid = {
    'n_estimators': [100, 200, 300],          # Số lượng cây (boosting iterations)
    'max_depth': [-1, 5, 10],              # Độ sâu tối đa của cây (-1: không giới hạn)
    'learning_rate': [0.01, 0.05, 0.1],      # Tốc độ học
}

# 2. Tạo LGBMRegressor
lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1)  # n_jobs=-1: Sử dụng tất cả các cores

# 3. Tạo GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    cv=3,                                        # Số folds trong cross-validation
    scoring='neg_mean_squared_error',            # Độ đo đánh giá (âm của MSE)
    verbose=1,
    n_jobs=-1
)

# 4. Chạy Grid Search trên tập huấn luyện
grid_search.fit(x_train, y_train)

# 5. In ra các tham số tốt nhất và hiệu năng tương ứng
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# 6. Lấy mô hình tốt nhất
best_lgbm_model = grid_search.best_estimator_

        # # 7. Đánh giá mô hình tốt nhất trên tập test
        # print("\nĐánh giá trên tập test với mô hình tốt nhất:")
        # evaluate_model(best_lgbm_model, x_test, y_test, "Best LGBM")
        #
        #













