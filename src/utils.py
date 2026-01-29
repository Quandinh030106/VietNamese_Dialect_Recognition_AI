import os
import pandas as pd

# Mapping nhãn từ chữ sang số
CLS_TO_IDX = { 
    'North': 0, 
    'North Central Coast': 1,
    'South Central Coast': 2, 
    'Central Highland': 3,
    'South East': 4,
    'South West': 5,
}

# Mapping ngược từ số sang chữ để in kết quả
IDX_TO_CLS = {v: k for k, v in CLS_TO_IDX.items()}

def process_filepath(df, base_data_folder):
    """
    Hàm xử lý đường dẫn file âm thanh dựa trên cấu trúc folder.
    base_data_folder: Đường dẫn đến thư mục chứa các folder con '01', '02',...
    """
    # Đường dẫn các folder con
    regions = {
        1: os.path.join(base_data_folder, "01"),
        2: os.path.join(base_data_folder, "02"),
        3: os.path.join(base_data_folder, "03"),
        4: os.path.join(base_data_folder, "04"),
        5: os.path.join(base_data_folder, "05"),
        6: os.path.join(base_data_folder, "06"),
    }

    # Cập nhật lại đường dẫn trong dataframe
    for i, row in df.iterrows():
        cls_id = row["classID"]
        sub_fold = str(row["subregion_fold"]).zfill(2) # Thêm số 0 vào trước nếu cần (ví dụ 1 -> 01)
        filename = row["filename"]
        
        if cls_id in regions:
             # Cấu trúc: base/01/01/filename.wav hoặc tương tự
             # Lưu ý: Điều chỉnh logic nối chuỗi này khớp với thực tế folder của bạn trên máy local
             new_path = os.path.join(regions[cls_id], sub_fold, filename)
             df.at[i, "filename"] = new_path
             
    return df