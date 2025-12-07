import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# ============================================
# 配置区：根据你本机路径/文件名修改
# ============================================
SUBMISSION_DATA_DIR = "m5_baseline\m5-forecasting-main\submissions"
DATA_DIR = "m5_baseline\m5-forecasting-main\data"
PRED_FILE = "submission_accuracy_ngb_validation.csv"  # 你队友的预测文件名

QUANTILE_FILE = "submission_uncertainty_ngb_validation.csv"  # 你队友的分位点预测文件名

SALES_EVAL_FILE = "sales_train_evaluation.csv"   # 用来算 scale 和权重
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

# 如果有 sales_train_evaluation.csv，就用它做真值；否则就用 validation
SALES_EVAL_FILE_CANDIDATE = "sales_train_evaluation.csv"

FIRST = 1914


def compute_single_rmsse(row, forecast_cols, history_cols):
    """计算单个序列的 RMSSE 和 RMSE"""
    id_val = row['id']
    
    forecast_true = np.array([row[f'{col}_true'] for col in forecast_cols])
    forecast_pred = np.array([row[f'{col}_pred'] for col in forecast_cols])
    mse_forecast = np.mean((forecast_true - forecast_pred) ** 2)

    history_values = np.array([row[col] for col in history_cols])
    
    actual_values = history_values[1:]
    naive_pred = history_values[:-1]
    mse_naive = np.mean((actual_values - naive_pred) ** 2)

    if mse_naive == 0:
        rmsse = 0.0 if mse_forecast == 0 else np.inf
    else:
        rmsse = np.sqrt(mse_forecast / mse_naive)
    
    return {
        'id': id_val,
        'RMSSE': rmsse,
        'RMSE': np.sqrt(mse_forecast)
    }

def compute_rmsse(pred, sales_eval):
    forecast_cols = [f'd_{i}' for i in range(1914, 1942)]
    history_cols = [f'd_{i}' for i in range(1, 1914)]

    pred_forecast_names = [f'F{i}' for i in range(1, 29)]
    col_mapping = dict(zip(pred_forecast_names, forecast_cols))
    pred.rename(columns=col_mapping, inplace=True)
    
    pred_data = pred[['id'] + forecast_cols].copy()
    sales_data = sales_eval[['id'] + forecast_cols + history_cols].copy()
    
    merged = pred_data.merge(sales_data, on='id', suffixes=('_pred', '_true'))
    total_series = len(merged)
    
    print(f"开始并行计算 {total_series} 个序列...")
    
    # 并行计算，n_jobs=-1 表示使用所有 CPU 核心
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(compute_single_rmsse)(row, forecast_cols, history_cols) 
        for _, row in tqdm(merged.iterrows(), total=total_series, desc="计算进度")
    )
    
    result_df = pd.DataFrame(results)
    return result_df

def evaluate_point_forecast(file_name=None):

    pred_path = os.path.join(SUBMISSION_DATA_DIR, file_name)
    eval_path = os.path.join(DATA_DIR, SALES_EVAL_FILE)

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"找不到预测文件: {pred_path}")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"找不到销量训练文件: {eval_path}")

    print("读取队友预测文件:", pred_path)
    pred = pd.read_csv(pred_path)

    print("读取 sales_train_evaluation:", eval_path)
    sales_eval = pd.read_csv(eval_path) # ground truth 用 evaluation（包含 d_1914~d_1941）

    pred = pred[pred['id'].str.contains('validation')] # 只保留 validation 部分预测

    # ===== 关键修正：统一 id 后缀, 把"evaluation"和"validation"去掉 =====
    def normalize_ids(df):
        if "id" in df.columns:
            df["id"] = df["id"].astype(str).str.replace("_validation", "").str.replace("_evaluation", "")
        return df

    pred = normalize_ids(pred)
    sales_eval = normalize_ids(sales_eval)

    res = compute_rmsse(pred, sales_eval)

    # 简单 summary
    print('=' * 60)
    print("RMSSE 计算结果汇总")
    print('=' * 60)
    print(f"总序列数: {len(res)}")
    
    print(f"\nRMSSE 统计:")
    print(f"  平均值: {res['RMSSE'].mean():.6f}")
    print(f"  中位数: {res['RMSSE'].median():.6f}")
    # 特殊值统计
    inf_count = np.isinf(res['RMSSE']).sum()
    zero_count = (res['RMSSE'] == 0).sum()
    print(f"\n特殊值统计:")
    print(f"  RMSSE = inf 的序列数: {inf_count}")
    print(f"  RMSSE = 0 的序列数: {zero_count}")
    print('=' * 60)
    
    print(f"\nRMSE 统计:")
    print(f"  平均值: {res['RMSE'].mean():.6f}")
    print(f"  中位数: {res['RMSE'].median():.6f}")
    
    # 保存结果
    output_path = os.path.join(DATA_DIR, f"metrics_accuracy_{file_name}")
    res.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")


def pinball_loss(y_true, y_pred, quantile):
    """计算单个分位数的 Pinball Loss"""
    error = y_true - y_pred
    return np.where(error >= 0, quantile * error, (quantile - 1) * error)

def compute_single_spl(base_id, group_data, forecast_cols, history_cols, sales_eval):
    """计算单个序列（所有分位数）的 SPL"""
    # 提取该 base_id 对应的真实值
    sales_row = sales_eval[sales_eval['id'] == base_id]
    if sales_row.empty:
        return {'id': base_id, 'SPL': np.nan}
    
    # 获取预测期真实值 (d_1914 ~ d_1941)
    y_true = sales_row[forecast_cols].values.flatten()
    
    # 计算 scale：历史期 (d_1 ~ d_1913) 的一阶差分绝对值的平均
    history_values = sales_row[history_cols].values.flatten()
    diffs = np.abs(np.diff(history_values))
    scale = np.mean(diffs)
    
    if scale == 0:
        scale = 1.0  # 避免除零
    
    # 对该 base_id 的所有分位数预测计算 pinball loss
    pinball_losses = []
    for _, row in group_data.iterrows():
        quantile = row['quantile']
        y_pred = row[forecast_cols].values
        
        # 计算该分位数的 pinball loss
        pl = pinball_loss(y_true, y_pred, quantile)
        pinball_losses.append(np.mean(pl))
    
    # 对所有分位数的 pinball loss 求平均，再除以 scale
    avg_pinball_loss = np.mean(pinball_losses)
    spl = avg_pinball_loss / scale
    
    return {'id': base_id, 'SPL': spl,'PL': avg_pinball_loss}

def evaluate_quantile_forecast(file_name=None):
    pred_path = os.path.join(SUBMISSION_DATA_DIR, file_name)
    eval_path = os.path.join(DATA_DIR, SALES_EVAL_FILE)

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"找不到预测文件: {pred_path}")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"找不到销量训练文件: {eval_path}")

    print("读取分位数预测文件:", pred_path)
    pred = pd.read_csv(pred_path)

    print("读取 sales_train_evaluation:", eval_path)
    sales_eval = pd.read_csv(eval_path)

    pred['id'] = pred['id'].astype(str).str.replace("_validation", "")
    # 解析 id，提取 base_id 和 quantile
    # 格式: id1_0.005_validation -> base_id: id1, quantile: 0.005
    pred['base_id'] = pred['id'].str.rsplit('_', n=1).str[0]
    pred['quantile'] = pred['id'].str.rsplit('_', n=1).str[1].astype(float)
        
    sales_eval['id'] = sales_eval['id'].astype(str).str.replace("_evaluation", "")
    
    # 定义预测期和历史期的列
    forecast_cols = [f'd_{i}' for i in range(1914, 1942)]  # d_1914 ~ d_1941
    history_cols = [f'd_{i}' for i in range(1, 1914)]      # d_1 ~ d_1913
    
    # 重命名预测列 F1-F28 为 d_1914-d_1941
    pred_forecast_names = [f'F{i}' for i in range(1, 29)]
    col_mapping = dict(zip(pred_forecast_names, forecast_cols))
    pred.rename(columns=col_mapping, inplace=True)
    
    # 按 base_id 分组
    grouped = pred.groupby('base_id')
    unique_ids = pred['base_id'].unique()
    total_series = len(unique_ids)
    
    print(f"开始并行计算 {total_series} 个序列的 SPL...")
    
    # 并行计算每个 base_id 的 SPL
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(compute_single_spl)(
            base_id, 
            grouped.get_group(base_id), 
            forecast_cols, 
            history_cols, 
            sales_eval
        )
        for base_id in tqdm(unique_ids, desc="计算 SPL 进度")
    )
    
    result_df = pd.DataFrame(results)
    
    # 简单 summary
    print('=' * 60)
    print("SPL 计算结果汇总")
    print('=' * 60)
    print(f"总序列数: {len(result_df)}")
    print(f"\nSPL 统计:")
    print(f"  平均值: {result_df['SPL'].mean():.6f}")
    print(f"  中位数: {result_df['SPL'].median():.6f}")
    
    # 特殊值统计
    nan_count = result_df['SPL'].isna().sum()
    print(f"\n特殊值统计:")
    print(f"  SPL = NaN 的序列数: {nan_count}")
    print('=' * 60)

    print(f"\nPL 统计:")
    print(f"  平均值: {result_df['PL'].mean():.6f}")
    print(f"  中位数: {result_df['PL'].median():.6f}")
    
    # 保存结果
    output_path = os.path.join(DATA_DIR, f"metrics_uncertainty_{file_name}")
    result_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")
    
    return result_df


if __name__ == "__main__":

    # 评估点预测
    # file_names = ['submission_accuracy_ngb_validation.csv',
    #               'submission_accuracy_pois_ts_validation.csv']
    # for file_name in file_names:
    #     evaluate_point_forecast(file_name)
    
    # 评估分位数预测
    quantile_files = ['submission_uncertainty_pois_ts_validation.csv','submission_uncertainty_ngb_validation.csv']
    for file_name in quantile_files:
        evaluate_quantile_forecast(file_name)
