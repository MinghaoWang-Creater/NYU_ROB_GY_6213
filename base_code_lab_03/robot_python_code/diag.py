import numpy as np
import cv2
import json
from scipy.optimize import least_squares

# ==========================================
# 1. 加载数据
# ==========================================
def load_data():
    # 假设文件名为 points_data.json
    try:
        with open("points_data.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("错误：找不到 points_data.json 文件")
        return None, None

    X_robot_raw = []
    Y_cam_raw = []
    for entry in data:
        p = entry[0]
        # entry[1:] 包含一帧或多帧相机观测
        for read in entry[1:]:
            # 存储原始数据：[x_m, y_m, theta_deg]
            X_robot_raw.append([p[0]*0.01, p[1]*0.01, p[2]]) 
            Y_cam_raw.append(np.array(read, dtype=np.float64))
    return np.array(X_robot_raw), np.array(Y_cam_raw)

def get_T(tvec, rvec):
    """从 rvec 和 tvec 构造 4x4 变换矩阵"""
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T

X_robot_raw, Y_cam_raw = load_data()
if X_robot_raw is None:
    exit()

# 机器人中心到 Marker 的静态偏移
t_robot_tag = np.array([0.05, -0.05, 0.12], dtype=np.float64)

# ==========================================
# 2. 暴力搜索所有坐标系配置
# ==========================================

# 尝试绕 X 轴旋转的不同角度 (Marker 贴法不同)
rotations = [0, np.pi/2, np.pi, 3*np.pi/2]
# 尝试机器人 Yaw 角的正负 (极性问题)
polarities = [1, -1]

print(f"{'Marker_Rot(X)':<15} | {'Yaw_Sign':<10} | {'Resulting MSE':<15}")
print("-" * 50)

best_overall_mse = float('inf')
best_config = None

for r_x in rotations:
    # 构造修正矩阵，强制使用 float64
    R_flip = cv2.Rodrigues(np.array([r_x, 0.0, 0.0], dtype=np.float64))[0]
    
    for sign in polarities:
        def residual_func(params):
            t_wc, r_wc, scale = params[:3], params[3:6], params[6]
            T_wc = get_T(t_wc, r_wc)
            T_cw = np.linalg.inv(T_wc)
            
            res = []
            for i in range(len(X_robot_raw)):
                # 机器人位姿 (应用假设的 Yaw 极性)
                yaw = X_robot_raw[i, 2] * np.pi / 180.0 * sign
                T_wr = get_T([X_robot_raw[i, 0], X_robot_raw[i, 1], 0.0], [0.0, 0.0, yaw])
                
                # 观测值处理
                R_obs_raw, _ = cv2.Rodrigues(Y_cam_raw[i, 3:])
                R_obs = R_obs_raw @ R_flip
                t_obs = Y_cam_raw[i, :3] * scale
                
                # 理论预测
                T_rt = np.eye(4)
                T_rt[:3, 3] = t_robot_tag
                T_ct_pred = T_cw @ T_wr @ T_rt
                
                # 平移残差
                res.extend(T_ct_pred[:3, 3] - t_obs)
                
                # 旋转残差 (R_pred * R_obs^T -> 旋转向量误差)
                R_pred = T_ct_pred[:3, :3]
                R_err_mat = R_pred @ R_obs.T
                r_err, _ = cv2.Rodrigues(R_err_mat)
                res.extend(r_err.flatten())
                
            return np.array(res)

        # 初始猜测：高度 z=1.7, 绕 X 旋转 pi(向下看), 缩放 1.397
        p0 = np.array([0.0, 0.0, 1.7, np.pi, 0.0, 0.0, 1.397], dtype=np.float64)
        
        # 快速拟合 (限制迭代次数)
        sol = least_squares(residual_func, p0, loss='soft_l1', max_nfev=60)
        mse = np.mean(sol.fun**2)
        
        print(f"{np.degrees(r_x):<15.1f} | {sign:<10} | {mse:<15.6f}")
        
        if mse < best_overall_mse:
            best_overall_mse = mse
            best_config = (r_x, sign, sol.x)

# ==========================================
# 3. 输出冠军配置
# ==========================================
r_x_best, sign_best, final_p = best_config
print("\n" + "="*45)
print("🎯 找到最佳坐标系匹配方案！")
print("="*45)
print(f"Marker 绕 X 轴修正角度: {np.degrees(r_x_best):.1f} 度")
print(f"机器人 Yaw 极性 (1为正常, -1为反向): {sign_best}")
print(f"最终平均残差 (MSE): {best_overall_mse:.6f}")
print("-" * 45)
print(f"最终相机世界坐标 (x,y,z): {final_p[:3]}")
print(f"最终优化 Scale: {final_p[6]:.4f}")
print("="*45)