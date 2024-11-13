
import numpy as np
def calculate_rotation_angle(v1, v2):
    # 将二维数组转换为一维数组，避免形状错误
    v1 = np.array(v1).flatten()  # 将 v1 变为 1D 数组
    v2 = np.array(v2).flatten()  # 将 v2 变为 1D 数组
    # 计算点积
    dot_product = np.dot(v1, v2)

    # 计算叉积的z分量，适用于二维向量 (A_x * B_y - A_y * B_x)
    cross_product_z = v1[0] * v2[1] - v1[1] * v2[0]

    # 计算旋转角度，atan2 会返回弧度，考虑点积和叉积的符号
    angle_radians = np.arctan2(cross_product_z, dot_product)

    # 将弧度转换为角度
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

v1 = [0,1]
v2 = [0,-1]
angle_degrees = calculate_rotation_angle(v1,v2)
print(f"angle_degrees is {angle_degrees}")