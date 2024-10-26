import cv2
import numpy as np

class calib():
    def __init__(self):
        # fx = 1000.19
        # fy = 998.87
        # cx = 915.85
        # cy = 586.99
        # k1 = -0.09
        # k2 = 0.28
        # k3 = -0.3
        # p1 = 0
        # p2 = -0.01

        # self.rvec =np.array ([[-0.0081371 ],
        #     [ 0.03347018],
        #     [ 0.02776796]])
        # self.tvec = np.array( [[ 511.99394532],
        #     [-733.78755916],
        #     [1727.20415713]])

        fx = 964.12
        fy = 969.67
        cx = 968.65
        cy = 557.79
        k1 = -0.17
        k2 = 0.49
        k3 = -0.56
        p1 = 0
        p2 = 0.0
        self.rvec =np.array ([[ 0.00504822],
        [ 0.06811006],
        [-1.58667399]])
        self.tvec = np.array([[  63.33747746],
        [-252.18241015],
        [1017.71342447]])
        
        # 已知内参矩阵和畸变系数 (根据你的实际情况替换)
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])  # 你的相机内参矩阵
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])  # 你的畸变系数

        

        self.rotation_matrix, _ = cv2.Rodrigues(self.rvec)
        pass

    def Test(self):
        # image_points = np.array([[1205, 169],
        #             [1347, 173],
        #             [1344, 354],
        #             [1202, 351]
        #             ] , dtype= np.float64)
        image_points = np.array([[1028, 320],
                    [1024, 84],
                    [1327, 76],
                    [1329, 314]
                    ] , dtype= np.float64)
        object_points = np.array([[0,0,0],
                       [250,0,0],
                       [250,320,0],
                       [0,320,0]],dtype=np.float64)
        # 使用PnP求解外参 (旋转向量rvec和平移向量tvec)
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs)

        # 打印结果
        print("旋转向量 (rvec):", rvec)
        print("平移向量 (tvec):", tvec)

        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        print("旋转矩阵 (rotation_matrix):", rotation_matrix)
        
    def projectPoint(self,image_points):
        # 假设你有2D点 (u, v)
        # image_points = np.array([[320, 240]], dtype=np.float64)

        # 去除畸变
        undistorted_point = cv2.undistortPoints(image_points, self.camera_matrix, self.dist_coeffs)
        # undistorted_point = np.append(undistorted_point, [[1]], axis=1).T  # 归一化相机坐标

        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(self.rvec)
        print(f"rotation_matrix is {rotation_matrix}")

        # 假设点在 Z_w = 0 平面上 (同样的平面)
        Z_w = 0

        # 计算相机坐标系下的3D点 (X_c, Y_c, Z_c)，在归一化平面上
        camera_coords = np.linalg.inv(rotation_matrix).dot(undistorted_point * Z_w - self.tvec)

        # 输出结果，该点在平面上的 3D 世界坐标
        print("像素点 (u, v) 对应的平面上的 3D 坐标 (X_w, Y_w, Z_w=0):", camera_coords.ravel())
    def projectPointNoUnistort(self,u, v):
        camera_matrix = self.camera_matrix
        dist_coeffs = self.dist_coeffs
        rotation_matrix = self.rotation_matrix
        tvec = self.tvec
        rvec = self.rvec


       # 去畸变 (假设有畸变，先去畸变)
        undistorted_point = cv2.undistortPoints(np.array([[u, v]], dtype=np.float64), camera_matrix, dist_coeffs)
        # undistorted_point = np.append(undistorted_point, [[[3]]], axis=2).T
        # 将2D像素坐标转换为归一化的相机坐标
        # undistorted_point = np.array([undistorted_point[0,0,0], undistorted_point[0,0,1], 1], dtype=np.float64).reshape(3, 1)

        # 假设点位于 Z_w = 0 的平面上 (与标定平面相同)
        Z_w = tvec[2]

        # 使用内参矩阵逆矩阵将归一化相机坐标转换为 3D 点
        uv1 = np.append(undistorted_point[0], 1).reshape(3, 1)  # 将去畸变后的点转为3x1矩阵
        camera_coords = uv1 * Z_w

        # 转换为世界坐标
        world_coords = np.linalg.inv(rotation_matrix).dot(camera_coords - tvec)

        # 输出 3D 世界坐标
        print("像素点对应的3D世界坐标 (X_w, Y_w, Z_w=0):", world_coords.ravel())
        return world_coords

if __name__ == '__main__':
    import math
    mycalib = calib()
    mycalib.Test()
    # point = np.array([600, 1200])
    # point = np.array([[472, 986]], dtype=np.float64)
    point = np.array([[472, 986]], dtype=np.float64)
    # redPoint = [858, 355]
    # mice = [809, 515]
    redPoint = [424,  233]
    mice = [592,286]
    world_coords1 = mycalib.projectPointNoUnistort(redPoint[0], redPoint[1])
    world_coords2 = mycalib.projectPointNoUnistort(mice[0], mice[1])
    # diff = wor
    vector = world_coords1 - world_coords2
    print(f"vector is {vector}")
    block = 12800 / 72
    distX = vector[0] * block
    distY = vector[1] * block
    print(f"distX is {distX} distY is {distY} ")
    print(f" distance is {math.sqrt(vector[0]*vector[0] + vector[1] * vector[1])}")
    # point2 = np.array([[252, 996]], dtype=np.float64)
    # mycalib.projectPoint(point2)

    pass