import cv2
import numpy as np


def extract_by_color_threshold(image, point_roi, radius, threshold=40):
    # 提取 point 区域
    point = image[point_roi[0] - radius:point_roi[0] + radius, point_roi[1] - radius:point_roi[1] + radius]

    # 计算 point 区域的颜色均值
    mean_color = np.mean(point, axis=(0, 1))  # 计算 point 区域的 BGR 均值

    # 设置颜色范围阈值
    lower_bound = np.clip(mean_color - threshold, 0, 255)  # 下界
    upper_bound = np.clip(mean_color + threshold, 0, 255)  # 上界

    # 生成掩码 (将图像转换到HSV或保留BGR看情况)
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # 找到掩码中符合颜色范围的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在图像上画出轮廓框
    result_image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # 获取最小矩形框
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 在图像上画出绿色框

    return result_image, mask, lower_bound, upper_bound
def detect_red_dots(image,lower_bound, upper_bound, region_top_left, region_bottom_right):
    # 转换图像到HSV色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # 定义红色的HSV范围 (这里选择两个范围，因为红色在HSV中跨越180度)
    # lower_red1 = np.array([0, 50, 50])  # 红色的下界
    # upper_red1 = np.array([10, 255, 255])  # 红色的上界
    # lower_red2 = np.array([170, 50, 50])
    # upper_red2 = np.array([180, 255, 255])
    #
    # # 创建掩码，过滤出红色部分
    # mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    # mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    # red_mask = cv2.bitwise_or(mask1, mask2)

    red_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # 指定的检测区域，裁剪图像和掩码
    region = image[region_top_left[1]:region_bottom_right[1], region_top_left[0]:region_bottom_right[0]]
    region_mask = red_mask[region_top_left[1]:region_bottom_right[1], region_top_left[0]:region_bottom_right[0]]
    # cv2.imshow("region", region)

    # 寻找红色点的轮廓
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    red_dots_positions = []

    for contour in contours:
        # 计算每个红色区域的中心点
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]) + region_top_left[0]
            cY = int(M["m01"] / M["m00"]) + region_top_left[1]
            red_dots_positions.append((cX, cY))
            # 在原始图像上标记红点
            cv2.circle(image, (cX, cY), 5, (0, 255, 255), -1)  # 绿色圆圈表示检测到的红点

    return red_dots_positions, image

if __name__ == '__main__':
    # 加载图像
    image_path = r'E:\yq\code\DLC_Project\12.jpg'  # 替换为你的图像路径
    image = cv2.imread(image_path)
    roi = [[420,385,490,455], [1600, 20, 1720, 120], [830, 830, 900, 940]]
    # 将ROI区域设为黑色 (0, 0, 0) 对应黑色
    image[roi[0][1]:roi[0][3], roi[0][0]:roi[0][2]] = (0, 0, 0)
    image[roi[1][1]:roi[1][3], roi[1][0]:roi[1][2]] = (0, 0, 0)
    image[roi[2][1]:roi[2][3], roi[2][0]:roi[2][2]] = (0, 0, 0)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # point_roi = [407,983]
    # radius = 5
    # point = image[point_roi[0] - radius:point_roi[0] + radius, point_roi[1] - radius:point_roi[1] + radius]
    # 提取指定的point区域

    point_roi = [983,407]
    radius = 5
    point = hsv_image[point_roi[1] - radius:point_roi[1] + radius, point_roi[0] - radius:point_roi[0] + radius]
    mean_hsv = np.mean(point, axis=(0, 1))  # 计算point区域的HSV均值
    threshold = np.array([10, 10, 10])  # 可以根据需要调整H, S, V的偏差范围
    lower_bound = np.clip(mean_hsv - threshold, [0, 0, 0], [180, 255, 255])  # HSV范围的下界
    upper_bound = np.clip(mean_hsv + threshold, [0, 0, 0], [180, 255, 255])  # HSV范围的上界
    # cv2.imwrite("point.jpg", point)
    # cv2.imshow("point", point)

    # 指定区域的左上和右下坐标 (x1, y1) -> (x2, y2)
    region_top_left = (520, 0)  # 左上角坐标

    region_bottom_right = (1600, 1000)  # 右下角坐标

    test_image = cv2.imread(r"E:\yq\code\DLC_Project\extracted_frame.jpg")

    # 检测指定区域内的红点
    red_dots_positions, output_image = detect_red_dots(test_image,lower_bound, upper_bound, region_top_left, region_bottom_right)

    print(f"red_dots_positions is {red_dots_positions}")

    # 打印红点的位置
    print("Detected red dot positions:", red_dots_positions)

    # 显示检测结果
    cv2.imshow("Red Dots Detection", output_image)

    # 按 'q' 键退出
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cv2.destroyAllWindows()
