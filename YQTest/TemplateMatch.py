import cv2
import numpy as np

def match_template_rgb(template, image):
    """
    使用 RGB 图像进行模板匹配，找到模板在目标图像中的位置。
    
    :param template_path: 模板图像路径
    :param image_path: 目标图像路径
    :return: 匹配区域范围 (x, y, width, height) 和中心点 (center_x, center_y)
    """
    # 读取模板和目标图像（保持为彩色格式）
    

    # 检查图像是否成功加载
    if template is None or image is None:
        raise ValueError("无法加载模板图像或目标图像，请检查路径是否正确！")

    # 模板匹配（适用于彩色图像）
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # 获取匹配结果中最大值的位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 模板的宽度和高度
    h, w, _ = template.shape

    # 匹配的左上角点
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 计算中心点
    center_x = top_left[0] + w // 2
    center_y = top_left[1] + h // 2

    # 返回匹配区域和中心点
    return (top_left[0], top_left[1], w, h), (center_x, center_y)
if __name__ == "__main__":

    # 示例用法
    template_path = 'Template.jpg'  # 替换为模板图像路径
    image_path = 'frame1.jpg'        # 替换为目标图像路径
    template = cv2.imread(template_path)
    image = cv2.imread(image_path)
    roi_range, center = match_template_rgb(template, image)
    print(f"ROI范围: {roi_range}")
    print(f"中心点: {center}")

    # 可视化结果
    image = cv2.imread(image_path)
    top_left = (roi_range[0], roi_range[1])
    bottom_right = (roi_range[0] + roi_range[2], roi_range[1] + roi_range[3])
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # 画矩形
    cv2.circle(image, center, 5, (255, 0, 0), -1)  # 标记中心点
    cv2.imshow('Matched Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
