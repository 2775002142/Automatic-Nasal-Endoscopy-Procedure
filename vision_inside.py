# 文件路径: ~/FR5/src/fr5_vision_control/fr5_vision_control/vision_inside.py

import cv2
import numpy as np
import math
# 从 utils 导入 clamp (因为不再依赖 move_inside.py)
from fr5_vision_control.utils import clamp 

# ==========================================
# 视觉处理系统 (基于APF算法) 
# ==========================================
class APFVisionSystem:
    """
    基于人工势场法(APF)的视觉处理系统
    """
    def __init__(self, debug=False):
        self.K_ATT = 5.0
        self.K_REP = 3.0
        self.REP_RANGE = 80.0
        self.MIN_AREA_RATIO = 0.005

        self.BASE_DARK_PERCENTILE = 8
        self.BASE_BRIGHT_PERCENTILE = 80

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.debug = debug

        self.center = (0, 0)
        self.crop_offset = (0, 0)
        self.width = 0
        self.height = 0

    def process_frame(self, image):
        if image is None:
            return None, np.array([0.0, 0.0]), None # 统一返回None给vis_image

        image_cropped, self.crop_offset = self._crop_black_borders(image)
        self.height, self.width = image_cropped.shape[:2]
        if self.width == 0 or self.height == 0: # 处理裁剪后图像为空的情况
             return image, np.array([0.0, 0.0]), None

        self.center = (self.width // 2, self.height // 2)

        gray_image = self._preprocess_image(image_cropped)
        dark_regions, bright_regions = self._find_regions(gray_image)
        goal = self._select_goal(dark_regions)

        if goal:
            attractive_force = self._calculate_attractive_force(goal)
            repulsive_force = self._calculate_repulsive_force(bright_regions)
            total_force = attractive_force + repulsive_force
        else:
            total_force = np.array([0.0, 0.0])

        # _visualize_result 仍然会调用，但现在它接受 filtered_goal
        vis_image = self._visualize_result(image, total_force, goal, dark_regions, bright_regions)
        return vis_image, total_force, goal

    def _crop_black_borders(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, (0, 0)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        if cv2.contourArea(max_contour) / (image.shape[0] * image.shape[1]) < 0.1:
            return image, (0, 0)
        return image[y:y+h, x:x+w], (x, y)

    def _preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        return cv2.GaussianBlur(enhanced, (7, 7), 0)

    def _find_regions(self, gray_img):
        dark_thresh = np.percentile(gray_img, self.BASE_DARK_PERCENTILE)
        bright_thresh = np.percentile(gray_img, self.BASE_BRIGHT_PERCENTILE)

        dark_mask = cv2.inRange(gray_img, 0, int(dark_thresh))
        bright_mask = cv2.inRange(gray_img, int(bright_thresh), 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        min_area = self.MIN_AREA_RATIO * self.height * self.width
        d_cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        b_cnts, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return ([c for c in d_cnts if cv2.contourArea(c) > min_area],
                [c for c in b_cnts if cv2.contourArea(c) > min_area])

    def _select_goal(self, regions):
        if not regions:
            return None
        best_goal, max_score = None, -1
        for region in regions:
            M = cv2.moments(region)
            if M['m00'] == 0:
                continue
            area = M['m00']
            cx, cy = int(M['m10'] / area), int(M['m01'] / area)
            score = (cy / self.height * 0.2) + (area / (self.width * self.height) * 0.8)
            if score > max_score:
                max_score, best_goal = score, (cx, cy)
        return best_goal

    def _calculate_attractive_force(self, goal):
        dx, dy = goal[0] - self.center[0], goal[1] - self.center[1]
        return self.K_ATT * np.array([dx, dy])

    def _calculate_repulsive_force(self, obstacles):
        total_rep = np.array([0.0, 0.0])
        center_pt = (float(self.center[0]), float(self.center[1]))

        for obs in obstacles:
            if cv2.pointPolygonTest(obs, center_pt, False) > 0:
                M = cv2.moments(obs)
                if M['m00'] == 0:
                    continue
                ox, oy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                vec = np.array([self.center[0] - ox, self.center[1] - oy])
                norm = np.linalg.norm(vec)
                if norm > 0:
                    total_rep += self.K_REP * 100 * (vec / norm)
                continue

            dist_sq_min = float('inf')
            closest_pt = None
            for pt_wrapper in obs:
                pt = pt_wrapper[0]
                d_sq = (self.center[0] - pt[0]) ** 2 + (self.center[1] - pt[1]) ** 2
                if d_sq < dist_sq_min:
                    dist_sq_min, closest_pt = d_sq, pt

            if closest_pt is None:
                continue
            dist = math.sqrt(dist_sq_min)
            if 0 < dist < self.REP_RANGE:
                vec = np.array([self.center[0] - closest_pt[0], self.center[1] - closest_pt[1]])
                mag = self.K_REP * ((1.0 / dist) - (1.0 / self.REP_RANGE)) * (1.0 / dist ** 2)
                total_rep += mag * (vec / dist)
        return total_rep

    def _visualize_result(self, original_img, force, goal, darks, brights, filtered_goal=None):
        vis = original_img.copy()
        ox, oy = self.crop_offset
        # 确保 center 不为 (0,0)
        if self.width == 0 or self.height == 0: # 修正这里，如果裁剪后尺寸为0
            return vis
            
        cx, cy = self.center[0] + ox, self.center[1] + oy

        # 中心点 (黄色)
        cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)

        # 原始检测目标点 (橙色空心圆)
        if goal is not None:
            gx_abs = goal[0] + ox
            gy_abs = goal[1] + oy
            cv2.circle(vis, (gx_abs, gy_abs), 8, (0, 165, 255), 2)

        # 卡尔曼滤波后的目标点 (绿色实心圆) - 这是我们主要跟随的点
        if filtered_goal is not None:
            fgx_abs = int(filtered_goal[0] + ox)
            fgy_abs = int(filtered_goal[1] + oy)
            cv2.circle(vis, (fgx_abs, fgy_abs), 7, (0, 255, 0), -1)

        darks_abs = [c + (ox, oy) for c in darks]
        brights_abs = [c + (ox, oy) for c in brights]
        cv2.drawContours(vis, darks_abs, -1, (255, 0, 0), 1)
        cv2.drawContours(vis, brights_abs, -1, (0, 0, 255), 1)

        # 力场向量
        if np.linalg.norm(force) > 1e-4:
            unit_vec = force / np.linalg.norm(force)
            end_pt = (int(cx + unit_vec[0] * 50), int(cy + unit_vec[1] * 50))
            cv2.arrowedLine(vis, (cx, cy), end_pt, (0, 255, 0), 2)

        return vis

# ==========================================
# 力场向量 → 旋转角度 + 平移量 转换器 (ForceToMotionConverter)
# ==========================================
class ForceToMotionConverter:
    """
    将APF输出的像素级力场向量转换为工具坐标系下的旋转角度和平移量。
    - 动态增益：力场越大，旋转/平移调整幅度越大。
    - 非线性映射：使用曲线因子使调整在接近目标时更精细。
    """
    def __init__(self,
                 force_deadzone=10.0,
                 max_force_for_scale=1150.0, # 假设力场幅度在此范围内达到最大增益
                 min_rotation_gain=0.001,
                 max_rotation_gain=0.012,
                 rotation_gain_curve_factor=0.6,
                 max_rotation_deg=0.5, # 注意这里是单次运动的最大角度，不是累计
                 min_translate_step_mm=0.02,
                 max_translate_step_mm=0.15,
                 translate_step_curve_factor=0.7,
                 max_translate_per_phase_mm=1.0): # 注意这里是单次运动的最大平移，不是累计
        
        self.force_deadzone = force_deadzone
        self.max_force_for_scale = max_force_for_scale
        self.min_rotation_gain = min_rotation_gain
        self.max_rotation_gain = max_rotation_gain
        self.rotation_gain_curve_factor = rotation_gain_curve_factor
        self.max_rotation_deg = max_rotation_deg
        self.min_translate_step_mm = min_translate_step_mm
        self.max_translate_step_mm = max_translate_step_mm
        self.translate_step_curve_factor = translate_step_curve_factor
        self.max_translate_per_phase_mm = max_translate_per_phase_mm

    def _calculate_dynamic_params(self, force_mag):
        if force_mag < self.force_deadzone:
            return 0.0, 0.0

        # 将力场幅度归一化到 [0, 1] 范围
        norm_force = clamp((force_mag - self.force_deadzone) / (self.max_force_for_scale - self.force_deadzone), 0.0, 1.0)
        
        # 计算动态旋转增益 (非线性曲线)
        rot_gain = self.min_rotation_gain + (self.max_rotation_gain - self.min_rotation_gain) * (norm_force ** self.rotation_gain_curve_factor)
        # 计算动态平移步长 (非线性曲线)
        trans_step = self.min_translate_step_mm + (self.max_translate_step_mm - self.min_translate_step_mm) * (norm_force ** self.translate_step_curve_factor)
        
        return rot_gain, trans_step

    def convert(self, force_vector, current_phase_dx, current_phase_dy):
        fx, fy = float(force_vector[0]), float(force_vector[1])
        force_mag = math.sqrt(fx**2 + fy**2)

        dyn_rot_gain, dyn_trans_step = self._calculate_dynamic_params(force_mag)
        if dyn_rot_gain == 0.0 and dyn_trans_step == 0.0:
            return 0.0, 0.0, 0.0, 0.0

        # !! 重要: 旋转正负号需根据实际机器人和相机安装测试 !!
        # 假设：
        # +fy (目标偏下) -> -Rx (下俯)  (通常 Rx 增加代表绕X轴正方向旋转，导致Y轴坐标减小，即仰起，所以需要负号来下俯)
        # +fx (目标偏右) -> +Ry (右转)  (通常 Ry 增加代表绕Y轴正方向旋转，导致X轴坐标减小，即左转，所以需要正号来右转)
        # 请根据实际测试验证这些符号！
        rx_deg = -fy * dyn_rot_gain
        ry_deg = fx * dyn_rot_gain
        
        # 限制单次旋转角度
        rx_deg = clamp(rx_deg, -self.max_rotation_deg, self.max_rotation_deg)
        ry_deg = clamp(ry_deg, -self.max_rotation_deg, self.max_rotation_deg)

        dx_mm, dy_mm = 0.0, 0.0
        if force_mag > 1e-6:
            dir_x, dir_y = fx / force_mag, fy / force_mag
            
            # 假设:
            # +fx (目标偏右) -> +dx (平移向右)
            # +fy (目标偏下) -> +dy (平移向下)
            dx_cand = dir_x * dyn_trans_step
            dy_cand = dir_y * dyn_trans_step
            
            # 检查阶段累计限制 (如果当前累计加上本次移动会超限，则本次移动为0)
            if abs(current_phase_dx + dx_cand) <= self.max_translate_per_phase_mm:
                dx_mm = dx_cand
            if abs(current_phase_dy + dy_cand) <= self.max_translate_per_phase_mm:
                dy_mm = dy_cand
        
        return rx_deg, ry_deg, dx_mm, dy_mm
