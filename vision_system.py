import cv2
import mediapipe as mp
import math


class VisionSystem:
    """
    视觉处理系统
    功能：去除黑边 -> 提取有效画面 -> 识别鼻孔目标
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[Vision] MediaPipe 模型已加载")

    def crop_effective_area(self, image):
        """裁切黑边，获取有效视野"""
        if image is None:
            return None, (0, 0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image, (0, 0)

        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # 过滤噪点：有效面积太小则不裁剪
        if w * h < (image.shape[0] * image.shape[1] * 0.05):
            return image, (0, 0)

        cropped_image = image[y:y + h, x:x + w]
        return cropped_image, (x, y)

    def detect_nose_target(self, effective_image, target_side='center'):
        """
        寻找目标：基于几何插值定位精确鼻孔中心
        target_side: 'left' (人脸左鼻孔), 'right' (人脸右鼻孔), 'center' (中间)
        """
        if effective_image is None:
            return None, 0.0, 0.0

        h, w = effective_image.shape[:2]
        rgb_image = cv2.cvtColor(effective_image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 279: 左鼻翼底部基准点
            # 49:  右鼻翼底部基准点
            pt_base_left = landmarks[279]
            pt_base_right = landmarks[49]

            # 鼻小柱中点
            mid_x = (pt_base_left.x + pt_base_right.x) * 0.5
            mid_y = (pt_base_left.y + pt_base_right.y) * 0.5

            target_norm_x = 0.0
            target_norm_y = 0.0

            target_side_str = str(target_side).strip().lower()
            if target_side_str == 'left':
                # 左鼻孔中心 = 左鼻翼点 与 中点 的中点
                target_norm_x = (pt_base_left.x + mid_x) * 0.5
                target_norm_y = (pt_base_left.y + mid_y) * 0.5

            elif target_side_str == 'right':
                # 右鼻孔中心 = 右鼻翼点 与 中点 的中点
                target_norm_x = (pt_base_right.x + mid_x) * 0.5
                target_norm_y = (pt_base_right.y + mid_y) * 0.5

            else:
                # 默认中点
                target_norm_x = mid_x
                target_norm_y = mid_y

            tx = int(target_norm_x * w)
            ty = int(target_norm_y * h)

            # 鼻孔基准间距：用于深度比例估算
            nostril_distance_px = math.hypot(
                (pt_base_left.x - pt_base_right.x) * w,
                (pt_base_left.y - pt_base_right.y) * h
            )

            # 鼻翼总宽度：用于到达判定
            left_wing_outer = landmarks[358]
            right_wing_outer = landmarks[129]
            feature_width = math.hypot(
                (left_wing_outer.x - right_wing_outer.x) * w,
                (left_wing_outer.y - right_wing_outer.y) * h
            )

            return (tx, ty), float(feature_width), float(nostril_distance_px)

        return None, 0.0, 0.0

    def release(self):
        """释放MediaPipe资源"""
        if hasattr(self, 'mp_face_mesh'):
            self.mp_face_mesh.close()
            print("[Vision] MediaPipe 资源已释放")