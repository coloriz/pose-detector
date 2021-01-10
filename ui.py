import cv2 as cv
import numpy as np
from PIL import ImageFont, Image, ImageDraw


class InformationLayer:
    size = 360, 140
    font_cv = cv.FONT_HERSHEY_SCRIPT_SIMPLEX
    fontsize_cv = 1.8
    font_s = ImageFont.truetype('assets/NanumSquareRoundR.ttf', 24)
    font_m = ImageFont.truetype('assets/NanumSquareRoundR.ttf', 36)
    theme_color = (252, 252, 252, 220)
    shadow_color = (14, 14, 14, 200)
    shadow_shift = 2
    arc_color = (83, 239, 255, 255)
    arc_glow_color = (255, 255, 255, 255)
    arc_xy = 70, 70
    arc_radius = 60
    arc_width = 2
    arc_glow_width = arc_width + 8
    pose_xy = 140, 90
    index_xy = 140, 20

    def __init__(self, pose_name, pose_index, max_pose, duration):
        self.duration = duration
        # Create empty image
        layer = np.full((self.size[1], self.size[0], 4), self.arc_glow_color, np.uint8)
        layer[..., 3] = 0
        # Draw circle
        self.draw_circle(layer)
        # Convert numpy array to PIL image
        layer = Image.fromarray(layer, 'RGBA')
        layer_draw = ImageDraw.Draw(layer)
        # Draw text
        self.draw_text(layer_draw, self.pose_xy, pose_name, self.font_m, stroke_width=1)
        # Draw index
        self.draw_text(layer_draw, self.index_xy, f'{pose_index}', self.font_m, stroke_width=1)
        # Draw order
        self.draw_text(layer_draw, np.add(self.index_xy, (25, 10)), f'/{max_pose}', self.font_s, stroke_width=1)

        self.layer = layer

    @classmethod
    def draw_circle(cls, img: np.ndarray):
        arc_shadow_xy = tuple(np.add(cls.arc_xy, cls.shadow_shift))
        cv.circle(img, arc_shadow_xy, cls.arc_radius, cls.shadow_color, cls.arc_width, cv.LINE_AA)
        cv.circle(img, cls.arc_xy, cls.arc_radius, cls.theme_color, cls.arc_width, cv.LINE_AA)

    @classmethod
    def draw_arc(cls, img: np.ndarray, angle):
        center = cls.arc_xy
        axes = cls.arc_radius, cls.arc_radius
        # Draw progress bar
        cv.ellipse(img, center, axes, -90, 361, angle, cls.arc_color, cls.arc_width + 1, cv.LINE_AA)
        # Draw glow
        # First, draw glow on empty mask with real size and blur it so that it spreads out little bit
        mask = np.zeros((cls.size[1], cls.size[0]), np.uint8)
        cv.ellipse(mask, center, axes, -90, angle - 5, angle + 5, cls.arc_glow_color[3], cls.arc_glow_width, cv.LINE_AA)
        cv.blur(mask, (9, 9), mask)
        # Second, draw glow on empty image with bigger size
        glow = np.zeros((cls.size[1], cls.size[0], 3), np.uint8)
        cv.ellipse(glow, center, axes, -90, angle - 10, angle + 10, cls.arc_glow_color[:3], cls.arc_glow_width + 15, cv.LINE_AA)
        # Finally, put glow on top of the original image and merge two alpha channels
        mask_test = mask
        mask_norm = mask_test.astype(np.float32) / 255
        for i in range(3):
            img[..., i] = (1 - mask_norm) * img[..., i] + mask_norm * glow[..., i]
        img[..., 3] |= mask

    @classmethod
    def draw_text(cls, image_draw: ImageDraw.ImageDraw, xy, text, font, **kwargs):
        shadow_xy = np.add(xy, cls.shadow_shift)
        image_draw.text(shadow_xy, text, cls.shadow_color, font, **kwargs)
        image_draw.text(xy, text, cls.theme_color, font, **kwargs)

    @classmethod
    def draw_text_cv(cls, img: np.ndarray, center, text, font, scale, thickness=None):
        text_size, baseline = cv.getTextSize(text, font, scale, thickness)
        baseline += thickness
        text_org = center[0] - text_size[0] // 2, center[1] + text_size[1] // 2
        shadow_org = tuple(np.add(text_org, cls.shadow_shift))
        cv.putText(img, text, shadow_org, font, scale, cls.shadow_color, thickness, cv.LINE_AA)
        cv.putText(img, text, text_org, font, scale, cls.theme_color, thickness, cv.LINE_AA)

    def alpha_composite(self, bg, x, y, elapsed):
        arr = np.array(self.layer)
        # Draw elasped
        self.draw_text_cv(arr, self.arc_xy, f'{int(elapsed)}', self.font_cv, self.fontsize_cv, 3)
        # Draw progress arc
        angle = 360 * (elapsed / self.duration)
        self.draw_arc(arr, angle)
        # Alpha compositing
        arr_rgb, alpha = arr[:, :, :3], arr[:, :, 3].astype(np.float32) / 255
        bg_roi = bg[y:y + self.size[1], x:x + self.size[0]]
        for i in range(3):
            bg_roi[:, :, i] = (1 - alpha) * bg_roi[:, :, i] + alpha * arr_rgb[:, :, 2 - i]
