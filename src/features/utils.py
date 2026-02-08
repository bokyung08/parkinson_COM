import cv2
import numpy as np

LENGTH_PERCENTS = {"Trunk": 52.9}

def center_calculate(p1_x, p2_x, p1_y, p2_y, percent):
    """두 좌표 사이에서 percent% 지점의 무게중심 좌표를 계산"""
    x_center = p1_x + (p2_x - p1_x) * (percent / 100)
    y_center = p1_y + (p2_y - p1_y) * (percent / 100)
    return x_center, y_center
