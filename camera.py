#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense Viewer (Numeric Control Only)
 - 슬라이더 제거, 숫자 입력만으로 카메라 옵션 조정
 - 영상과 조정 패널 완전 분리
 - 창 크기 변경/최대화 시 부드럽게 비율 유지
"""

import tkinter as tk
from tkinter import ttk
import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image, ImageTk

# ============================================================
# ✅ RealSense 초기화
# ============================================================
pipeline = rs.pipeline()
config = rs.config()
width, height = 640, 480
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 15)
pipeline.start(config)

device = pipeline.get_active_profile().get_device()
color_sensor = device.query_sensors()[1]

# 수동 모드 기본 설정
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

# ============================================================
# ✅ Tkinter 기본 설정
# ============================================================
root = tk.Tk()
root.title("🎛️ RealSense RGB Control (Numeric Only)")
root.geometry("960x720")
root.minsize(800, 600)

# 상단: 영상 / 하단: 제어
top_frame = ttk.Frame(root)
top_frame.pack(fill="both", expand=True)

bottom_frame = ttk.Frame(root)
bottom_frame.pack(fill="x", padx=10, pady=5)

# ============================================================
# ✅ 영상 표시 라벨
# ============================================================
image_label = ttk.Label(top_frame)
image_label.pack(fill="both", expand=True)

# ============================================================
# ✅ 조절 가능한 옵션 리스트
# ============================================================
OPTIONS = [
    ("Exposure", rs.option.exposure, 1, 1000),
    ("Gain", rs.option.gain, 0, 128),
    ("Brightness", rs.option.brightness, -64, 64),
    ("Contrast", rs.option.contrast, 0, 100),
    ("Gamma", rs.option.gamma, 100, 500),
    ("Hue", rs.option.hue, -180, 180),
    ("Saturation", rs.option.saturation, 0, 100),
    ("Sharpness", rs.option.sharpness, 0, 100),
    ("WhiteBalance", rs.option.white_balance, 2800, 6500),
]

# ============================================================
# ✅ 숫자 입력 필드 생성
# ============================================================
entry_vars = {}

def make_entry(parent, name, option, minv, maxv):
    frame = ttk.Frame(parent)
    frame.pack(side="left", padx=10)

    ttk.Label(frame, text=name).pack()
    var = tk.DoubleVar()

    try:
        var.set(color_sensor.get_option(option))
    except Exception:
        var.set((minv + maxv) / 2)

    entry = ttk.Entry(frame, textvariable=var, width=8, justify="center")
    entry.pack()

    def apply_value(event=None):
        try:
            value = float(var.get())
            if value < minv: value = minv
            if value > maxv: value = maxv
            color_sensor.set_option(option, value)
            var.set(value)
        except ValueError:
            pass

    entry.bind("<Return>", apply_value)
    entry_vars[name] = var

for name, opt, mn, mx in OPTIONS:
    make_entry(bottom_frame, name, opt, mn, mx)

# ============================================================
# ✅ Auto 옵션 (체크박스)
# ============================================================
auto_frame = ttk.Frame(bottom_frame)
auto_frame.pack(side="left", padx=10)

auto_wb = tk.BooleanVar(value=False)
auto_exp = tk.BooleanVar(value=False)

def toggle_auto_white():
    color_sensor.set_option(rs.option.enable_auto_white_balance, float(auto_wb.get()))

def toggle_auto_exposure():
    color_sensor.set_option(rs.option.enable_auto_exposure, float(auto_exp.get()))

ttk.Checkbutton(auto_frame, text="Auto WB", variable=auto_wb, command=toggle_auto_white).pack()
ttk.Checkbutton(auto_frame, text="Auto EXP", variable=auto_exp, command=toggle_auto_exposure).pack()

# ============================================================
# ✅ 프레임 업데이트 루프 (비율 유지)
# ============================================================
def update_frame():
    global frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        root.after(10, update_frame)
        return

    frame = np.asanyarray(color_frame.get_data())
    label_w, label_h = image_label.winfo_width(), image_label.winfo_height()
    if label_w <= 1 or label_h <= 1:
        label_w, label_h = width, height

    # 비율 유지 리사이즈 (Center crop)
    aspect_src = width / height
    aspect_dst = label_w / label_h
    if aspect_dst > aspect_src:
        new_h = label_h
        new_w = int(new_h * aspect_src)
    else:
        new_w = label_w
        new_h = int(new_w / aspect_src)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # 중앙 정렬용 패딩
    canvas = np.zeros((label_h, label_w, 3), dtype=np.uint8)
    y0 = (label_h - new_h) // 2
    x0 = (label_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=image)
    image_label.imgtk = imgtk
    image_label.configure(image=imgtk)

    root.after(20, update_frame)

# ============================================================
# ✅ 클릭 시 HSV/LAB 출력
# ============================================================
def on_click(event):
    if frame is None:
        return
    label_w, label_h = image_label.winfo_width(), image_label.winfo_height()
    x = int(event.x * width / label_w)
    y = int(event.y * height / label_h)
    if 0 <= x < width and 0 <= y < height:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        h, s, v = hsv[y, x]
        l, a, b = lab[y, x]
        print(f"(x={x}, y={y}) → HSV=({h},{s},{v})  LAB=({l},{a},{b})")

image_label.bind("<Button-1>", on_click)

# ============================================================
# ✅ 종료 시 정리
# ============================================================
root.after(100, update_frame)
root.protocol("WM_DELETE_WINDOW", lambda: (pipeline.stop(), root.destroy()))
root.mainloop()
