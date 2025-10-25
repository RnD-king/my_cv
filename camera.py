#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ RealSense Viewer (Fixed 840x780, Pause with Space)
 - 4:3 비율 유지 (640x480)
 - 숫자 입력으로 파라미터 제어
 - Spacebar로 영상 일시정지 / 재시작
 - 클릭 시 HSV/LAB 출력
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
cam_w, cam_h = 640, 480
config.enable_stream(rs.stream.color, cam_w, cam_h, rs.format.bgr8, 15)
pipeline.start(config)

device = pipeline.get_active_profile().get_device()
color_sensor = device.query_sensors()[1]

# 수동 모드 기본 설정
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

# ============================================================
# ✅ Tkinter 창 설정
# ============================================================
root = tk.Tk()
root.title("🎛️ RealSense RGB Control (Fixed 840x780)")
root.geometry("840x780")
root.resizable(False, False)
root.attributes('-fullscreen', False)

# ============================================================
# ✅ 프레임 구성
# ============================================================
frame_video = ttk.Frame(root, width=840, height=630)
frame_video.pack_propagate(False)
frame_video.pack()

frame_params = ttk.Frame(root, height=90)
frame_params.pack(fill="x", pady=5)

frame_status = ttk.Frame(root, height=30)
frame_status.pack(fill="x", pady=(0, 5))

# ============================================================
# ✅ 영상 라벨
# ============================================================
image_label = ttk.Label(frame_video)
image_label.place(relx=0.5, rely=0.5, anchor="center")

# ============================================================
# ✅ 제어 가능한 옵션 리스트
# ============================================================
OPTIONS = [
    ("Exposure", rs.option.exposure, 1, 1000),
    ("Gain", rs.option.gain, 0, 128),
    ("Brightness", rs.option.brightness, -64, 64),
    ("Contrast", rs.option.contrast, 0, 100),
    ("Gamma", rs.option.gamma, 100, 500),
    ("Saturation", rs.option.saturation, 0, 100),
    ("Sharpness", rs.option.sharpness, 0, 100),
    ("WhiteBalance", rs.option.white_balance, 2800, 6500),
]

entry_vars = {}

def make_entry(parent, name, option, minv, maxv):
    """숫자 입력 전용"""
    frame = ttk.Frame(parent)
    frame.pack(side="left", padx=6)

    ttk.Label(frame, text=name, font=("Arial", 9)).pack()
    var = tk.DoubleVar()

    try:
        var.set(color_sensor.get_option(option))
    except Exception:
        var.set((minv + maxv) / 2)

    entry = ttk.Entry(frame, textvariable=var, width=8, justify="center")
    entry.pack(pady=2)

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
    make_entry(frame_params, name, opt, mn, mx)

# ============================================================
# ✅ 상태 표시 라벨 (Paused / Playing)
# ============================================================
status_label = ttk.Label(frame_status, text="Status: Playing", font=("Arial", 11, "bold"))
status_label.pack(anchor="center")

# ============================================================
# ✅ Space 키로 일시정지 / 재시작
# ============================================================
paused = False

def toggle_pause(event=None):
    global paused
    paused = not paused
    if paused:
        status_label.config(text="Status: Paused")
    else:
        status_label.config(text="Status: Playing")

root.bind("<space>", toggle_pause)

# ============================================================
# ✅ 영상 업데이트 루프 (4:3 비율 유지)
# ============================================================
frame = None

def update_frame():
    global frame
    if not paused:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            frame = np.asanyarray(color_frame.get_data())

    if frame is not None:
        # 4:3 비율 유지 (840x630 영역)
        target_w, target_h = 840, 630
        aspect_src = cam_w / cam_h
        aspect_dst = target_w / target_h

        if aspect_dst > aspect_src:
            new_h = target_h
            new_w = int(new_h * aspect_src)
        else:
            new_w = target_w
            new_h = int(new_w / aspect_src)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

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
    x = int(event.x * cam_w / 840)
    y = int(event.y * cam_h / 630)
    if 0 <= x < cam_w and 0 <= y < cam_h:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        h, s, v = hsv[y, x]
        l, a, b = lab[y, x]
        print(f"(x={x}, y={y}) → HSV=({h},{s},{v})  LAB=({l},{a},{b})")

image_label.bind("<Button-1>", on_click)

# ============================================================
# ✅ 실행 및 종료
# ============================================================
root.after(100, update_frame)
root.protocol("WM_DELETE_WINDOW", lambda: (pipeline.stop(), root.destroy()))
root.mainloop()
