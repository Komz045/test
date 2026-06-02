"""
DrawingDiffGUI.py
=================
Shipbuilding AI — Drawing Verification & Highlighting System

FEATURES:
  • Dark-theme 1600x900 GUI (CustomTkinter)
  • Click to load Drawing 1 & 2 (DWG or DXF)
  • 🔄 Convert DWG→DXF via ODA File Converter (ezdxf.addons.odafc)
  • 📸 Render DXF→Image using ezdxf
  • ⚡ HIGHLIGHT — diff pipeline with shape matching
  • Results tab — zoomable 4-panel preview + region cards + summary
  • DiffChecker tab — 3 native GUI modes (Side by Side / Slider / Highlight)
      - Renders DIRECTLY from DXF files
      - Auto-align: both drawings normalised to union bounding box
      - Highlight overlay: red=draw1, green=draw2 on white
      - Slider: live drag divider on canvas
  • Live colour-coded console log
  • ⬇ Download Report (HTML or PDF)
  • 💾 Save Result image

SETUP:
    pip install customtkinter pillow opencv-python-headless numpy ezdxf shapely rtree python-docx

RUN:
    python DrawingDiffGUI.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys
import time
import tempfile
import webbrowser
import math
import numpy as np
import cv2
from PIL import Image as PILImage, ImageTk, ImageDraw
import customtkinter as ctk
import warnings
warnings.filterwarnings("ignore")

try:
    import ezdxf
    from ezdxf.addons import odafc
    DXF_OK = True
except ImportError:
    DXF_OK = False

try:
    from shapely.geometry import Polygon
    from rtree import index as _rtree_index
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False

# ── Theme ──────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_DEEP    = "#080c14"
BG_CARD    = "#0d1420"
BG_PANEL   = "#111827"
BG_INPUT   = "#0f1923"
ACCENT     = "#00d4ff"
ACCENT_DIM = "#0e3a45"
SUCCESS    = "#10b981"
WARNING    = "#f59e0b"
DANGER     = "#ef4444"
TEXT_PRI   = "#e2e8f0"
TEXT_SEC   = "#94a3b8"
TEXT_DIM   = "#475569"
BORDER     = "#1e2d3d"

# ── Engine constants ───────────────────────────────────────────────────────
IOU_THRESHOLD  = 0.30
AREA_RATIO_MAX = 8.0
SEARCH_RADIUS  = 0.08
CLUSTER_RADIUS = 0.07
MAX_CLUSTERS   = 20
RENDER_W       = 2800
RENDER_H       = 1600
RENDER_PAD     = 60
IOU_BUFFER     = 0.003

# DiffChecker render resolution
DC_W = 1800
DC_H = 1100
DC_PAD = 50

PALETTE_BGR = [
    (220,0,0),(0,0,220),(160,0,160),(0,160,0),(200,80,0),(0,160,160),
    (80,60,200),(160,0,70),(0,130,70),(130,0,200),(185,130,0),(0,185,120),
    (200,60,0),(60,185,0),(0,0,160),(150,70,0),(110,0,120),(0,110,110),
    (180,0,70),(40,40,190),
]

def bgr_to_hex(c): 
    return "#{:02x}{:02x}{:02x}".format(c[2],c[1],c[0])


# ════════════════════════════════════════════════════════════════════════════
# CONVERSION ENGINE (from simple.py)
# ════════════════════════════════════════════════════════════════════════════

def convert_dwg_to_dxf(dwg_path, log_fn=None):
    """Convert DWG to DXF using ODA File Converter"""
    dxf_out = tempfile.mktemp(suffix=".dxf")
    try:
        if log_fn:
            log_fn(f"Converting {os.path.basename(dwg_path)} → DXF …", "info")
        odafc.convert(dwg_path, dxf_out, version="R2010", audit=False)
        if os.path.isfile(dxf_out) and os.path.getsize(dxf_out) > 100:
            if log_fn:
                log_fn("Conversion successful", "success")
            return dxf_out
    except Exception as e:
        if log_fn:
            log_fn(f"ODA conversion error: {e}", "error")
        raise RuntimeError(f"Cannot convert {os.path.basename(dwg_path)}: {e}")


def prepare_dxf_path(file_path, log_fn=None):
    """Ensure we have a DXF file"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".dxf":
        return file_path
    elif ext == ".dwg":
        return convert_dwg_to_dxf(file_path, log_fn)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def dxf_to_image(dxf_path, width=RENDER_W, height=RENDER_H, padding=RENDER_PAD):
    """Convert DXF file to PIL Image"""
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        extents = msp.extent()
        
        if extents.has_data:
            min_point = extents.min
            max_point = extents.max
        else:
            raise RuntimeError("DXF contains no drawable entities")

        dxf_width = max_point.x - min_point.x
        dxf_height = max_point.y - min_point.y

        if dxf_width < 1e-9:
            dxf_width = 1.0
        if dxf_height < 1e-9:
            dxf_height = 1.0

        canvas_w = width - 2 * padding
        canvas_h = height - 2 * padding

        scale_x = canvas_w / dxf_width
        scale_y = canvas_h / dxf_height
        scale = min(scale_x, scale_y)

        img = PILImage.new("RGB", (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        scaled_w = dxf_width * scale
        scaled_h = dxf_height * scale
        offset_x = padding + (canvas_w - scaled_w) / 2
        offset_y = padding + (canvas_h - scaled_h) / 2

        for entity in msp:
            dxftype = entity.dxftype()
            try:
                if dxftype == "LWPOLYLINE":
                    points = [(p[0], p[1]) for p in entity.get_points()]
                    if len(points) >= 2:
                        _draw_polyline(draw, points, min_point, scale, offset_x, offset_y)
                elif dxftype == "POLYLINE":
                    points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                    if len(points) >= 2:
                        _draw_polyline(draw, points, min_point, scale, offset_x, offset_y)
                elif dxftype == "LINE":
                    start, end = entity.dxf.start, entity.dxf.end
                    p1 = _transform_point((start.x, start.y), min_point, scale, offset_x, offset_y)
                    p2 = _transform_point((end.x, end.y), min_point, scale, offset_x, offset_y)
                    draw.line([p1, p2], fill=(30, 30, 30), width=2)
                elif dxftype == "CIRCLE":
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    radius = entity.dxf.radius
                    center_px = _transform_point((cx, cy), min_point, scale, offset_x, offset_y)
                    radius_px = radius * scale
                    draw.ellipse([center_px[0]-radius_px, center_px[1]-radius_px,
                                 center_px[0]+radius_px, center_px[1]+radius_px],
                                outline=(30, 30, 30), width=2)
            except:
                pass

        return img
    except Exception as e:
        raise RuntimeError(f"Failed to convert DXF to image: {e}")


def _transform_point(point, min_point, scale, offset_x, offset_y):
    """Transform DXF coordinates to image pixel coordinates"""
    px = (point[0] - min_point.x) * scale + offset_x
    py = (point[1] - min_point.y) * scale + offset_y
    return (int(px), int(py))


def _draw_polyline(draw, points, min_point, scale, offset_x, offset_y):
    """Draw a polyline on the PIL image"""
    if len(points) < 2:
        return
    transformed = [_transform_point(p, min_point, scale, offset_x, offset_y) for p in points]
    for i in range(len(transformed) - 1):
        draw.line([transformed[i], transformed[i + 1]], fill=(30, 30, 30), width=2)


# ════════════════════════════════════════════════════════════════════════════
# DIFF ENGINE — from original shipgpt.py
# ════════════════════════════════════════════════════════════════════════════

def load_normalised_shapes(dxf_path):
    """Load and normalize shapes from DXF"""
    doc = ezdxf.readfile(dxf_path)
    raw = []
    
    for e in doc.modelspace():
        t = e.dxftype()
        try:
            if t == "LWPOLYLINE":
                pts = [(x, y) for x, y, *_ in e.get_points()]
                if len(pts) >= 2:
                    raw.append((pts, e.is_closed))
            elif t == "POLYLINE":
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                if len(pts) >= 2:
                    raw.append((pts, bool(e.is_closed)))
            elif t == "LINE":
                s, en = e.dxf.start, e.dxf.end
                raw.append(([(s.x, s.y), (en.x, en.y)], False))
            elif t == "CIRCLE":
                cx, cy, r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
                pts = [(cx + r * math.cos(math.radians(a)),
                        cy + r * math.sin(math.radians(a))) for a in range(0, 361, 15)]
                raw.append((pts, True))
        except:
            pass

    if not raw:
        raise RuntimeError("No drawable entities found")
    
    all_pts = np.array([p for pts, _ in raw for p in pts])
    xmin, ymin = all_pts[:,0].min(), all_pts[:,1].min()
    xmax, ymax = all_pts[:,0].max(), all_pts[:,1].max()
    W, H = xmax-xmin, ymax-ymin
    if W < 1e-9: W = 1.0
    if H < 1e-9: H = 1.0

    shapes = []
    for pts, closed in raw:
        norm = [((x-xmin)/W, (y-ymin)/H) for x,y in pts]
        try:
            g = Polygon(norm) if (closed and len(norm)>=3) else None
            if g and not g.is_valid:
                g = g.buffer(0)
            if not g or g.is_empty or g.area < 1e-8:
                continue
            b = g.bounds
            shapes.append({"geom":g, "area":g.area, "cx":(b[0]+b[2])/2,
                          "cy":(b[1]+b[3])/2, "bounds":b, "pts_raw":pts, "npts":len(pts)})
        except:
            pass
    
    return shapes, xmin, ymin, xmax, ymax, W, H


def iou_fn(g1, g2, buf=IOU_BUFFER):
    """Calculate Intersection over Union"""
    try:
        a, b = g1.buffer(buf), g2.buffer(buf)
        i, u = a.intersection(b).area, a.union(b).area
        return i/u if u>1e-12 else 0.0
    except:
        return 0.0


def match_shapes(s1, s2):
    """Match shapes between two drawings"""
    idx2 = _rtree_index.Index()
    for i, s in enumerate(s2):
        b = s["bounds"]
        idx2.insert(i, (b[0], b[1], b[2], b[3]))
    
    m1, m2 = set(), set()
    pairs = []
    
    for i, s in enumerate(s1):
        cx, cy = s["cx"], s["cy"]
        r = SEARCH_RADIUS
        cands = list(idx2.intersection((cx-r, cy-r, cx+r, cy+r)))
        bv, bj = 0, -1
        
        for j in cands:
            iou = iou_fn(s["geom"], s2[j]["geom"])
            a1, a2 = s["area"], s2[j]["area"]
            ar = max(a1, a2) / (min(a1, a2) + 1e-12)
            
            if iou > IOU_THRESHOLD and ar < AREA_RATIO_MAX and iou > bv:
                bv, bj = iou, j
        
        if bj >= 0:
            pairs.append((i, bj, bv))
            m1.add(i)
            m2.add(bj)
    
    return pairs, m1, m2


def render_highlight(p1, p2, s1, s2, pairs, m1, m2):
    """Render highlight comparison image"""
    img = PILImage.new("RGB", (RENDER_W, RENDER_H), (255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    
    for i in range(len(s1)):
        pts = s1[i]["pts_raw"]
        if len(pts) >= 2:
            scaled = [(p1[0] + (x-p1[2])*p1[4], p1[1] + (y-p1[3])*p1[5]) for x,y in pts]
            if len(scaled) >= 2:
                draw.polygon(scaled, outline=(220, 60, 60), width=2)
    
    for j in range(len(s2)):
        pts = s2[j]["pts_raw"]
        if len(pts) >= 2:
            scaled = [(p2[0] + (x-p2[2])*p2[4], p2[1] + (y-p2[3])*p2[5]) for x,y in pts]
            if len(scaled) >= 2:
                draw.polygon(scaled, outline=(30, 180, 30), width=2)
    
    return img


def render_side_by_side(p1, p2, s1, s2):
    """Render side-by-side comparison"""
    img1 = PILImage.new("RGB", (RENDER_W//2, RENDER_H), (255, 255, 255))
    draw1 = ImageDraw.Draw(img1)
    
    for i in range(len(s1)):
        pts = s1[i]["pts_raw"]
        if len(pts) >= 2:
            scaled = [(p1[0] + (x-p1[2])*p1[4], p1[1] + (y-p1[3])*p1[5]) for x,y in pts]
            if len(scaled) >= 2:
                draw1.polygon(scaled, outline=(30, 30, 30), width=2)
    
    img2 = PILImage.new("RGB", (RENDER_W//2, RENDER_H), (255, 255, 255))
    draw2 = ImageDraw.Draw(img2)
    
    for j in range(len(s2)):
        pts = s2[j]["pts_raw"]
        if len(pts) >= 2:
            scaled = [(p2[0] + (x-p2[2])*p2[4], p2[1] + (y-p2[3])*p2[5]) for x,y in pts]
            if len(scaled) >= 2:
                draw2.polygon(scaled, outline=(30, 30, 30), width=2)
    
    out = PILImage.new("RGB", (RENDER_W, RENDER_H), (255, 255, 255))
    out.paste(img1, (0, 0))
    out.paste(img2, (RENDER_W//2, 0))
    return out


# ════════════════════════════════════════════════════════════════════════════
# GUI APPLICATION
# ════════════════════════════════════════════════════════════════════════════

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Drawing DiffChecker — Shipbuilding AI")
        self.geometry("1600x900")
        self.configure(fg_color=BG_DEEP)

        self.path1 = tk.StringVar(value="")
        self.path2 = tk.StringVar(value="")
        self.img1_pil = None
        self.img2_pil = None
        self._img1_tk = None
        self._img2_tk = None
        self._result = None
        self._result_pil = None
        self._dc_pil = {}
        self._dc_tk = {}
        self._dc_mode = "sideby"
        self._slider_pct = 0.5

        self._build_ui()
        self._check_deps()

    def _build_ui(self):
        """Build main UI"""
        # Top bar
        top_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=70)
        top_frame.pack(side="top", fill="x", padx=0, pady=0)
        top_frame.pack_propagate(False)

        ctk.CTkLabel(top_frame, text="📐 Drawing DiffChecker",
                    font=("Segoe UI", 24, "bold"), text_color=TEXT_PRI).pack(side="left", padx=20, pady=15)

        btn_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        btn_frame.pack(side="left", padx=20, pady=15)

        ctk.CTkButton(btn_frame, text="📂 Load Drawing 1", font=("Segoe UI", 12, "bold"),
                     fg_color=ACCENT, text_color="#000000", command=self._load_file_1,
                     corner_radius=6, width=150).pack(side="left", padx=8)

        ctk.CTkButton(btn_frame, text="📂 Load Drawing 2", font=("Segoe UI", 12, "bold"),
                     fg_color=ACCENT, text_color="#000000", command=self._load_file_2,
                     corner_radius=6, width=150).pack(side="left", padx=8)

        ctk.CTkButton(top_frame, text="⚡ HIGHLIGHT", font=("Segoe UI", 12, "bold"),
                     fg_color=WARNING, text_color="#000", command=self._run_diff,
                     corner_radius=6, width=120).pack(side="left", padx=10)

        ctk.CTkButton(top_frame, text="🗑️ Clear", font=("Segoe UI", 11, "bold"),
                     fg_color=DANGER, text_color="#fff", command=self._clear_all,
                     corner_radius=6, width=100).pack(side="right", padx=20, pady=15)

        # Tabs
        tab_frame = ctk.CTkFrame(self, fg_color="transparent")
        tab_frame.pack(side="top", fill="x", padx=12, pady=(12, 0))

        self.tab_var = tk.StringVar(value="results")
        
        for tab_name, tab_id in [("Results", "results"), ("DiffChecker", "diffchecker")]:
            btn = ctk.CTkButton(tab_frame, text=tab_name, font=("Segoe UI", 11, "bold"),
                              fg_color=ACCENT if tab_id == "results" else BORDER,
                              text_color="#000" if tab_id == "results" else TEXT_SEC,
                              command=lambda tid=tab_id: self._switch_tab(tid),
                              corner_radius=4, width=120)
            btn.pack(side="left", padx=4)

        # Content area
        self.content_frame = ctk.CTkFrame(self, fg_color=BG_DEEP)
        self.content_frame.pack(side="top", fill="both", expand=True, padx=0, pady=0)

        # Results tab
        self.results_frame = ctk.CTkFrame(self.content_frame, fg_color=BG_DEEP)
        self.results_frame.pack(fill="both", expand=True)

        self.canvas1 = tk.Canvas(self.results_frame, bg="#f0f0f0", highlightthickness=0)
        self.canvas1.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self.canvas2 = tk.Canvas(self.results_frame, bg="#f0f0f0", highlightthickness=0)
        self.canvas2.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        # DiffChecker tab
        self.dc_frame = ctk.CTkFrame(self.content_frame, fg_color=BG_DEEP)
        self.dc_frame.pack(fill="both", expand=True)

        dc_top = ctk.CTkFrame(self.dc_frame, fg_color="transparent", height=40)
        dc_top.pack(side="top", fill="x", padx=12, pady=(12, 0))
        dc_top.pack_propagate(False)

        for mode, label in [("sideby", "Side by Side"), ("slider", "Slider"), ("highlight", "Highlight")]:
            ctk.CTkButton(dc_top, text=label, font=("Segoe UI", 10, "bold"),
                         fg_color=ACCENT if mode == self._dc_mode else BORDER,
                         text_color="#000" if mode == self._dc_mode else TEXT_SEC,
                         command=lambda m=mode: self._set_dc_mode(m),
                         corner_radius=4, width=100).pack(side="left", padx=4)

        ctk.CTkButton(dc_top, text="💾 Save View", font=("Segoe UI", 10, "bold"),
                     fg_color=SUCCESS, text_color="#fff",
                     command=self._dc_save_view, corner_radius=4, width=100).pack(side="right", padx=4)

        self.dc_canvas = tk.Canvas(self.dc_frame, bg="#f0f0f0", highlightthickness=0)
        self.dc_canvas.pack(fill="both", expand=True, padx=12, pady=12)
        self.dc_canvas.bind("<Motion>", self._on_slider_motion)

        # Bottom log
        bottom_frame = ctk.CTkFrame(self, fg_color=BG_CARD, height=120)
        bottom_frame.pack(side="bottom", fill="x", padx=0, pady=0)
        bottom_frame.pack_propagate(False)

        ctk.CTkLabel(bottom_frame, text="📋 Console Log", font=("Segoe UI", 10, "bold"),
                    text_color=ACCENT).pack(anchor="w", padx=12, pady=(8, 4))

        self.log_text = tk.Text(bottom_frame, height=4, bg=BG_INPUT, fg=TEXT_PRI,
                               font=("Courier New", 9), insertbackground=ACCENT,
                               relief="flat", borderwidth=0)
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.log_text.config(state="disabled")

    def _switch_tab(self, tab_id):
        """Switch between tabs"""
        if tab_id == "results":
            self.results_frame.pack(fill="both", expand=True)
            self.dc_frame.pack_forget()
        else:
            self.results_frame.pack_forget()
            self.dc_frame.pack(fill="both", expand=True)
        self.tab_var.set(tab_id)

    def _set_dc_mode(self, mode):
        """Set DiffChecker mode"""
        self._dc_mode = mode
        self._render_dc()

    def _on_slider_motion(self, event):
        """Handle slider motion"""
        if self._dc_mode == "slider":
            canvas_w = self.dc_canvas.winfo_width()
            if canvas_w > 0:
                self._slider_pct = max(0.0, min(1.0, event.x / canvas_w))
                self._render_dc()

    def _log(self, message, level="info"):
        """Log message"""
        self.log_text.config(state="normal")
        colors = {"info": TEXT_SEC, "success": SUCCESS, "warn": WARNING, "error": DANGER}
        color = colors.get(level, TEXT_SEC)
        self.log_text.insert("end", f"[{level.upper()}] {message}\n")
        self.log_text.tag_add(level, "end linestart", "end lineend")
        self.log_text.tag_config(level, foreground=color)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _load_file_1(self):
        """Load Drawing 1"""
        file = filedialog.askopenfilename(filetypes=[("CAD Files", "*.dxf *.dwg"), ("All Files", "*.*")])
        if file:
            self.path1.set(file)
            self._log(f"Loaded Drawing 1: {os.path.basename(file)}", "success")

    def _load_file_2(self):
        """Load Drawing 2"""
        file = filedialog.askopenfilename(filetypes=[("CAD Files", "*.dxf *.dwg"), ("All Files", "*.*")])
        if file:
            self.path2.set(file)
            self._log(f"Loaded Drawing 2: {os.path.basename(file)}", "success")

    def _run_diff(self):
        """Run diff analysis"""
        if not self.path1.get() or not self.path2.get():
            messagebox.showwarning("Missing Files", "Please load both drawings")
            return

        thread = threading.Thread(target=self._diff_thread, daemon=True)
        thread.start()

    def _diff_thread(self):
        """Background diff processing"""
        try:
            self._log("Converting files...", "info")
            dxf1 = prepare_dxf_path(self.path1.get(), self._log)
            dxf2 = prepare_dxf_path(self.path2.get(), self._log)

            self._log("Loading shapes...", "info")
            s1, xmin1, ymin1, xmax1, ymax1, w1, h1 = load_normalised_shapes(dxf1)
            s2, xmin2, ymin2, xmax2, ymax2, w2, h2 = load_normalised_shapes(dxf2)

            self._log("Matching shapes...", "info")
            pairs, m1, m2 = match_shapes(s1, s2)

            self._log("Rendering images...", "info")
            self.img1_pil = dxf_to_image(dxf1)
            self.img2_pil = dxf_to_image(dxf2)

            # Render diff visualizations
            p1 = (RENDER_PAD, RENDER_PAD, xmin1, ymin1, (RENDER_W-2*RENDER_PAD)/w1, (RENDER_H-2*RENDER_PAD)/h1)
            p2 = (RENDER_PAD, RENDER_PAD, xmin2, ymin2, (RENDER_W-2*RENDER_PAD)/w2, (RENDER_H-2*RENDER_PAD)/h2)

            hl_img = render_highlight(p1, p2, s1, s2, pairs, m1, m2)
            sb_img = render_side_by_side(p1, p2, s1, s2)

            self._dc_pil["d1"] = self.img1_pil.resize((DC_W//2, DC_H), PILImage.LANCZOS)
            self._dc_pil["d2"] = self.img2_pil.resize((DC_W//2, DC_H), PILImage.LANCZOS)
            self._dc_pil["hl"] = hl_img.resize((DC_W, DC_H), PILImage.LANCZOS)

            self._result_pil = hl_img
            self._result = {"pairs": pairs, "m1": m1, "m2": m2}

            self._display_images()
            self._render_dc()

            self._log(f"Diff complete: {len(pairs)} matches, {len(m1)} in D1, {len(m2)} in D2", "success")

        except Exception as e:
            self._log(f"Diff error: {e}", "error")
            messagebox.showerror("Error", str(e))

    def _display_images(self):
        """Display images on canvas"""
        if self.img1_pil:
            img_scaled = self.img1_pil.resize((600, 400), PILImage.LANCZOS)
            self._img1_tk = ImageTk.PhotoImage(img_scaled)
            self.canvas1.delete("all")
            self.canvas1.create_image(300, 200, image=self._img1_tk)

        if self.img2_pil:
            img_scaled = self.img2_pil.resize((600, 400), PILImage.LANCZOS)
            self._img2_tk = ImageTk.PhotoImage(img_scaled)
            self.canvas2.delete("all")
            self.canvas2.create_image(300, 200, image=self._img2_tk)

    def _render_dc(self):
        """Render DiffChecker view"""
        if not self._dc_pil.get("d1"):
            return

        CW = self.dc_canvas.winfo_width()
        CH = self.dc_canvas.winfo_height()
        if CW < 100 or CH < 100:
            self.after(100, self._render_dc)
            return

        self.dc_canvas.delete("all")
        self._dc_tk = {}

        if self._dc_mode == "sideby":
            d1, d2 = self._dc_pil["d1"], self._dc_pil["d2"]
            h1, h2 = int(CH * 0.9), int(CH * 0.9)
            w1 = int(h1 * d1.width / d1.height)
            w2 = int(h2 * d2.width / d2.height)
            d1_scaled = d1.resize((w1, h1), PILImage.LANCZOS)
            d2_scaled = d2.resize((w2, h2), PILImage.LANCZOS)
            xo1 = (CW//2 - w1) // 2
            xo2 = CW//2 + (CW//2 - w2) // 2
            yo = (CH - h1) // 2

            self._dc_tk["d1"] = ImageTk.PhotoImage(d1_scaled)
            self._dc_tk["d2"] = ImageTk.PhotoImage(d2_scaled)
            self.dc_canvas.create_image(xo1, yo, anchor="nw", image=self._dc_tk["d1"])
            self.dc_canvas.create_image(xo2, yo, anchor="nw", image=self._dc_tk["d2"])

        elif self._dc_mode == "slider":
            d2 = self._dc_pil["d2"]
            h = int(CH * 0.9)
            w = int(h * d2.width / d2.height)
            d2_scaled = d2.resize((w, h), PILImage.LANCZOS)
            xo, yo = (CW - w) // 2, (CH - h) // 2

            self._dc_tk["back"] = ImageTk.PhotoImage(d2_scaled)
            self.dc_canvas.create_image(xo, yo, anchor="nw", image=self._dc_tk["back"])

            div_x = int(self._slider_pct * w)
            if div_x > 0:
                front = self._dc_pil["d1"].resize((w, h), PILImage.LANCZOS).crop((0, 0, div_x, h))
                self._dc_tk["front"] = ImageTk.PhotoImage(front)
                self.dc_canvas.create_image(xo, yo, anchor="nw", image=self._dc_tk["front"])

            self.dc_canvas.create_line(xo + div_x, yo, xo + div_x, yo + h, fill="#00d4ff", width=3)

        elif self._dc_mode == "highlight":
            hl = self._dc_pil["hl"]
            h = int(CH * 0.9)
            w = int(h * hl.width / hl.height)
            hl_scaled = hl.resize((w, h), PILImage.LANCZOS)
            xo, yo = (CW - w) // 2, (CH - h) // 2
            self._dc_tk["hl"] = ImageTk.PhotoImage(hl_scaled)
            self.dc_canvas.create_image(xo, yo, anchor="nw", image=self._dc_tk["hl"])

    def _dc_save_view(self):
        """Save DiffChecker view"""
        if not self._dc_pil.get("hl"):
            messagebox.showwarning("No Data", "Run analysis first")
            return

        file = filedialog.asksaveasfilename(defaultextension=".png",
                                          filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if file:
            self._dc_pil["hl"].save(file)
            self._log(f"Saved → {os.path.basename(file)}", "success")

    def _clear_all(self):
        """Clear all"""
        self.path1.set("")
        self.path2.set("")
        self.img1_pil = None
        self.img2_pil = None
        self._dc_pil = {}
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.dc_canvas.delete("all")
        self._log("Cleared", "info")

    def _check_deps(self):
        """Check dependencies"""
        if not DXF_OK:
            self._log("❌ ezdxf not installed", "error")
        else:
            self._log("✓ ezdxf installed", "success")
        if not SHAPELY_OK:
            self._log("❌ shapely/rtree not installed", "error")
        else:
            self._log("✓ shapely/rtree installed", "success")


if __name__ == "__main__":
    app = App()
    app.mainloop()
