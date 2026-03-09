#!/usr/bin/env python3
"""Render split-screen demo: dashcam video (left) + terminal output (right) → MP4"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont

# ── Config ──
CAST_FILE = Path(__file__).parent / "strands-cosmos-demo.cast"
VIDEO_FILE = Path(__file__).parent.parent / "sample.mp4"
OUTPUT_MP4 = Path(__file__).parent / "strands-cosmos-demo.mp4"

WIDTH = 1920       # Total width
HEIGHT = 1080      # Total height
LEFT_W = 896       # Video panel (slightly less than half)
RIGHT_W = WIDTH - LEFT_W  # Terminal panel
FPS = 15
BG_COLOR = (18, 18, 24)       # Dark terminal bg
BORDER_COLOR = (118, 185, 0)  # NVIDIA green
FONT_SIZE = 16
LINE_HEIGHT = 20

# ── Load cast file ──
print("Loading cast file...")
with open(CAST_FILE) as f:
    lines = f.readlines()

header = json.loads(lines[0])
events = [json.loads(line) for line in lines[1:] if line.strip()]
cast_duration = events[-1][0] if events else 10.0
total_duration = max(cast_duration + 2.0, 18.0)  # At least 18s
total_frames = int(total_duration * FPS)

print(f"Cast: {len(events)} events, {cast_duration:.1f}s")
print(f"Output: {WIDTH}x{HEIGHT} @ {FPS}fps, {total_duration:.1f}s, {total_frames} frames")

# ── Setup terminal emulator ──
term_cols = header.get("width", 80)
term_rows = header.get("height", 30)
screen = pyte.Screen(term_cols, term_rows)
stream = pyte.Stream(screen)

# ANSI color map
ANSI_COLORS = {
    "black": (0, 0, 0), "red": (204, 51, 51), "green": (80, 200, 80),
    "brown": (204, 204, 51), "blue": (51, 51, 204), "magenta": (176, 48, 176),
    "cyan": (51, 204, 204), "white": (204, 204, 204),
    "default": (204, 204, 204),
}
ANSI_BRIGHT = {
    "black": (85, 85, 85), "red": (255, 85, 85), "green": (85, 255, 85),
    "brown": (255, 255, 85), "blue": (85, 85, 255), "magenta": (255, 85, 255),
    "cyan": (85, 255, 255), "white": (255, 255, 255),
}

# ── Try to find a monospace font ──
font = None
font_paths = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
]
for fp in font_paths:
    if os.path.exists(fp):
        font = ImageFont.truetype(fp, FONT_SIZE)
        break
if font is None:
    font = ImageFont.load_default()
    print("Warning: using default font (may look rough)")


def get_color(char):
    """Get foreground color for a pyte character."""
    fg = char.fg if hasattr(char, 'fg') else "default"
    bold = char.bold if hasattr(char, 'bold') else False
    
    if isinstance(fg, str):
        if bold and fg in ANSI_BRIGHT:
            return ANSI_BRIGHT[fg]
        return ANSI_COLORS.get(fg, (204, 204, 204))
    elif isinstance(fg, int):
        # 256-color
        if fg < 8:
            colors = list(ANSI_COLORS.values())
            return colors[fg] if fg < len(colors) else (204, 204, 204)
        elif fg < 16:
            colors = list(ANSI_BRIGHT.values())
            return colors[fg - 8] if (fg - 8) < len(colors) else (204, 204, 204)
        return (204, 204, 204)
    return (204, 204, 204)


def render_terminal_frame(draw, x_offset, y_offset):
    """Render current terminal state to PIL draw context."""
    for row_idx in range(min(term_rows, 45)):
        y = y_offset + row_idx * LINE_HEIGHT
        if y + LINE_HEIGHT > HEIGHT:
            break
        for col_idx in range(term_cols):
            char = screen.buffer[row_idx][col_idx]
            ch = char.data if hasattr(char, 'data') else str(char)
            if ch and ch != ' ':
                color = get_color(char)
                x = x_offset + col_idx * (FONT_SIZE * 0.6)
                draw.text((x, y), ch, fill=color, font=font)


# ── Extract video frames ──
print("Extracting video frames...")
frame_dir = tempfile.mkdtemp(prefix="cosmos_demo_")

# Get video duration
probe = subprocess.run(
    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
     "-of", "default=noprint_wrappers=1:nokey=1", str(VIDEO_FILE)],
    capture_output=True, text=True
)
video_duration = float(probe.stdout.strip())

# Extract frames for looping
subprocess.run([
    "ffmpeg", "-y", "-i", str(VIDEO_FILE),
    "-vf", f"scale={LEFT_W - 40}:-1,fps={FPS}",
    "-q:v", "2",
    f"{frame_dir}/vframe_%05d.png"
], capture_output=True)

video_frames = sorted(Path(frame_dir).glob("vframe_*.png"))
print(f"Extracted {len(video_frames)} video frames")

# ── Render composite frames ──
print("Rendering composite frames...")
output_frame_dir = tempfile.mkdtemp(prefix="cosmos_output_")

event_idx = 0

for frame_num in range(total_frames):
    current_time = frame_num / FPS
    
    # Feed cast events up to current time
    while event_idx < len(events) and events[event_idx][0] <= current_time:
        _, _, text = events[event_idx]
        stream.feed(text)
        event_idx += 1
    
    # Create frame
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # ── Left panel: video ──
    # Draw panel background
    draw.rectangle([0, 0, LEFT_W - 1, HEIGHT - 1], fill=(10, 10, 15))
    
    # Draw border
    draw.rectangle([0, 0, LEFT_W - 2, HEIGHT - 1], outline=BORDER_COLOR, width=2)
    
    # Header
    draw.rectangle([0, 0, LEFT_W - 2, 35], fill=(25, 25, 35))
    draw.text((15, 8), "📹 Dashcam Input — sample.mp4", fill=(200, 200, 200), font=font)
    
    # Video frame (loop)
    if video_frames:
        vf_idx = int((current_time * FPS) % len(video_frames))
        vf = Image.open(video_frames[vf_idx])
        # Center video in left panel
        vx = (LEFT_W - vf.width) // 2
        vy = 40 + (HEIGHT - 40 - vf.height) // 2
        img.paste(vf, (vx, vy))
    
    # ── Right panel: terminal ──
    right_x = LEFT_W + 2
    draw.rectangle([right_x, 0, WIDTH - 1, HEIGHT - 1], fill=BG_COLOR)
    draw.rectangle([right_x, 0, WIDTH - 1, HEIGHT - 1], outline=BORDER_COLOR, width=2)
    
    # Header
    draw.rectangle([right_x, 0, WIDTH - 1, 35], fill=(25, 25, 35))
    draw.text((right_x + 15, 8), "🧠 Cosmos Agent — Chain-of-Thought", fill=(200, 200, 200), font=font)
    
    # Terminal content
    render_terminal_frame(draw, right_x + 15, 45)
    
    # ── Bottom bar ──
    bar_y = HEIGHT - 30
    draw.rectangle([0, bar_y, WIDTH, HEIGHT], fill=(25, 25, 35))
    progress = current_time / total_duration
    bar_w = int((WIDTH - 40) * progress)
    draw.rectangle([20, bar_y + 8, 20 + bar_w, bar_y + 14], fill=BORDER_COLOR)
    draw.rectangle([20, bar_y + 8, WIDTH - 20, bar_y + 14], outline=(60, 60, 70))
    
    time_str = f"{current_time:.1f}s / {total_duration:.1f}s"
    draw.text((WIDTH - 150, bar_y + 5), time_str, fill=(150, 150, 150), font=font)
    
    # Branding
    draw.text((20, bar_y + 5), "strands-cosmos • Jetson AGX Thor", fill=(118, 185, 0), font=font)
    
    # Save frame
    frame_path = f"{output_frame_dir}/frame_{frame_num:05d}.png"
    img.save(frame_path, optimize=True)
    
    if frame_num % 30 == 0:
        print(f"  Frame {frame_num}/{total_frames} ({current_time:.1f}s)")

# ── Encode MP4 ──
print("Encoding MP4...")
subprocess.run([
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", f"{output_frame_dir}/frame_%05d.png",
    "-c:v", "mpeg4",
    "-q:v", "5",
    "-pix_fmt", "yuv420p",
    str(OUTPUT_MP4)
], capture_output=True, text=True)

# Check if MP4 was created
if not OUTPUT_MP4.exists():
    # Try again without capture
    result = subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", f"{output_frame_dir}/frame_%05d.png",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(OUTPUT_MP4)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr[-500:]}")

# Cleanup
import shutil
shutil.rmtree(frame_dir)
shutil.rmtree(output_frame_dir)

size_mb = OUTPUT_MP4.stat().st_size / (1024 * 1024)
print(f"\n✅ Done! Output: {OUTPUT_MP4} ({size_mb:.1f} MB)")
