"""
Gospel Lyric Video Generator
"Jesus Comin' By and By"
Produces a 3-minute (180-second) 1920x1080 video.
"""

from moviepy import (
    ColorClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip
)
from moviepy.video.fx import CrossFadeIn, CrossFadeOut
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os, glob

# Resolve font paths from the system TTF directory
def _find_font(name_fragment):
    hits = glob.glob(f'/usr/share/fonts/**/*{name_fragment}*', recursive=True)
    ttf  = [h for h in hits if h.lower().endswith(('.ttf', '.otf'))]
    return ttf[0] if ttf else None

FONT_BOLD   = _find_font('DejaVuSans-Bold') or _find_font('FreeSansBold') or 'DejaVu-Sans-Bold'
FONT_NORMAL = _find_font('DejaVuSans.ttf')  or _find_font('FreeSans.ttf') or 'DejaVu-Sans'

# ---------------------------------------------------------------------------
# Lyrics with timestamps (seconds) — total target: 180 s
# Each entry: (start, end, text, section)
# ---------------------------------------------------------------------------
SECTIONS = [
    # Intro hold
    (0, 6, "", "intro"),

    # Verse 1
    (6,  12, "Last I saw that star shine yonder in the sky,", "verse"),
    (12, 18, "Burnin' bright like a promise that will never die,", "verse"),
    (18, 24, "Pointed straight to a Savior born for you and I,", "verse"),
    (24, 32, "Oh I felt His Spirit whisper,\n\"He's comin' by and by.\"", "verse"),

    # Verse 2
    (32, 38, "Through the trials and the valleys, through the tears I've cried,", "verse"),
    (38, 44, "There's a hope that keeps me steady deep inside,", "verse"),
    (44, 50, "Ain't no grave gonna hold me, ain't no shadow gonna hide,", "verse"),
    (50, 58, "When I hear that trumpet sound,\nHe's comin' by and by.", "verse"),

    # Chorus 1
    (58, 63,  "Hallelujah!\nHe's comin' for the ready,", "chorus"),
    (63, 67,  "Lift your voice and testify!", "chorus"),
    (67, 72,  "Saints are shoutin', hearts are steady,", "chorus"),
    (72, 78,  "Jesus Christ is comin' by and by!", "chorus"),
    (78, 83,  "Oh the heavens will open wide,", "chorus"),
    (83, 88,  "And we'll meet Him in the sky,", "chorus"),
    (88, 96,  "Hallelujah! Hallelujah!\nJesus comin' by and by!", "chorus"),

    # Verse 3
    (96,  102, "I can see that eastern sky begin to glow,", "verse"),
    (102, 108, "Feel the fire of revival in my soul,", "verse"),
    (108, 114, "Every burden gonna vanish, every knee will bow,", "verse"),
    (114, 122, "And the King of all creation's\ncomin' for me now.", "verse"),

    # Bridge
    (122, 128, "Oh can't you hear the angels singin'?", "bridge"),
    (128, 134, "Oh can't you feel redemption ringin'?", "bridge"),
    (134, 142, "Every chain is breakin',\nevery heart awakenin',", "bridge"),
    (142, 150, "Glory to the Lamb!", "bridge"),

    # Chorus 2
    (150, 155, "Hallelujah!\nHe's comin' for the ready,", "chorus"),
    (155, 159, "Lift your voice and testify!", "chorus"),
    (159, 164, "Saints are shoutin', hearts are steady,", "chorus"),
    (164, 170, "Jesus Christ is comin' by and by!", "chorus"),
    (170, 175, "Oh the heavens will open wide,", "chorus"),
    (175, 179, "And we'll meet Him in the sky,", "chorus"),
    (179, 184, "Hallelujah! Hallelujah!\nJesus comin' by and by!", "chorus"),

    # Outro
    (184, 188, "By and by…  (He's comin')", "outro"),
    (188, 192, "By and by…  (Oh yes He is)", "outro"),
    (192, 198, "Hallelujah, hallelujah,", "outro"),
    (198, 207, "Jesus comin' by and by!", "outro"),

    # Fade-out hold
    (207, 213, "", "intro"),
]

TOTAL_DURATION = 213  # seconds (≈3:33 with lead-in/out; clamp to 180 below)

# Clamp to exactly 180 s
TARGET = 180

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
# Deep midnight blue → gold gradient sky feel
BG_TOP    = (10, 18, 60)      # deep navy
BG_BOTTOM = (40, 10, 80)      # deep purple

SECTION_COLORS = {
    "verse":  (255, 240, 200),   # warm parchment
    "chorus": (255, 220, 50),    # gold
    "bridge": (200, 230, 255),   # cool light blue
    "outro":  (255, 200, 120),   # soft amber
    "intro":  (255, 255, 255),
}

GLOW_COLORS = {
    "verse":  (180, 120, 40),
    "chorus": (200, 160, 0),
    "bridge": (80, 140, 220),
    "outro":  (200, 120, 40),
    "intro":  (100, 100, 100),
}

W, H = 1920, 1080
FPS = 24


# ---------------------------------------------------------------------------
# Helpers: background frames
# ---------------------------------------------------------------------------

def make_gradient_bg(t, total):
    phase = t / total
    top   = np.clip(np.array(BG_TOP,    float) + phase * np.array([5, 15, 30]),  0, 255)
    bot   = np.clip(np.array(BG_BOTTOM, float) + phase * np.array([20, 0, 10]), 0, 255)
    rows  = np.linspace(0, 1, H)[:, np.newaxis, np.newaxis]
    arr   = (top * (1 - rows) + bot * rows).astype(np.uint8)  # (H,1,3)
    return np.broadcast_to(arr, (H, W, 3)).copy()


def make_star_overlay():
    """Static star field as RGBA PIL image."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(42)
    n_stars = 350
    xs = rng.integers(0, W, n_stars)
    ys = rng.integers(0, H // 2, n_stars)  # stars only in top half
    radii = rng.choice([1, 1, 1, 2, 2, 3], size=n_stars)
    alphas = rng.integers(120, 255, n_stars)
    for x, y, r, a in zip(xs, ys, radii, alphas):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 240, int(a)))
    return img


def make_cross_overlay():
    """Subtle glowing cross centred on right side as RGBA PIL image."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = int(W * 0.80), int(H * 0.42)
    arm_w, arm_h, cross_t = 22, 110, 22
    # Vertical bar
    draw.rectangle([cx - cross_t // 2, cy - arm_h,
                    cx + cross_t // 2, cy + arm_h], fill=(255, 230, 120, 60))
    # Horizontal bar
    draw.rectangle([cx - arm_w, cy - cross_t // 2,
                    cx + arm_w, cy + cross_t // 2], fill=(255, 230, 120, 60))
    # Gaussian glow
    img = img.filter(ImageFilter.GaussianBlur(radius=18))
    return img


# Pre-render static overlays
_star_np = np.array(make_star_overlay())
_cross_np = np.array(make_cross_overlay())


def blend_rgba(base, overlay_rgba):
    """Alpha-composite overlay_rgba (H×W×4) onto base (H×W×3)."""
    alpha = overlay_rgba[:, :, 3:4].astype(float) / 255.0
    rgb   = overlay_rgba[:, :, :3].astype(float)
    return np.clip(base.astype(float) * (1 - alpha) + rgb * alpha, 0, 255).astype(np.uint8)


def make_bg_frame(t):
    frame = make_gradient_bg(t, TARGET)
    frame = blend_rgba(frame, _star_np)
    frame = blend_rgba(frame, _cross_np)
    return frame


# ---------------------------------------------------------------------------
# Build background clip (animated)
# ---------------------------------------------------------------------------

print("Building background…")
bg_clip = ColorClip(size=(W, H), color=BG_TOP, duration=TARGET)
bg_clip = bg_clip.with_fps(FPS)

# We'll bake the gradient as a sequence of frames via ImageClip composition
# For simplicity use a short set of key-frame images + CrossFadeIn

def gradient_frame(t):
    return make_bg_frame(t)

from moviepy import VideoClip
bg_clip = VideoClip(gradient_frame, duration=TARGET).with_fps(FPS)


# ---------------------------------------------------------------------------
# Lyric clips
# ---------------------------------------------------------------------------

def make_lyric_clip(text, start, end, section):
    duration = end - start
    if duration <= 0 or not text.strip():
        return None

    color   = SECTION_COLORS.get(section, (255, 255, 255))
    hex_col = "#{:02X}{:02X}{:02X}".format(*color)

    # Font size depends on line length
    max_len = max(len(line) for line in text.split("\n")) if text else 1
    if max_len <= 30:
        fontsize = 72
    elif max_len <= 45:
        fontsize = 60
    else:
        fontsize = 52

    txt = TextClip(
        text=text,
        font_size=fontsize,
        color=hex_col,
        font=FONT_BOLD,
        text_align="center",
        method="caption",
        size=(int(W * 0.85), None),
        stroke_color="#000000",
        stroke_width=3,
    )

    # Fade in/out (duration must be set before effects)
    fade = min(0.6, duration * 0.2)
    txt = txt.with_duration(duration)
    txt = txt.with_effects([CrossFadeIn(fade), CrossFadeOut(fade)])
    txt = txt.with_start(start)

    # Position: bottom third for verses, centre for chorus/bridge
    if section == "chorus":
        txt = txt.with_position(("center", int(H * 0.38)))
    elif section == "bridge":
        txt = txt.with_position(("center", int(H * 0.42)))
    elif section == "outro":
        txt = txt.with_position(("center", int(H * 0.50)))
    else:
        txt = txt.with_position(("center", int(H * 0.58)))

    return txt


print("Building lyric clips…")
lyric_clips = []
for (start, end, text, section) in SECTIONS:
    # Clamp to TARGET
    start = min(start, TARGET)
    end   = min(end, TARGET)
    clip  = make_lyric_clip(text, start, end, section)
    if clip:
        lyric_clips.append(clip)


# ---------------------------------------------------------------------------
# Section label (small, top-left)
# ---------------------------------------------------------------------------

def make_label(label_text, start, end):
    dur = min(end, TARGET) - min(start, TARGET)
    if dur <= 0:
        return None
    lbl = TextClip(
        text=label_text,
        font_size=28,
        color="#AAAACC",
        font=FONT_NORMAL,
        method="label",
    )
    lbl = lbl.with_duration(dur)
    lbl = lbl.with_effects([CrossFadeIn(0.4), CrossFadeOut(0.4)])
    lbl = lbl.with_start(min(start, TARGET))
    lbl = lbl.with_position((60, 40))
    return lbl


section_labels = {
    6:   ("♪ Verse 1", 32),
    32:  ("♪ Verse 2", 58),
    58:  ("✦ Chorus",  96),
    96:  ("♪ Verse 3", 122),
    122: ("♩ Bridge",  150),
    150: ("✦ Chorus",  184),
    184: ("♪ Outro",   TARGET),
}

label_clips = []
for start, (label, end) in section_labels.items():
    c = make_label(label, start, end)
    if c:
        label_clips.append(c)


# ---------------------------------------------------------------------------
# Title card (first 6 seconds)
# ---------------------------------------------------------------------------

title_main = TextClip(
    text="Jesus Comin' By and By",
    font_size=90,
    color="#FFD700",
    font=FONT_BOLD,
    method="label",
    stroke_color="#000000",
    stroke_width=4,
)
title_main = title_main.with_duration(6)
title_main = title_main.with_effects([CrossFadeIn(1.0), CrossFadeOut(1.0)])
title_main = title_main.with_start(0)
title_main = title_main.with_position(("center", int(H * 0.38)))

title_sub = TextClip(
    text="A Gospel Celebration",
    font_size=44,
    color="#FFFFFF",
    font=FONT_NORMAL,
    method="label",
    stroke_color="#000000",
    stroke_width=2,
)
title_sub = title_sub.with_duration(6)
title_sub = title_sub.with_effects([CrossFadeIn(1.5), CrossFadeOut(1.0)])
title_sub = title_sub.with_start(0)
title_sub = title_sub.with_position(("center", int(H * 0.54)))


# ---------------------------------------------------------------------------
# Compose everything
# ---------------------------------------------------------------------------

print("Compositing final video…")
all_clips = [bg_clip, title_main, title_sub] + label_clips + lyric_clips

final = CompositeVideoClip(all_clips, size=(W, H)).with_duration(TARGET)
final = final.with_fps(FPS)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

OUTPUT = os.path.join(os.path.dirname(__file__), "gospel_video.mp4")
print(f"Rendering to {OUTPUT}  ({TARGET}s @ {FPS}fps)…")
final.write_videofile(
    OUTPUT,
    fps=FPS,
    codec="libx264",
    audio=False,
    preset="fast",
    ffmpeg_params=["-crf", "23"],
    logger="bar",
)
print("Done! →", OUTPUT)
