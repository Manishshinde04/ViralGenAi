"""
ViralGen AI — Core DIP Processing Engine
=========================================
All image processing algorithms using Pillow + OpenCV.
Functions are documented for viva explanation.

Concepts covered:
- Spatial domain filtering (brightness, contrast, sharpness)
- Frequency domain concepts (edge detection, noise reduction)
- Color histogram analysis (dominant color detection)
- Morphological operations (background removal)
- Image quality metrics (brightness balance, contrast, noise)
"""

import io
import os
import math
import random
import numpy as np
import cv2
import qrcode
from PIL import (
    Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont,
    ImageOps, ImageStat
)
from django.conf import settings


# ─────────────────────────────────────────────────────────────
# FEATURE 17: IMAGE HISTOGRAM GENERATION
# Viva: Histogram = frequency distribution of pixel intensities
# ─────────────────────────────────────────────────────────────

def generate_histogram(img, width=720, height=400):
    """
    Generate a beautiful RGB + Grayscale histogram visualization.
    Viva: A histogram represents the frequency distribution of pixel
    intensities (0-255). Each channel (R, G, B, Gray) is plotted
    separately using cv2.calcHist().

    Returns: PIL Image of the histogram chart.
    """
    img_rgb = img.convert('RGB')
    cv_img = np.array(img_rgb)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # Create dark background canvas
    chart = np.zeros((height, width, 3), dtype=np.uint8)
    chart[:] = (20, 15, 10)  # Dark background (#0a0f14)

    # Draw subtle grid lines
    for i in range(1, 5):
        y = int(height * i / 5)
        cv2.line(chart, (50, y), (width - 20, y), (40, 40, 50), 1)
    for i in range(0, 256, 32):
        x = int(50 + i * (width - 70) / 255)
        cv2.line(chart, (x, 20), (x, height - 40), (40, 40, 50), 1)

    # Calculate histograms
    colors_bgr = [
        (50, 50, 255),   # Red channel
        (50, 200, 50),   # Green channel
        (255, 142, 79),  # Blue channel (accent blue)
    ]
    channel_names = ['Red', 'Green', 'Blue']

    plot_left = 50
    plot_right = width - 20
    plot_top = 20
    plot_bottom = height - 40
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    max_val = 0
    hists = []

    # Calculate all channel histograms
    for i in range(3):
        hist = cv2.calcHist([cv_img], [i], None, [256], [0, 256])
        hists.append(hist)
        max_val = max(max_val, hist.max())

    # Grayscale histogram
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hists.append(hist_gray)
    max_val = max(max_val, hist_gray.max())

    # Draw filled area histograms with transparency effect
    for idx, (hist, color) in enumerate(zip(hists[:3], colors_bgr)):
        overlay = chart.copy()
        pts = [(plot_left, plot_bottom)]
        for j in range(256):
            x = int(plot_left + j * plot_w / 255)
            y = int(plot_bottom - (hist[j][0] / max_val) * plot_h * 0.9)
            pts.append((x, y))
        pts.append((plot_right, plot_bottom))
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, chart, 0.85, 0, chart)

        # Draw line on top
        for j in range(1, 256):
            x1 = int(plot_left + (j - 1) * plot_w / 255)
            y1 = int(plot_bottom - (hist[j - 1][0] / max_val) * plot_h * 0.9)
            x2 = int(plot_left + j * plot_w / 255)
            y2 = int(plot_bottom - (hist[j][0] / max_val) * plot_h * 0.9)
            cv2.line(chart, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Draw grayscale histogram as dotted line
    gray_color = (180, 180, 180)
    for j in range(1, 256):
        x1 = int(plot_left + (j - 1) * plot_w / 255)
        y1 = int(plot_bottom - (hist_gray[j - 1][0] / max_val) * plot_h * 0.9)
        x2 = int(plot_left + j * plot_w / 255)
        y2 = int(plot_bottom - (hist_gray[j][0] / max_val) * plot_h * 0.9)
        if j % 3 != 0:  # Dotted effect
            cv2.line(chart, (x1, y1), (x2, y2), gray_color, 1, cv2.LINE_AA)

    # Draw axes
    cv2.line(chart, (plot_left, plot_top), (plot_left, plot_bottom), (100, 100, 120), 2)
    cv2.line(chart, (plot_left, plot_bottom), (plot_right, plot_bottom), (100, 100, 120), 2)

    # X-axis labels
    for val in [0, 64, 128, 192, 255]:
        x = int(plot_left + val * plot_w / 255)
        cv2.putText(chart, str(val), (x - 10, plot_bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 150, 170), 1, cv2.LINE_AA)

    # Title
    cv2.putText(chart, 'Image Histogram — Pixel Intensity Distribution',
                (plot_left, plot_top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 210, 230), 1, cv2.LINE_AA)

    # Legend
    legend_x = width - 200
    legend_y = 40
    legend_items = [('Red', (50, 50, 255)), ('Green', (50, 200, 50)),
                    ('Blue', (255, 142, 79)), ('Gray', gray_color)]
    for i, (name, color) in enumerate(legend_items):
        y = legend_y + i * 22
        cv2.rectangle(chart, (legend_x, y - 5), (legend_x + 14, y + 5), color, -1)
        cv2.putText(chart, name, (legend_x + 22, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 190, 210), 1, cv2.LINE_AA)

    # Y-axis label
    cv2.putText(chart, 'Frequency', (5, plot_top + plot_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 130, 150), 1, cv2.LINE_AA)

    # Convert BGR to RGB for PIL
    chart_rgb = cv2.cvtColor(chart, cv2.COLOR_BGR2RGB)
    return Image.fromarray(chart_rgb)


def get_histogram_stats(img):
    """
    Get statistical data from image histogram for display.
    Returns channel-wise statistics: mean, std, min, max, mode.
    """
    img_rgb = img.convert('RGB')
    cv_img = np.array(img_rgb)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    stats = {}
    channels = {'red': cv_img[:, :, 0], 'green': cv_img[:, :, 1],
                'blue': cv_img[:, :, 2], 'gray': gray}

    for name, channel in channels.items():
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
        mode_val = int(np.argmax(hist))
        stats[name] = {
            'mean': round(float(np.mean(channel)), 1),
            'std': round(float(np.std(channel)), 1),
            'min': int(np.min(channel)),
            'max': int(np.max(channel)),
            'mode': mode_val,
            'median': int(np.median(channel)),
        }

    # Overall image stats
    stats['brightness'] = stats['gray']['mean']
    stats['contrast'] = stats['gray']['std']
    stats['dynamic_range'] = stats['gray']['max'] - stats['gray']['min']

    return stats


# ─────────────────────────────────────────────────────────────
# FEATURE 18: AI CAPTION GENERATOR
# Viva: Image analysis + NLP-style caption generation
# Uses statistical analysis of image properties
# ─────────────────────────────────────────────────────────────

def generate_ai_caption(img):
    """
    Generate AI-powered captions for photos by analyzing:
    1. Color palette and dominant tones
    2. Brightness and mood detection
    3. Edge density (complexity/detail level)
    4. Saturation (vibrancy)
    5. Image dimensions and composition

    Returns dict with multiple caption styles and hashtags.
    """
    img_rgb = img.convert('RGB')
    cv_img = np.array(img_rgb)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # ── Analyze image properties ──
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))

    # Saturation analysis
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    mean_hue = float(np.mean(hsv[:, :, 0]))

    # Edge density (how detailed/complex the image is)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges)) / 255.0

    # Dominant color detection (simplified)
    dominant_colors = detect_dominant_colors(img, num_colors=4)
    dominant_hex = dominant_colors[0]['hex'] if dominant_colors else '#808080'

    # Color temperature analysis
    avg_r = float(np.mean(cv_img[:, :, 0]))
    avg_g = float(np.mean(cv_img[:, :, 1]))
    avg_b = float(np.mean(cv_img[:, :, 2]))

    # ── Determine mood ──
    mood = _detect_mood(mean_brightness, std_contrast, mean_saturation, avg_r, avg_b)

    # ── Determine scene type ──
    scene = _detect_scene(edge_density, mean_brightness, mean_saturation,
                          avg_r, avg_g, avg_b, img.size)

    # ── Determine color theme ──
    color_theme = _detect_color_theme(avg_r, avg_g, avg_b, mean_saturation)

    # ── Generate captions ──
    captions = _build_captions(mood, scene, color_theme, mean_brightness,
                              mean_saturation, edge_density)

    # ── Generate hashtags ──
    hashtags = _build_hashtags(mood, scene, color_theme)

    return {
        'captions': captions,
        'hashtags': hashtags,
        'mood': mood,
        'scene': scene,
        'color_theme': color_theme,
        'analysis': {
            'brightness': round(mean_brightness, 1),
            'contrast': round(std_contrast, 1),
            'saturation': round(mean_saturation, 1),
            'edge_density': round(edge_density * 100, 1),
            'dominant_color': dominant_hex,
            'warmth': 'warm' if avg_r > avg_b + 15 else ('cool' if avg_b > avg_r + 15 else 'neutral'),
        }
    }


def _detect_mood(brightness, contrast, saturation, avg_r, avg_b):
    """Detect image mood from visual properties."""
    if brightness < 60:
        if contrast > 60:
            return 'dramatic'
        return 'moody'
    elif brightness < 100:
        if saturation > 120:
            return 'vibrant'
        elif contrast > 70:
            return 'bold'
        return 'calm'
    elif brightness < 160:
        if saturation > 130:
            return 'energetic'
        elif avg_r > avg_b + 30:
            return 'warm'
        elif avg_b > avg_r + 30:
            return 'serene'
        return 'balanced'
    else:
        if saturation < 40:
            return 'minimal'
        elif avg_r > avg_b + 20:
            return 'dreamy'
        return 'bright'


def _detect_scene(edge_density, brightness, saturation, r, g, b, size):
    """Detect likely scene type from image analysis."""
    w, h = size
    aspect = w / max(h, 1)

    if saturation < 30 and brightness > 180:
        return 'document'
    elif g > r and g > b and saturation > 60:
        return 'nature'
    elif brightness < 80 and saturation < 60:
        return 'night'
    elif brightness > 180 and b > r and b > g:
        return 'sky'
    elif edge_density > 0.15:
        return 'urban'
    elif edge_density < 0.03 and saturation < 50:
        return 'minimal'
    elif saturation > 100 and brightness > 120:
        return 'colorful'
    elif r > 150 and g > 100 and b < 100:
        return 'sunset'
    elif aspect < 0.8:
        return 'portrait'
    elif aspect > 1.5:
        return 'landscape'
    else:
        return 'general'


def _detect_color_theme(r, g, b, saturation):
    """Detect the primary color theme."""
    if saturation < 25:
        return 'monochrome'
    elif r > g and r > b:
        if r > 180:
            return 'fiery'
        return 'warm_tones'
    elif g > r and g > b:
        return 'natural_green'
    elif b > r and b > g:
        if b > 180:
            return 'ocean_blue'
        return 'cool_tones'
    elif r > 150 and g > 100 and b < 80:
        return 'golden'
    elif r > 150 and b > 150 and g < 100:
        return 'purple_haze'
    else:
        return 'mixed'


def _build_captions(mood, scene, color_theme, brightness, saturation, edge_density):
    """Build multiple caption styles based on analysis."""
    captions = {}

    # ── Engaging / Social Media ──
    engaging_map = {
        ('dramatic', 'urban'): "Where shadows meet the city lights ✨",
        ('dramatic', 'night'): "The night speaks in whispers of light 🌙",
        ('moody', 'night'): "Lost in the beauty of the dark 🌑",
        ('moody', 'nature'): "Nature's quiet symphony 🍃",
        ('vibrant', 'nature'): "Nature painted in its boldest colors 🌿",
        ('vibrant', 'colorful'): "Life is too short for dull colors! 🎨",
        ('energetic', 'urban'): "City vibes, unstoppable energy ⚡",
        ('energetic', 'colorful'): "Turning up the color dial to maximum 🔥",
        ('warm', 'sunset'): "Chasing sunsets, one frame at a time 🌅",
        ('warm', 'landscape'): "Golden hour never disappoints ✨",
        ('serene', 'sky'): "Skies that speak to the soul 💙",
        ('serene', 'nature'): "Finding peace in nature's embrace 🌊",
        ('calm', 'landscape'): "Breathe in the view, breathe out the stress 🏔️",
        ('calm', 'nature'): "Where calm meets beautiful 🌸",
        ('bold', 'urban'): "Bold moves, bolder visions 🏙️",
        ('balanced', 'general'): "Perfectly balanced, as all things should be ⚖️",
        ('minimal', 'minimal'): "Less is more. Always. ◽",
        ('minimal', 'document'): "Clean lines, clear thoughts 📝",
        ('bright', 'sky'): "Blue skies ahead, always ☀️",
        ('bright', 'colorful'): "Bright days make the best memories 🌈",
        ('dreamy', 'landscape'): "Somewhere between reality and a dream 💫",
        ('dreamy', 'portrait'): "Dreams look like this ✨",
    }

    # Fallback engaging captions by mood
    mood_fallbacks = {
        'dramatic': "Every shadow tells a story 🎭",
        'moody': "In the mood for something extraordinary 🌙",
        'vibrant': "Living life in full color 🎨",
        'energetic': "Energy you can feel through the screen ⚡",
        'warm': "Warmth in every pixel 🌅",
        'serene': "Serenity captured in a frame 💫",
        'calm': "Peace is a powerful thing 🕊️",
        'bold': "Fortune favors the bold 🔥",
        'balanced': "Harmony in every detail ✨",
        'minimal': "Simplicity is the ultimate sophistication 🤍",
        'bright': "Shining brighter than ever ☀️",
        'dreamy': "Where dreams become reality 💭",
    }

    captions['engaging'] = engaging_map.get(
        (mood, scene),
        mood_fallbacks.get(mood, "A moment worth remembering ✨")
    )

    # ── Professional / Portfolio ──
    if scene == 'nature':
        captions['professional'] = "Capturing the raw beauty of the natural world through the lens."
    elif scene == 'urban':
        captions['professional'] = "An exploration of urban architecture and the geometry of city life."
    elif scene == 'portrait':
        captions['professional'] = "A study in light, shadow, and human expression."
    elif scene == 'landscape':
        captions['professional'] = "Wide-angle perspective revealing the grandeur of the landscape."
    elif scene == 'night':
        captions['professional'] = "Low-light photography capturing the essence of the night."
    elif scene == 'sunset':
        captions['professional'] = "Golden hour photography — the magic of natural lighting."
    elif scene == 'sky':
        captions['professional'] = "Atmospheric photography exploring light and cloud formations."
    else:
        captions['professional'] = "A carefully composed frame exploring light and color harmony."

    # ── Descriptive / Alt-text ──
    warmth = 'warm' if brightness > 140 else ('cool' if brightness < 90 else 'neutral')
    detail = 'highly detailed' if edge_density > 0.1 else ('minimal' if edge_density < 0.03 else 'moderately detailed')
    vibe = 'vibrant' if saturation > 100 else ('muted' if saturation < 50 else 'natural')

    captions['descriptive'] = (
        f"A {detail} image with {warmth} tones and {vibe} colors. "
        f"The overall mood is {mood}, with a {color_theme.replace('_', ' ')} color palette."
    )

    # ── Witty / Fun ──
    witty_map = {
        'dramatic': "I didn't choose the dramatic life, it chose my camera 📸",
        'moody': "Current mood: aesthetically unbothered 😎",
        'vibrant': "My camera sees the world in ultra HD vibes 🎬",
        'energetic': "This pic has more energy than my morning coffee ☕",
        'warm': "If warmth had a face, it'd look like this 🔆",
        'serene': "POV: You found inner peace in a photo 🧘",
        'calm': "Calm? More like FABULOUS-ly composed 💅",
        'bold': "Go bold or go home 🚀",
        'balanced': "My life isn't balanced but at least this photo is 😅",
        'minimal': "I decluttered my life. Starting with this photo 🧹",
        'bright': "Warning: This photo may cause excessive smiling 😄",
        'dreamy': "Is this a photo or a daydream? You decide 💤",
    }
    captions['witty'] = witty_map.get(mood, "Plot twist: this photo just won the internet 🏆")

    # ── Story / Narrative ──
    story_starters = {
        'dramatic': "There's something about the way light cuts through darkness...",
        'moody': "Some moments are best felt, not explained.",
        'vibrant': "Colors have a language of their own, and this one speaks volumes.",
        'energetic': "Every frame has a heartbeat. This one races.",
        'warm': "There's a warmth here that goes beyond temperature.",
        'serene': "In the stillness, everything becomes clear.",
        'calm': "Time seems to slow down in moments like these.",
        'bold': "The boldest stories are often told without words.",
        'balanced': "When everything falls into place, you just know.",
        'minimal': "In the space between nothing and everything, there's this.",
        'bright': "Some days, the world just decides to glow.",
        'dreamy': "Somewhere between waking and sleeping, there's this view.",
    }
    captions['story'] = story_starters.get(mood, "Every picture holds a thousand untold stories.")

    return captions


def _build_hashtags(mood, scene, color_theme):
    """Generate relevant hashtags based on analysis."""
    tags = set()

    # Mood-based
    mood_tags = {
        'dramatic': ['#dramatic', '#moodygrams', '#darkart'],
        'moody': ['#moody', '#moodygrams', '#aesthetic'],
        'vibrant': ['#vibrant', '#colorpop', '#boldcolors'],
        'energetic': ['#energetic', '#lively', '#dynamic'],
        'warm': ['#warmtones', '#goldenhour', '#warmth'],
        'serene': ['#serene', '#peaceful', '#tranquil'],
        'calm': ['#calm', '#peaceful', '#zen'],
        'bold': ['#bold', '#striking', '#powerful'],
        'balanced': ['#balanced', '#harmony', '#composition'],
        'minimal': ['#minimal', '#minimalism', '#lessismore'],
        'bright': ['#bright', '#sunshine', '#lightandbright'],
        'dreamy': ['#dreamy', '#ethereal', '#dreamscape'],
    }
    tags.update(mood_tags.get(mood, ['#photography']))

    # Scene-based
    scene_tags = {
        'nature': ['#nature', '#naturephotography', '#earthfocus'],
        'urban': ['#urban', '#cityscape', '#streetphotography'],
        'night': ['#nightphotography', '#nightvibes', '#afterdark'],
        'sky': ['#skyphotography', '#cloudporn', '#skylovers'],
        'sunset': ['#sunset', '#sunsetlovers', '#goldenhour'],
        'portrait': ['#portrait', '#portraitphotography', '#faces'],
        'landscape': ['#landscape', '#landscapephotography', '#scenic'],
        'colorful': ['#colorsplash', '#coloursoflife', '#rainbow'],
        'minimal': ['#minimalart', '#whitespace', '#cleandesign'],
        'document': ['#document', '#typography', '#design'],
        'general': ['#picoftheday', '#photooftheday', '#capture'],
    }
    tags.update(scene_tags.get(scene, ['#photography']))

    # Color theme
    color_tags = {
        'monochrome': ['#blackandwhite', '#bnw', '#monochrome'],
        'fiery': ['#fiery', '#flames', '#redaesthetic'],
        'warm_tones': ['#warmpalette', '#earthtones', '#cozy'],
        'natural_green': ['#greenery', '#plantsofinstagram', '#lush'],
        'ocean_blue': ['#oceanblue', '#blueaesthetic', '#deepblue'],
        'cool_tones': ['#coolvibes', '#bluevibes', '#icytones'],
        'golden': ['#goldenvibes', '#amber', '#honeytones'],
        'purple_haze': ['#purplehaze', '#ultraviolet', '#lavender'],
        'mixed': ['#colorful', '#multicolor', '#spectrum'],
    }
    tags.update(color_tags.get(color_theme, ['#colors']))

    # Universal tags
    tags.update(['#photography', '#photoediting', '#viralgenai'])

    return sorted(list(tags))[:15]  # Cap at 15 hashtags


# ─────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────

def pil_to_cv(pil_img):
    """Convert PIL Image to OpenCV numpy array (BGR)."""
    rgb = np.array(pil_img.convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img):
    """Convert OpenCV numpy array (BGR) to PIL Image."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_image(file_obj):
    """Load image from file object and convert to RGBA."""
    img = Image.open(file_obj)
    if img.mode != 'RGBA':
        img = img.convert('RGB')
    return img


def save_image_to_bytes(img, fmt='PNG'):
    """Save PIL image to bytes buffer."""
    buf = io.BytesIO()
    save_fmt = 'JPEG' if fmt.upper() in ['JPG', 'JPEG'] else fmt.upper()
    if save_fmt == 'JPEG' and img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(buf, format=save_fmt, quality=95)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────
# FEATURE 13: AUTO IMAGE ENHANCEMENT (CORE DIP)
# Viva: Spatial domain — point operations on pixel intensity
# ─────────────────────────────────────────────────────────────

def auto_enhance(img):
    """
    Auto-enhance image using adaptive correction.
    Steps:
    1. Analyze image statistics (mean brightness, contrast)
    2. Apply adaptive brightness correction
    3. Enhance contrast using histogram equalization concept
    4. Sharpen edges using unsharp masking
    """
    stat = ImageStat.Stat(img.convert('L'))
    mean_brightness = stat.mean[0]  # 0-255

    # Adaptive brightness correction
    brightness_factor = 128 / max(mean_brightness, 1)
    brightness_factor = max(0.7, min(1.5, brightness_factor))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    # Contrast enhancement (CLAHE-inspired)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    # Sharpness improvement (Unsharp masking)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.4)

    # Color saturation boost
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.15)

    return img


# ─────────────────────────────────────────────────────────────
# FEATURE 2: PROCESSING MODE SELECTION
# Viva: Different spatial filters applied as convolution kernels
# ─────────────────────────────────────────────────────────────

def apply_mode(img, mode):
    """
    Apply predefined processing mode.
    Modes: high_contrast | soft_enhancement | edge_highlight | noise_reduction
    """
    if mode == 'high_contrast':
        # High contrast: strong contrast + sharpness + brightness correction
        img = ImageEnhance.Contrast(img).enhance(2.2)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        img = ImageEnhance.Brightness(img).enhance(1.05)
        img = ImageEnhance.Color(img).enhance(1.3)

    elif mode == 'soft_enhancement':
        # Soft: slight Gaussian blur + warm tones
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        img = ImageEnhance.Brightness(img).enhance(1.1)
        img = ImageEnhance.Color(img).enhance(1.2)
        img = ImageEnhance.Contrast(img).enhance(0.9)

    elif mode == 'edge_highlight':
        # Edge highlight: Sobel edge detection blended with original
        # Viva: Sobel operator = gradient-based edge detector
        cv_img = pil_to_cv(img.convert('RGB'))
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
        edges = np.uint8(np.clip(edges, 0, 255))
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Blend edges with original
        blended = cv2.addWeighted(cv_img, 0.7, edges_colored, 0.3, 0)
        img = cv_to_pil(blended)
        img = ImageEnhance.Contrast(img).enhance(1.5)

    elif mode == 'noise_reduction':
        # Noise reduction: Median filter (removes salt-and-pepper noise)
        # Viva: Median filter preserves edges better than mean filter
        cv_img = pil_to_cv(img.convert('RGB'))
        denoised = cv2.medianBlur(cv_img, 5)
        # Bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        img = cv_to_pil(denoised)
        img = ImageEnhance.Sharpness(img).enhance(1.2)

    return img


# ─────────────────────────────────────────────────────────────
# FEATURE 6: CATEGORY-BASED PROCESSING PRESETS
# ─────────────────────────────────────────────────────────────

def apply_category_preset(img, category):
    """Apply optimized filters based on image category."""
    if category == 'portrait':
        # Skin tone enhancement, slight blur for smooth look
        img = ImageEnhance.Brightness(img).enhance(1.1)
        img = ImageEnhance.Color(img).enhance(1.15)
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img = ImageEnhance.Sharpness(img).enhance(0.8)

    elif category == 'landscape':
        # Vibrant colors, high contrast for scenery
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = ImageEnhance.Color(img).enhance(1.5)
        img = ImageEnhance.Sharpness(img).enhance(1.3)

    elif category == 'document':
        # High contrast B&W for text readability
        # Viva: Otsu's thresholding for binary document processing
        cv_img = pil_to_cv(img.convert('RGB'))
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = Image.fromarray(thresh).convert('RGB')

    elif category == 'low_light':
        # Brighten + denoise for dark images
        img = ImageEnhance.Brightness(img).enhance(1.8)
        img = ImageEnhance.Contrast(img).enhance(1.3)
        cv_img = pil_to_cv(img.convert('RGB'))
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
        img = cv_to_pil(denoised)

    return img


# ─────────────────────────────────────────────────────────────
# FEATURE 5: AUTO COLOR DETECTION
# Viva: Color histogram analysis using OpenCV
# ─────────────────────────────────────────────────────────────

def detect_dominant_colors(img, num_colors=6):
    """
    Detect dominant colors using k-means clustering on pixel colors.
    Viva: K-means clustering applied to 3D color space (R, G, B)
    Returns list of (hex_color, percentage) tuples.
    """
    img_rgb = img.convert('RGB').resize((150, 150))
    pixels = np.float32(np.array(img_rgb).reshape(-1, 3))

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10,
        cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)

    # Count occurrences
    unique, counts = np.unique(labels, return_counts=True)
    total = sum(counts)

    colors = []
    for idx, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        r, g, b = centers[idx]
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        percentage = round((count / total) * 100, 1)
        colors.append({'hex': hex_color, 'percentage': percentage})

    return colors


def suggest_filter_for_colors(colors):
    """Suggest an enhancement filter based on dominant colors."""
    if not colors:
        return 'auto_enhance'

    # Check average brightness of dominant colors
    top_hex = colors[0]['hex']
    r = int(top_hex[1:3], 16)
    g = int(top_hex[3:5], 16)
    b = int(top_hex[5:7], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if luminance < 80:
        return 'low_light'  # Dark image → brighten
    elif luminance > 200:
        return 'soft_enhancement'  # Very bright → soften
    elif abs(r - b) > 60:
        return 'high_contrast'  # Strong color cast → contrast
    else:
        return 'soft_enhancement'


# ─────────────────────────────────────────────────────────────
# FEATURE 3: WATERMARK
# ─────────────────────────────────────────────────────────────

def add_watermark(img, opacity=128, text='Processed by DIP Lab'):
    """
    Add watermark text to bottom-right corner.
    Viva: Alpha compositing — overlay with transparency
    """
    img = img.convert('RGBA')
    width, height = img.size

    # Create watermark layer
    watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)

    # Font size proportional to image size
    font_size = max(16, width // 40)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
    except Exception:
        font = ImageFont.load_default()

    # Position: bottom-right with padding
    padding = 15
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = width - text_width - padding
    y = height - text_height - padding

    # Draw semi-transparent background rect
    draw.rectangle([x - 8, y - 4, x + text_width + 8, y + text_height + 4],
                   fill=(0, 0, 0, int(opacity * 0.6)))

    # Draw text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity))

    # Composite
    out = Image.alpha_composite(img, watermark)
    return out.convert('RGB')


# ─────────────────────────────────────────────────────────────
# FEATURE 4: BACKGROUND TEMPLATE OVERLAY
# ─────────────────────────────────────────────────────────────

def apply_background_template(img, template='minimal'):
    """Apply a styled background template behind the image."""
    padding = 60
    new_width = img.width + padding * 2
    new_height = img.height + padding * 2

    if template == 'minimal':
        bg = Image.new('RGB', (new_width, new_height), (245, 245, 245))

    elif template == 'dark':
        bg = Image.new('RGB', (new_width, new_height), (18, 18, 30))

    elif template == 'gradient':
        bg = Image.new('RGB', (new_width, new_height))
        draw = ImageDraw.Draw(bg)
        for y in range(new_height):
            ratio = y / new_height
            r = int(102 + ratio * (139 - 102))
            g = int(51 + ratio * (0 - 51))
            b = int(153 + ratio * (255 - 153))
            draw.line([(0, y), (new_width, y)], fill=(r, g, b))

    elif template == 'professional':
        bg = Image.new('RGB', (new_width, new_height), (13, 17, 23))
        draw = ImageDraw.Draw(bg)
        # Border frame
        border_color = (88, 166, 255)
        draw.rectangle([2, 2, new_width - 3, new_height - 3],
                       outline=border_color, width=3)
        draw.rectangle([8, 8, new_width - 9, new_height - 9],
                       outline=(44, 88, 132), width=1)
    else:
        bg = Image.new('RGB', (new_width, new_height), (240, 240, 240))

    # Paste image onto background
    bg.paste(img.convert('RGB'), (padding, padding))
    return bg


# ─────────────────────────────────────────────────────────────
# FEATURE 1: MULTI-OUTPUT IMAGE EXPORT (RESIZE FOR PLATFORM)
# ─────────────────────────────────────────────────────────────

PLATFORM_SIZES = {
    'instagram': (1080, 1080),   # 1:1
    'facebook': (1080, 1350),    # 4:5
    'linkedin': (1200, 628),     # 16:9 (approx)
    'twitter': (1200, 675),
    'original': None,
}


def resize_for_platform(img, platform='original'):
    """
    Resize image for social media platform dimensions.
    Viva: Image resampling — Lanczos/BICUBIC for quality downscaling
    """
    if platform == 'original' or platform not in PLATFORM_SIZES:
        return img

    target_w, target_h = PLATFORM_SIZES[platform]
    src_w, src_h = img.size

    # Calculate scale to fill target dimensions
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    # Resize with Lanczos (high quality)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to target
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img_cropped = img_resized.crop((left, top, left + target_w, top + target_h))

    return img_cropped


# ─────────────────────────────────────────────────────────────
# FEATURE 16: MODE-BASED PROCESSING (Night/Vintage/Summer)
# ─────────────────────────────────────────────────────────────

def apply_creative_mode(img, mode):
    """
    Apply creative processing modes.
    Viva: Color space transformations and LUT (Look-Up Table) application
    """
    if mode == 'night':
        # Night mode: brighten + blue tint + noise reduction
        img = ImageEnhance.Brightness(img).enhance(1.6)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        # Blue tint via color matrix
        cv_img = pil_to_cv(img.convert('RGB'))
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 8, 8, 7, 21)
        # Blue shift
        denoised[:, :, 0] = np.clip(denoised[:, :, 0].astype(int) + 20, 0, 255)  # B
        denoised[:, :, 2] = np.clip(denoised[:, :, 2].astype(int) - 10, 0, 255)  # R
        img = cv_to_pil(denoised)

    elif mode == 'vintage':
        # Vintage: sepia tone + reduced saturation + slight fade
        img = img.convert('RGB')
        r, g, b = img.split()
        r = r.point(lambda i: min(255, int(i * 1.1 + 30)))
        g = g.point(lambda i: min(255, int(i * 0.95 + 15)))
        b = b.point(lambda i: min(255, int(i * 0.8 - 10)))
        img = Image.merge('RGB', (r, g, b))
        img = ImageEnhance.Color(img).enhance(0.5)
        img = ImageEnhance.Contrast(img).enhance(0.85)
        # Add grain
        cv_img = pil_to_cv(img)
        noise = np.random.normal(0, 8, cv_img.shape).astype(np.int16)
        noisy = np.clip(cv_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv_to_pil(noisy)

    elif mode == 'summer':
        # Summer: warm tones, high saturation, bright
        img = ImageEnhance.Color(img).enhance(1.6)
        img = ImageEnhance.Brightness(img).enhance(1.15)
        # Warm (orange) shift
        img = img.convert('RGB')
        r, g, b = img.split()
        r = r.point(lambda i: min(255, int(i * 1.08 + 10)))
        g = g.point(lambda i: min(255, int(i * 1.02)))
        b = b.point(lambda i: max(0, int(i * 0.88 - 8)))
        img = Image.merge('RGB', (r, g, b))
        img = ImageEnhance.Contrast(img).enhance(1.1)

    return img


# ─────────────────────────────────────────────────────────────
# FEATURE 10: IMAGE QUALITY SCORE
# Viva: Statistical analysis of image quality metrics
# ─────────────────────────────────────────────────────────────

def calculate_quality_score(img):
    """
    Calculate image quality score (0-100) based on:
    - Brightness balance (25 pts): ideal mean = 128
    - Contrast level (25 pts): measured by std deviation
    - Sharpness/Detail (25 pts): Laplacian variance
    - Color richness (25 pts): saturation in HSV
    Returns dict with score and component breakdown.
    """
    img_rgb = img.convert('RGB')
    img_gray = img.convert('L')

    cv_gray = np.array(img_gray)
    cv_rgb = np.array(img_rgb)

    # 1. Brightness score (closer to 128 = better)
    mean_brightness = np.mean(cv_gray)
    brightness_score = max(0, 25 - abs(mean_brightness - 128) * 25 / 128)

    # 2. Contrast score (higher std dev = better up to a point)
    std_dev = np.std(cv_gray)
    contrast_score = min(25, (std_dev / 80) * 25)

    # 3. Sharpness score (Laplacian variance)
    laplacian = cv2.Laplacian(cv_gray, cv2.CV_64F)
    sharpness_var = laplacian.var()
    sharpness_score = min(25, (sharpness_var / 500) * 25)

    # 4. Color richness (mean saturation in HSV)
    cv_hsv = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2HSV)
    mean_saturation = np.mean(cv_hsv[:, :, 1])
    color_score = min(25, (mean_saturation / 255) * 25)

    total = brightness_score + contrast_score + sharpness_score + color_score
    total_clamped = max(0, min(100, round(total)))

    return {
        'total': total_clamped,
        'brightness': round(brightness_score, 1),
        'contrast': round(contrast_score, 1),
        'sharpness': round(sharpness_score, 1),
        'color': round(color_score, 1),
        'label': _quality_label(total_clamped),
    }


def _quality_label(score):
    if score >= 80:
        return ('Excellent', 'success')
    elif score >= 60:
        return ('Good', 'info')
    elif score >= 40:
        return ('Fair', 'warning')
    else:
        return ('Poor', 'danger')


# ─────────────────────────────────────────────────────────────
# FEATURE 14: BACKGROUND REMOVAL
# Viva: Thresholding and morphological operations
# ─────────────────────────────────────────────────────────────

def remove_background(img):
    """
    Remove simple white/light background using GrabCut algorithm.
    Viva: GrabCut = iterative graph-cut based segmentation
    Falls back to thresholding for simple images.
    """
    img_rgb = img.convert('RGB')
    cv_img = np.array(img_rgb)

    height, width = cv_img.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Use center 80% as foreground hint
    rect = (
        int(width * 0.05), int(height * 0.05),
        int(width * 0.9), int(height * 0.9)
    )

    try:
        cv2.grabCut(cv_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    except Exception:
        # Fallback: simple thresholding on grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        _, fg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Apply mask to create RGBA image
    img_rgba = img_rgb.convert('RGBA')
    r, g, b, a = img_rgba.split()
    fg_pil = Image.fromarray(fg_mask)
    img_out = Image.merge('RGBA', (r, g, b, fg_pil))

    return img_out


# ─────────────────────────────────────────────────────────────
# FEATURE 15: QR CODE GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_qr_code(text, size=200):
    """Generate QR code image from text input."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=8,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color='black', back_color='white').convert('RGB')
    qr_img = qr_img.resize((size, size), Image.LANCZOS)
    return qr_img


def overlay_qr_on_image(img, qr_text, position='bottom-right'):
    """Overlay QR code on processed image corner."""
    qr_img = generate_qr_code(qr_text, size=max(80, img.width // 8))

    img_out = img.convert('RGB').copy()
    qr_w, qr_h = qr_img.size
    padding = 15

    if position == 'bottom-right':
        x = img_out.width - qr_w - padding
        y = img_out.height - qr_h - padding
    elif position == 'bottom-left':
        x, y = padding, img_out.height - qr_h - padding
    elif position == 'top-right':
        x, y = img_out.width - qr_w - padding, padding
    else:
        x, y = padding, padding

    img_out.paste(qr_img, (x, y))
    return img_out


# ─────────────────────────────────────────────────────────────
# FEATURE 9: A/B PROCESSING MODE
# ─────────────────────────────────────────────────────────────

def generate_ab_versions(img):
    """
    Generate two processing versions for comparison.
    Version A: High Contrast enhancement
    Version B: Soft / smooth enhancement
    """
    version_a = apply_mode(img.copy(), 'high_contrast')
    version_a = add_watermark(version_a, text='Version A — High Contrast')

    version_b = apply_mode(img.copy(), 'soft_enhancement')
    version_b = add_watermark(version_b, text='Version B — Soft Enhancement')

    return version_a, version_b


# ─────────────────────────────────────────────────────────────
# MASTER PROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────

def process_image(img, options):
    """
    Master processing pipeline.
    options dict keys:
      - mode: str (high_contrast | soft_enhancement | edge_highlight | noise_reduction)
      - category: str (portrait | landscape | document | low_light)
      - creative_mode: str (night | vintage | summer | none)
      - auto_enhance: bool
      - add_watermark: bool
      - watermark_opacity: int (0-255)
      - background_template: str (minimal | dark | gradient | professional | none)
      - platform: str (instagram | facebook | linkedin | original)
      - remove_bg: bool
      - qr_text: str (optional)
    Returns: processed PIL Image, quality_score dict, colors list
    """
    # Step 1: Auto enhance if requested
    if options.get('auto_enhance', True):
        img = auto_enhance(img)

    # Step 2: Processing mode
    mode = options.get('mode', 'none')
    if mode and mode != 'none':
        img = apply_mode(img, mode)

    # Step 3: Category preset
    category = options.get('category', 'none')
    if category and category != 'none':
        img = apply_category_preset(img, category)

    # Step 4: Creative mode
    creative = options.get('creative_mode', 'none')
    if creative and creative != 'none':
        img = apply_creative_mode(img, creative)

    # Step 5: Background removal
    if options.get('remove_bg', False):
        img = remove_background(img)
        img = img.convert('RGB')  # Back to RGB for further processing

    # Step 6: Background template
    template = options.get('background_template', 'none')
    if template and template != 'none':
        img = apply_background_template(img, template)

    # Step 7: Platform resize
    platform = options.get('platform', 'original')
    img = resize_for_platform(img, platform)

    # Step 8: Watermark
    if options.get('add_watermark', False):
        opacity = options.get('watermark_opacity', 180)
        img = add_watermark(img, opacity=opacity)

    # Step 9: QR code overlay
    qr_text = options.get('qr_text', '')
    if qr_text:
        img = overlay_qr_on_image(img, qr_text)

    # Step 10: Calculate quality score
    quality = calculate_quality_score(img)

    # Step 11: Detect colors
    colors = detect_dominant_colors(img)

    return img, quality, colors
