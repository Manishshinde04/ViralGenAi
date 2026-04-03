"""
DIP App - Django Views
Handles all image processing requests, dashboard, analytics, history.
"""

import os
import uuid
import json
from io import BytesIO

from django.shortcuts import render, redirect
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.conf import settings
from PIL import Image

from .processing import (
    load_image, save_image_to_bytes, process_image,
    generate_ab_versions, detect_dominant_colors,
    calculate_quality_score, auto_enhance,
    generate_histogram, get_histogram_stats,
    generate_ai_caption
)
from .mongo_utils import (
    save_processing_record, get_user_history,
    get_analytics_data, save_bulk_job
)


def login_required_custom(view_func):
    """Custom login decorator using session-based auth."""
    def wrapper(request, *args, **kwargs):
        if not request.session.get('user_id'):
            return redirect('login')
        return view_func(request, *args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


def save_uploaded_image(file_obj, subfolder='uploads'):
    """Save uploaded image to media directory, return filename."""
    ext = os.path.splitext(file_obj.name)[1].lower() or '.jpg'
    filename = f"{uuid.uuid4().hex}{ext}"
    save_dir = os.path.join(settings.MEDIA_ROOT, subfolder)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb+') as dest:
        for chunk in file_obj.chunks():
            dest.write(chunk)
    return filename, filepath


def save_processed_image(img, fmt='PNG'):
    """Save processed PIL image to media/processed/, return filename."""
    ext = 'jpg' if fmt.upper() in ['JPG', 'JPEG'] else 'png'
    filename = f"processed_{uuid.uuid4().hex}.{ext}"
    save_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if fmt.upper() in ['JPG', 'JPEG'] and img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(filepath, format='JPEG' if ext == 'jpg' else 'PNG', quality=95)
    return filename, filepath


# ─── LANDING PAGE ───────────────────────────────────────────

def landing_view(request):
    if request.session.get('user_id'):
        return redirect('dashboard')
    return render(request, 'landing.html')


# ─── DASHBOARD ──────────────────────────────────────────────

@login_required_custom
def dashboard_view(request):
    user_id = request.session['user_id']
    analytics = get_analytics_data(user_id)
    history = get_user_history(user_id, limit=5)
    context = {
        'analytics': analytics,
        'history': history,
        'user_name': request.session.get('user_name', ''),
        'preference': request.session.get('user_preference', 'beginner'),
    }
    return render(request, 'dip/dashboard.html', context)


# ─── SINGLE IMAGE PROCESSING PAGE ───────────────────────────

@login_required_custom
def process_view(request):
    preference = request.session.get('user_preference', 'beginner')
    return render(request, 'dip/process.html', {'preference': preference})


@login_required_custom
def process_image_api(request):
    """AJAX endpoint: Upload + process image, return result."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    uploaded_file = request.FILES.get('image')
    if not uploaded_file:
        return JsonResponse({'error': 'No image provided'}, status=400)

    try:
        # Save original
        orig_filename, orig_path = save_uploaded_image(uploaded_file)

        # Load image
        img = Image.open(orig_path).convert('RGB')

        # Build processing options from POST data
        options = {
            'auto_enhance': request.POST.get('auto_enhance', 'true') == 'true',
            'mode': request.POST.get('mode', 'none'),
            'category': request.POST.get('category', 'none'),
            'creative_mode': request.POST.get('creative_mode', 'none'),
            'add_watermark': request.POST.get('add_watermark', 'false') == 'true',
            'watermark_opacity': int(request.POST.get('watermark_opacity', 180)),
            'background_template': request.POST.get('background_template', 'none'),
            'platform': request.POST.get('platform', 'original'),
            'remove_bg': request.POST.get('remove_bg', 'false') == 'true',
            'qr_text': request.POST.get('qr_text', ''),
        }

        # Process
        processed_img, quality, colors = process_image(img, options)

        # Save processed image
        export_fmt = request.POST.get('export_format', 'PNG')
        proc_filename, _ = save_processed_image(processed_img, fmt=export_fmt)

        # Build features list for analytics
        features_used = []
        if options.get('mode') not in ['none', '']:
            features_used.append(options['mode'])
        if options.get('creative_mode') not in ['none', '']:
            features_used.append(options['creative_mode'])
        if options.get('add_watermark'):
            features_used.append('watermark')
        if options.get('remove_bg'):
            features_used.append('bg_removal')

        # Save to MongoDB
        save_processing_record(
            user_id=request.session['user_id'],
            user_email=request.session.get('user_email', ''),
            data={
                'original_filename': orig_filename,
                'processed_filename': proc_filename,
                'filter_used': options.get('mode', 'none'),
                'tool_used': 'single_process',
                'quality_score': quality['total'],
                'platform': options.get('platform', 'original'),
                'creative_mode': options.get('creative_mode', 'none'),
                'category': options.get('category', 'none'),
                'features_used': features_used,
            }
        )

        return JsonResponse({
            'success': True,
            'processed_url': f"/media/processed/{proc_filename}",
            'original_url': f"/media/uploads/{orig_filename}",
            'quality': quality,
            'colors': colors,
            'download_filename': proc_filename,
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ─── A/B COMPARE ────────────────────────────────────────────

@login_required_custom
def ab_compare_view(request):
    """A/B comparison: generates two processed versions."""
    if request.method == 'POST':
        uploaded_file = request.FILES.get('image')
        if not uploaded_file:
            return JsonResponse({'error': 'No image'}, status=400)

        try:
            orig_filename, orig_path = save_uploaded_image(uploaded_file)
            img = Image.open(orig_path).convert('RGB')

            version_a, version_b = generate_ab_versions(img)

            fname_a, _ = save_processed_image(version_a, 'PNG')
            fname_b, _ = save_processed_image(version_b, 'PNG')

            quality_a = calculate_quality_score(version_a)
            quality_b = calculate_quality_score(version_b)

            return JsonResponse({
                'success': True,
                'version_a_url': f"/media/processed/{fname_a}",
                'version_b_url': f"/media/processed/{fname_b}",
                'quality_a': quality_a,
                'quality_b': quality_b,
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'dip/ab_compare.html')


# ─── BULK PROCESSING ────────────────────────────────────────

@login_required_custom
def bulk_view(request):
    return render(request, 'dip/bulk.html')


@login_required_custom
def bulk_process_api(request):
    """Process multiple uploaded images with the same operation."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    files = request.FILES.getlist('images')
    if not files:
        return JsonResponse({'error': 'No images provided'}, status=400)

    if len(files) > 10:
        return JsonResponse({'error': 'Maximum 10 images allowed'}, status=400)

    options = {
        'auto_enhance': request.POST.get('auto_enhance', 'true') == 'true',
        'mode': request.POST.get('mode', 'none'),
        'category': request.POST.get('category', 'none'),
        'creative_mode': request.POST.get('creative_mode', 'none'),
        'add_watermark': request.POST.get('add_watermark', 'false') == 'true',
        'watermark_opacity': 180,
        'background_template': 'none',
        'platform': request.POST.get('platform', 'original'),
        'remove_bg': False,
        'qr_text': '',
    }

    results = []
    processed_filenames = []

    for uploaded_file in files:
        try:
            orig_filename, orig_path = save_uploaded_image(uploaded_file)
            img = Image.open(orig_path).convert('RGB')
            processed_img, quality, _ = process_image(img, options)
            proc_filename, _ = save_processed_image(processed_img, 'PNG')
            processed_filenames.append(proc_filename)
            results.append({
                'original_name': uploaded_file.name,
                'processed_url': f"/media/processed/{proc_filename}",
                'quality_score': quality['total'],
                'status': 'success',
            })
        except Exception as e:
            results.append({
                'original_name': uploaded_file.name,
                'status': 'error',
                'error': str(e),
            })

    # Save bulk job to MongoDB
    save_bulk_job(
        user_id=request.session['user_id'],
        filenames=processed_filenames,
        operation=options.get('mode', 'auto_enhance')
    )

    return JsonResponse({'success': True, 'results': results})


# ─── ANALYTICS ──────────────────────────────────────────────

@login_required_custom
def analytics_view(request):
    user_id = request.session['user_id']
    analytics = get_analytics_data(user_id)
    return render(request, 'dip/analytics.html', {'analytics': analytics})


# ─── HISTORY ────────────────────────────────────────────────

@login_required_custom
def history_view(request):
    user_id = request.session['user_id']
    history = get_user_history(user_id, limit=50)
    return render(request, 'dip/history.html', {'history': history})


# ─── CAPTION & HISTOGRAM PAGE ───────────────────────────────

@login_required_custom
def caption_histogram_view(request):
    """Page for AI Caption Generator + Image Histogram."""
    return render(request, 'dip/caption_histogram.html')


@login_required_custom
def generate_caption_api(request):
    """AJAX endpoint: Generate AI captions for an uploaded image."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    uploaded_file = request.FILES.get('image')
    if not uploaded_file:
        return JsonResponse({'error': 'No image provided'}, status=400)

    try:
        orig_filename, orig_path = save_uploaded_image(uploaded_file)
        img = Image.open(orig_path).convert('RGB')

        # Generate AI captions
        caption_data = generate_ai_caption(img)

        # Save to MongoDB
        save_processing_record(
            user_id=request.session['user_id'],
            user_email=request.session.get('user_email', ''),
            data={
                'original_filename': orig_filename,
                'processed_filename': '',
                'filter_used': 'none',
                'tool_used': 'ai_caption',
                'quality_score': 0,
                'platform': 'original',
                'creative_mode': 'none',
                'category': 'none',
                'features_used': ['ai_caption'],
            }
        )

        return JsonResponse({
            'success': True,
            'original_url': f"/media/uploads/{orig_filename}",
            'captions': caption_data['captions'],
            'hashtags': caption_data['hashtags'],
            'mood': caption_data['mood'],
            'scene': caption_data['scene'],
            'color_theme': caption_data['color_theme'],
            'analysis': caption_data['analysis'],
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required_custom
def generate_histogram_api(request):
    """AJAX endpoint: Generate histogram for an uploaded image."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    uploaded_file = request.FILES.get('image')
    if not uploaded_file:
        return JsonResponse({'error': 'No image provided'}, status=400)

    try:
        orig_filename, orig_path = save_uploaded_image(uploaded_file)
        img = Image.open(orig_path).convert('RGB')

        # Generate histogram image
        hist_img = generate_histogram(img)
        hist_filename, _ = save_processed_image(hist_img, 'PNG')

        # Get histogram statistics
        stats = get_histogram_stats(img)

        # Save to MongoDB
        save_processing_record(
            user_id=request.session['user_id'],
            user_email=request.session.get('user_email', ''),
            data={
                'original_filename': orig_filename,
                'processed_filename': hist_filename,
                'filter_used': 'none',
                'tool_used': 'histogram',
                'quality_score': 0,
                'platform': 'original',
                'creative_mode': 'none',
                'category': 'none',
                'features_used': ['histogram'],
            }
        )

        return JsonResponse({
            'success': True,
            'original_url': f"/media/uploads/{orig_filename}",
            'histogram_url': f"/media/processed/{hist_filename}",
            'stats': stats,
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ─── DOWNLOAD ───────────────────────────────────────────────

@login_required_custom
def download_view(request, filename):
    """Serve processed image as download."""
    safe_name = os.path.basename(filename)
    filepath = os.path.join(settings.MEDIA_ROOT, 'processed', safe_name)
    if not os.path.exists(filepath):
        raise Http404('File not found')

    fmt = request.GET.get('format', 'PNG').upper()
    if fmt in ['JPG', 'JPEG']:
        content_type = 'image/jpeg'
        download_ext = '.jpg'
    else:
        content_type = 'image/png'
        download_ext = '.png'

    # If format conversion needed
    if fmt in ['JPG', 'JPEG'] and not safe_name.endswith('.jpg'):
        img = Image.open(filepath).convert('RGB')
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        response = FileResponse(buf, content_type=content_type)
        response['Content-Disposition'] = f'attachment; filename="viralgen_processed{download_ext}"'
        return response

    response = FileResponse(open(filepath, 'rb'), content_type=content_type)
    response['Content-Disposition'] = f'attachment; filename="viralgen_processed{download_ext}"'
    return response
