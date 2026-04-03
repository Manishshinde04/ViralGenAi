"""
Accounts App - Views
Google OAuth2 Authentication (Server-side Redirect Flow)
"""

import urllib.parse
import requests as http_requests
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from .models import UserProfile


def login_view(request):
    """Login page with Google Sign-In button."""
    if request.session.get('user_id'):
        return redirect('dashboard')
    return render(request, 'accounts/login.html')


def register_view(request):
    """Register page — same as login with Google (auto-registers)."""
    if request.session.get('user_id'):
        return redirect('dashboard')
    return render(request, 'accounts/register.html')


def google_login(request):
    """Redirect user to Google's OAuth2 consent screen."""
    redirect_uri = request.build_absolute_uri('/auth/google/callback/')

    params = {
        'client_id': settings.GOOGLE_CLIENT_ID,
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'offline',
        'prompt': 'select_account',
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
    return redirect(auth_url)


def google_callback(request):
    """Handle Google OAuth2 callback — exchange code for user info."""
    error = request.GET.get('error')
    if error:
        messages.error(request, f'Google login cancelled.')
        return redirect('login')

    code = request.GET.get('code')
    if not code:
        messages.error(request, 'No authorization code received from Google.')
        return redirect('login')

    redirect_uri = request.build_absolute_uri('/auth/google/callback/')

    try:
        # Exchange authorization code for tokens
        token_resp = http_requests.post(
            'https://oauth2.googleapis.com/token',
            data={
                'code': code,
                'client_id': settings.GOOGLE_CLIENT_ID,
                'client_secret': settings.GOOGLE_CLIENT_SECRET,
                'redirect_uri': redirect_uri,
                'grant_type': 'authorization_code',
            },
            timeout=10,
        )

        if token_resp.status_code != 200:
            messages.error(request, 'Failed to authenticate with Google. Please try again.')
            return redirect('login')

        tokens = token_resp.json()
        access_token = tokens.get('access_token')

        if not access_token:
            messages.error(request, 'No access token received from Google.')
            return redirect('login')

        # Get user info from Google
        userinfo_resp = http_requests.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            headers={'Authorization': f'Bearer {access_token}'},
            timeout=10,
        )

        if userinfo_resp.status_code != 200:
            messages.error(request, 'Failed to get user info from Google.')
            return redirect('login')

        userinfo = userinfo_resp.json()
        email = userinfo.get('email', '').lower()
        name = userinfo.get('name', '')
        picture = userinfo.get('picture', '')
        google_id = userinfo.get('sub', '')

        if not email:
            messages.error(request, 'No email found in Google response.')
            return redirect('login')

        # Create or get user
        user, created = UserProfile.objects.get_or_create(
            email=email,
            defaults={
                'name': name,
                'google_id': google_id,
                'profile_picture': picture,
                'is_verified': True,
            }
        )

        # Update existing user info
        if not created:
            if not user.name and name:
                user.name = name
            if google_id:
                user.google_id = google_id
            if picture:
                user.profile_picture = picture
            user.is_verified = True
            user.save()

        # Create session
        request.session['user_id'] = user.id
        request.session['user_email'] = user.email
        request.session['user_name'] = user.name
        request.session['user_preference'] = user.preference
        request.session['user_picture'] = user.profile_picture or ''

        action = 'Account created' if created else 'Logged in'
        messages.success(request, f'{action} successfully!')
        return redirect('dashboard')

    except http_requests.exceptions.Timeout:
        messages.error(request, 'Google authentication timed out. Please try again.')
        return redirect('login')
    except Exception as e:
        messages.error(request, f'Authentication failed: {str(e)}')
        return redirect('login')


def logout_view(request):
    """Clear session and redirect to landing."""
    request.session.flush()
    messages.success(request, 'Logged out successfully.')
    return redirect('landing')


def profile_view(request):
    """User profile settings — update preference."""
    if not request.session.get('user_id'):
        return redirect('login')

    user = UserProfile.objects.get(id=request.session['user_id'])

    if request.method == 'POST':
        preference = request.POST.get('preference', 'beginner')
        name = request.POST.get('name', user.name)
        user.preference = preference
        user.name = name
        user.save()
        request.session['user_name'] = name
        request.session['user_preference'] = preference
        messages.success(request, 'Profile updated!')

    return render(request, 'accounts/profile.html', {'user': user})
