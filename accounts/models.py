"""
Accounts App - Models
User profiles with Google OAuth2 authentication.
"""

from django.db import models
from django.utils import timezone


class UserProfile(models.Model):
    """User profile stored in SQLite."""
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=150, default='')
    google_id = models.CharField(max_length=255, blank=True, default='')
    profile_picture = models.URLField(max_length=500, blank=True, default='')
    is_verified = models.BooleanField(default=False)
    preference = models.CharField(
        max_length=20,
        choices=[('beginner', 'Beginner'), ('professional', 'Professional')],
        default='beginner'
    )
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.email

    class Meta:
        db_table = 'user_profiles'
