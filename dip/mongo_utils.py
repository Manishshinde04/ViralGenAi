"""
DIP App - MongoDB Integration Utility
Handles all MongoDB read/write operations for:
- Processing history
- Analytics data
- Bulk job tracking
"""

from pymongo import MongoClient
from django.conf import settings
from datetime import datetime, timezone


def get_db():
    """Get MongoDB database connection."""
    client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000)
    return client[settings.MONGO_DB_NAME]


def save_processing_record(user_id, user_email, data):
    """Save a processing record to MongoDB."""
    try:
        db = get_db()
        record = {
            'user_id': user_id,
            'user_email': user_email,
            'original_filename': data.get('original_filename', ''),
            'processed_filename': data.get('processed_filename', ''),
            'filter_used': data.get('filter_used', 'none'),
            'tool_used': data.get('tool_used', 'single_process'),
            'quality_score': data.get('quality_score', 0),
            'platform': data.get('platform', 'original'),
            'creative_mode': data.get('creative_mode', 'none'),
            'category': data.get('category', 'none'),
            'features_used': data.get('features_used', []),
            'created_at': datetime.now(timezone.utc),
        }
        db.processing_history.insert_one(record)
        return True
    except Exception as e:
        print(f"MongoDB save error: {e}")
        return False


def get_user_history(user_id, limit=20):
    """Get processing history for a user."""
    try:
        db = get_db()
        records = list(
            db.processing_history
            .find({'user_id': user_id}, {'_id': 0})
            .sort('created_at', -1)
            .limit(limit)
        )
        return records
    except Exception as e:
        print(f"MongoDB fetch error: {e}")
        return []


def get_analytics_data(user_id):
    """Get analytics/stats for a user."""
    try:
        db = get_db()
        col = db.processing_history

        # Total images processed
        total = col.count_documents({'user_id': user_id})

        # Filter usage counts
        pipeline_filters = [
            {'$match': {'user_id': user_id, 'filter_used': {'$ne': 'none'}}},
            {'$group': {'_id': '$filter_used', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
            {'$limit': 8}
        ]
        filter_stats = list(col.aggregate(pipeline_filters))

        # Tool usage counts
        pipeline_tools = [
            {'$match': {'user_id': user_id}},
            {'$group': {'_id': '$tool_used', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        tool_stats = list(col.aggregate(pipeline_tools))

        # Daily activity (last 7 days)
        from datetime import timedelta
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        pipeline_daily = [
            {'$match': {'user_id': user_id, 'created_at': {'$gte': seven_days_ago}}},
            {'$group': {
                '_id': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$created_at'}},
                'count': {'$sum': 1}
            }},
            {'$sort': {'_id': 1}}
        ]
        daily_stats = list(col.aggregate(pipeline_daily))

        # Average quality score
        pipeline_quality = [
            {'$match': {'user_id': user_id, 'quality_score': {'$gt': 0}}},
            {'$group': {'_id': None, 'avg_score': {'$avg': '$quality_score'}}}
        ]
        quality_result = list(col.aggregate(pipeline_quality))
        avg_quality = round(quality_result[0]['avg_score'], 1) if quality_result else 0

        return {
            'total': total,
            'filter_stats': filter_stats,
            'tool_stats': tool_stats,
            'daily_stats': daily_stats,
            'avg_quality': avg_quality,
        }
    except Exception as e:
        print(f"MongoDB analytics error: {e}")
        return {
            'total': 0, 'filter_stats': [], 'tool_stats': [],
            'daily_stats': [], 'avg_quality': 0
        }


def save_bulk_job(user_id, filenames, operation):
    """Save a bulk processing job record."""
    try:
        db = get_db()
        job = {
            'user_id': user_id,
            'filenames': filenames,
            'operation': operation,
            'status': 'completed',
            'count': len(filenames),
            'created_at': datetime.now(timezone.utc),
        }
        result = db.bulk_jobs.insert_one(job)
        return str(result.inserted_id)
    except Exception as e:
        print(f"MongoDB bulk job error: {e}")
        return None
