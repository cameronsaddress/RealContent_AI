"""
Celery application configuration.
"""

from celery import Celery
from celery.schedules import crontab
from config import settings

# Create Celery app
celery_app = Celery(
    "video_pipeline",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "tasks.pipeline",
        "tasks.scrape",
    ],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Result backend settings
    result_expires=3600,  # 1 hour

    # Task routing
    task_routes={
        "tasks.pipeline.*": {"queue": "pipeline"},
        "tasks.scrape.*": {"queue": "scrape"},
    },

    # Beat schedule (cron jobs)
    beat_schedule={
        # Daily scrape at 6am UTC
        "daily-scrape-6am": {
            "task": "tasks.scrape.run_scrape",
            "schedule": crontab(hour=6, minute=0),
            "args": ({
                "niche": "real estate",
                "platforms": ["tiktok", "instagram"],
                "hashtags": ["realestate", "homebuying", "realtor"]
            },),
        },
        # Pipeline check every 15 minutes
        "pipeline-check-15min": {
            "task": "tasks.pipeline.run_pipeline",
            "schedule": crontab(minute="*/15"),
            "args": (None,),  # No specific ID = get next approved
        },
    },
)


# Optional: Configure task-specific settings
celery_app.conf.task_annotations = {
    "tasks.pipeline.run_pipeline": {
        "rate_limit": "10/m",  # Max 10 pipelines per minute
        "max_retries": 3,
    },
    "tasks.scrape.run_scrape": {
        "rate_limit": "1/m",  # Max 1 scrape per minute
        "max_retries": 2,
    },
}


if __name__ == "__main__":
    celery_app.start()
