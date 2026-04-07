# ============================================
# FILE: app/config.py
# ============================================
import os
from datetime import timedelta


class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'

    # Database
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
    }

    # File upload
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}

    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)
    SESSION_REFRESH_EACH_REQUEST = True

    # Security
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_DURATION = timedelta(days=14)

    # Email
    MAIL_SERVER = os.environ.get('MAIL_SERVER') or 'smtp.gmail.com'
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')

    # Encryption
    ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY')

    # JWT
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)

    # Azure (for deployment)
    AZURE_IOT_HUB_CONNECTION_STRING = os.environ.get('AZURE_IOT_HUB_CONNECTION_STRING')
    AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///brain_tumor_dev.db'
    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql:///brain_tumor_prod'
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_SAMESITE = 'Lax'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

