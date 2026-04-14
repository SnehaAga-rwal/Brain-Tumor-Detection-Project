# ============================================
# FILE: app/__init__.py (COMPLETE FIX)
# ============================================
from flask import Flask, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from flask_migrate import Migrate
from flask_cors import CORS
from datetime import timedelta
import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
mail = Mail()
migrate = Migrate()


def create_app(config_class=None):
    """Application factory function"""
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static',
                static_url_path='/static')

    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///brain_tumor.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

    # Email configuration
    app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
    app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
    app.config['MAIL_SUPPRESS_SEND'] = not all([app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD']])

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    mail.init_app(app)
    migrate.init_app(app, db)
    CORS(app)

    # ============= DATABASE INIT (SAFE, MINIMAL) =============
    # Ensure tables exist before any route-level queries run (e.g., when using `flask run`).
    # Seeding is optional and only runs when the users table is empty.
    with app.app_context():
        # Import models so SQLAlchemy knows about all tables before create_all().
        from app.database import models as _models  # noqa: F401

        db.create_all()

        if os.getenv("SEED_DEFAULT_USERS", "true").lower() == "true":
            try:
                from app.database.models import User, Patient, Doctor
                from datetime import datetime

                # Only seed if there are no users yet (prevents duplicates on restart).
                if User.query.first() is None:
                    admin = User(
                        username="admin",
                        email="admin@braintumor.com",
                        role="admin",
                        email_verified=True,
                    )
                    admin.set_password("Admin@123")
                    db.session.add(admin)

                    patient_user = User(
                        username="patient",
                        email="patient@braintumor.com",
                        role="patient",
                        email_verified=True,
                    )
                    patient_user.set_password("Patient@123")
                    db.session.add(patient_user)
                    db.session.flush()
                    db.session.add(Patient(user_id=patient_user.id))

                    doctor_user = User(
                        username="doctor",
                        email="doctor@braintumor.com",
                        role="doctor",
                        email_verified=True,
                    )
                    doctor_user.set_password("Doctor@123")
                    db.session.add(doctor_user)
                    db.session.flush()
                    db.session.add(
                        Doctor(
                            user_id=doctor_user.id,
                            specialization="Neuro-Oncology",
                            license_number="DOC-0001",
                            hospital_affiliation="City Hospital",
                            verification_status=True,
                            verification_date=datetime.utcnow(),
                        )
                    )

                    db.session.commit()
            except Exception:
                # Never block app startup if seeding fails (tables are still created).
                db.session.rollback()

    # ============= JINJA2 FILTERS =============
    @app.template_filter('fromjson')
    def fromjson_filter(value):
        """Convert JSON string to Python object"""
        try:
            if value is None:
                return {}
            if isinstance(value, (dict, list)):
                return value
            if isinstance(value, str) and value.strip():
                return json.loads(value)
            return {}
        except Exception as e:
            app.logger.error(f"fromjson filter error: {e}")
            return {}

    @app.template_filter('range_diff')
    def range_diff_filter(values):
        """Calculate range difference for a list of numbers"""
        try:
            if not values:
                return 0
            valid_values = [v for v in values if v is not None and isinstance(v, (int, float))]
            if not valid_values:
                return 0
            return max(valid_values) - min(valid_values)
        except Exception as e:
            app.logger.error(f"range_diff filter error: {e}")
            return 0

    @app.template_filter('headline_confidence')
    def headline_confidence_filter(diagnosis):
        """Max of stored confidence and best single-model P(predicted class); fixes legacy blended-only rows."""
        from app.utils.confidence_display import diagnosis_headline_confidence
        return diagnosis_headline_confidence(diagnosis)

    @app.context_processor
    def utility_processor():
        """Add utility functions to template context"""
        from datetime import datetime
        return {
            'now': datetime.utcnow,
            'len': len
        }

    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'static', 'reports'), exist_ok=True)

    # ============= ERROR HANDLERS =============
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        app.logger.error(f"Server Error: {error}")
        return render_template('errors/500.html'), 500

    # ============= BLUEPRINT REGISTRATION =============
    from app.routes.auth_routes import bp as auth_bp
    from app.routes.patient_routes import bp as patient_bp
    from app.routes.doctor_routes import bp as doctor_bp
    from app.routes.admin_routes import bp as admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(patient_bp)
    app.register_blueprint(doctor_bp)
    app.register_blueprint(admin_bp)

    @app.route('/')
    @app.route('/index')
    def index():
        """Root route - redirect to login"""
        return redirect(url_for('auth.login'))

    @app.route('/favicon.ico')
    def favicon():
        """Handle favicon requests"""
        return ('', 204)

    @app.route('/health')
    def health():
        """Health check and ensemble weight load status (uses shared predictor singleton)."""
        try:
            from app.models.ensemble_model import get_ensemble_predictor

            st = get_ensemble_predictor().get_model_status()
            ready = st["resnet50"] and st["mobilenetv2"] and st["custom_cnn"]
            return {
                "status": "ok" if ready else "degraded",
                "models": {
                    "resnet50": st["resnet50"],
                    "mobilenetv2": st["mobilenetv2"],
                    "custom_cnn": st["custom_cnn"],
                },
                "message": (
                    "All ensemble weights loaded"
                    if ready
                    else "Some model files missing; copy .keras/.h5 weights into app/models/saved"
                ),
            }, 200
        except Exception as e:
            app.logger.exception("health check failed")
            return {"status": "error", "message": str(e)}, 500

    # ============= LOGGING =============
    if not app.debug:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler(
            'logs/brain_tumor.log',
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Brain Tumor Detection System startup')

    return app