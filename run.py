
"""
Brain Tumor Detection System
Main entry point for the application
"""
import os
import sys
from app import create_app, db
from app.database.models import User, Patient, Doctor, Scan, Diagnosis, Treatment, Report
from flask_migrate import Migrate

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create Flask app
app = create_app(os.getenv('FLASK_CONFIG') or 'default')
migrate = Migrate(app, db)


@app.shell_context_processor
def make_shell_context():
    """Make objects available in Flask shell"""
    return {
        'db': db,
        'User': User,
        'Patient': Patient,
        'Doctor': Doctor,
        'Scan': Scan,
        'Diagnosis': Diagnosis,
        'Treatment': Treatment,
        'Report': Report
    }


@app.cli.command("init-db")
def init_db():
    """Initialize the database"""
    with app.app_context():
        db.create_all()
        print('✅ Database initialized.')


@app.cli.command("create-admin")
def create_admin():
    """Create admin user"""
    from datetime import datetime

    with app.app_context():
        admin = User.query.filter_by(email='admin@braintumor.com').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@braintumor.com',
                role='admin',
                email_verified=True
            )
            admin.set_password('Admin@123')
            db.session.add(admin)
            db.session.commit()
            print('[OK] Admin user created: admin@braintumor.com / Admin@123')
        else:
            print('[i] Admin user already exists.')


@app.cli.command("create-test-users")
def create_test_users():
    """Create test patient and doctor users"""
    from datetime import datetime

    with app.app_context():
        # Patient
        patient_user = User.query.filter_by(email='patient@braintumor.com').first()
        if not patient_user:
            patient_user = User(
                username='patient',
                email='patient@braintumor.com',
                role='patient',
                email_verified=True
            )
            patient_user.set_password('Patient@123')
            db.session.add(patient_user)
            db.session.flush()
            patient = Patient(user_id=patient_user.id)
            db.session.add(patient)
            print('[OK] Patient user created: patient@braintumor.com / Patient@123')
        else:
            print('[i] Patient user already exists.')

        # Doctor
        doctor_user = User.query.filter_by(email='doctor@braintumor.com').first()
        if not doctor_user:
            doctor_user = User(
                username='doctor',
                email='doctor@braintumor.com',
                role='doctor',
                email_verified=True
            )
            doctor_user.set_password('Doctor@123')
            db.session.add(doctor_user)
            db.session.flush()
            doctor = Doctor(
                user_id=doctor_user.id,
                specialization='Neuro-Oncology',
                license_number='DOC-0001',
                hospital_affiliation='City Hospital',
                verification_status=True,
                verification_date=datetime.utcnow()
            )
            db.session.add(doctor)
            print('[OK] Doctor user created: doctor@braintumor.com / Doctor@123')
        else:
            print('[i] Doctor user already exists.')

        db.session.commit()


@app.after_request
def after_request(response):
    """Add security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


if __name__ == '__main__':
    # Windows consoles often use cp1252; avoid UnicodeEncodeError on emoji prints
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Initialize database on first run
    with app.app_context():
        db.create_all()

        # Create default users if they don't exist
        from app.database.models import User, Patient, Doctor

        # Admin
        if not User.query.filter_by(email='admin@braintumor.com').first():
            admin = User(username='admin', email='admin@braintumor.com', role='admin', email_verified=True)
            admin.set_password('Admin@123')
            db.session.add(admin)

        # Patient
        if not User.query.filter_by(email='patient@braintumor.com').first():
            patient_user = User(username='patient', email='patient@braintumor.com', role='patient', email_verified=True)
            patient_user.set_password('Patient@123')
            db.session.add(patient_user)
            db.session.flush()
            patient = Patient(user_id=patient_user.id)
            db.session.add(patient)

        # Doctor
        if not User.query.filter_by(email='doctor@braintumor.com').first():
            doctor_user = User(username='doctor', email='doctor@braintumor.com', role='doctor', email_verified=True)
            doctor_user.set_password('Doctor@123')
            db.session.add(doctor_user)
            db.session.flush()
            doctor = Doctor(
                user_id=doctor_user.id,
                specialization='Neuro-Oncology',
                license_number='DOC-0001',
                verification_status=True
            )
            db.session.add(doctor)

        db.session.commit()

    # Start IoT simulator
    try:
        from app.iot.iot_simulator import start_simulator

        if os.getenv('START_IOT', 'False').lower() == 'true':  # Disabled by default
            start_simulator(app)
            print('[OK] IoT Simulator started.')
    except Exception as e:
        print(f'[WARN] IoT Simulator could not start: {e}')

    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))

    print('\n*** Brain Tumor Detection System is running ***')
    print(f'    URL: http://localhost:{port}')
    print('    Admin:   admin@braintumor.com / Admin@123')
    print('    Patient: patient@braintumor.com / Patient@123')
    print('    Doctor:  doctor@braintumor.com / Doctor@123')
    print('    Press CTRL+C to stop\n')

    # Run app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=app.config.get('DEBUG', True),
        threaded=True
    )