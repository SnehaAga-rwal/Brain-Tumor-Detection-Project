# ============================================
# FILE: app/database/models.py
# ============================================
from app import db, login_manager
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from app.utils.encryption import encrypt_data, decrypt_data
import json


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    """User model for all roles (patient, doctor, admin)"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='patient')  # patient, doctor, admin

    # Authentication fields
    sso_id = db.Column(db.String(100), unique=True, nullable=True)
    email_verified = db.Column(db.Boolean, default=False)
    two_factor_enabled = db.Column(db.Boolean, default=False)
    two_factor_secret = db.Column(db.String(100), nullable=True)

    # Security tracking
    last_login = db.Column(db.DateTime, nullable=True)
    last_login_ip = db.Column(db.String(45), nullable=True)
    login_attempts = db.Column(db.Integer, default=0)
    account_locked_until = db.Column(db.DateTime, nullable=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient_profile = db.relationship('Patient', backref='user', uselist=False, cascade='all, delete-orphan')
    doctor_profile = db.relationship('Doctor', backref='user', uselist=False, cascade='all, delete-orphan')
    audit_logs = db.relationship('AuditLog', backref='user', lazy='dynamic')
    notifications = db.relationship('Notification', backref='user', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        return self.role == 'admin'

    def is_doctor(self):
        return self.role == 'doctor'

    def is_patient(self):
        return self.role == 'patient'

    def __repr__(self):
        return f'<User {self.username}>'


class Patient(db.Model):
    """Patient profile with encrypted sensitive data"""
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)

    # Encrypted fields (AES-256)
    _phone = db.Column('phone', db.LargeBinary, nullable=True)
    _date_of_birth = db.Column('date_of_birth', db.LargeBinary, nullable=True)
    _address = db.Column('address', db.LargeBinary, nullable=True)
    _medical_history = db.Column('medical_history', db.LargeBinary, nullable=True)
    _emergency_contact = db.Column('emergency_contact', db.LargeBinary, nullable=True)

    # Non-sensitive fields
    blood_group = db.Column(db.String(5), nullable=True)
    allergies = db.Column(db.Text, nullable=True)  # JSON string
    chronic_conditions = db.Column(db.Text, nullable=True)  # JSON string

    # Relationships
    scans = db.relationship('Scan', backref='patient', lazy='dynamic', cascade='all, delete-orphan')

    @property
    def phone(self):
        return decrypt_data(self._phone) if self._phone else None

    @phone.setter
    def phone(self, value):
        self._phone = encrypt_data(value) if value else None

    @property
    def date_of_birth(self):
        return decrypt_data(self._date_of_birth) if self._date_of_birth else None

    @date_of_birth.setter
    def date_of_birth(self, value):
        self._date_of_birth = encrypt_data(value) if value else None

    @property
    def address(self):
        return decrypt_data(self._address) if self._address else None

    @address.setter
    def address(self, value):
        self._address = encrypt_data(value) if value else None

    @property
    def medical_history(self):
        return decrypt_data(self._medical_history) if self._medical_history else None

    @medical_history.setter
    def medical_history(self, value):
        self._medical_history = encrypt_data(value) if value else None

    @property
    def emergency_contact(self):
        return decrypt_data(self._emergency_contact) if self._emergency_contact else None

    @emergency_contact.setter
    def emergency_contact(self, value):
        self._emergency_contact = encrypt_data(value) if value else None

    def __repr__(self):
        return f'<Patient {self.user.username}>'


class Doctor(db.Model):
    """Doctor profile"""
    __tablename__ = 'doctors'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)

    specialization = db.Column(db.String(100), nullable=False)
    license_number = db.Column(db.String(50), unique=True, nullable=False)

    # Encrypted fields
    _phone = db.Column('phone', db.LargeBinary, nullable=True)

    hospital_affiliation = db.Column(db.String(200), nullable=True)
    years_experience = db.Column(db.Integer, nullable=True)
    digital_signature = db.Column(db.Text, nullable=True)  # Base64 encoded signature image
    verification_status = db.Column(db.Boolean, default=False)
    verification_date = db.Column(db.DateTime, nullable=True)

    # Relationships
    diagnoses = db.relationship('Diagnosis', backref='doctor', lazy='dynamic')
    treatments = db.relationship('Treatment', backref='doctor', lazy='dynamic')

    @property
    def phone(self):
        return decrypt_data(self._phone) if self._phone else None

    @phone.setter
    def phone(self, value):
        self._phone = encrypt_data(value) if value else None

    def __repr__(self):
        return f'<Dr. {self.user.username} - {self.specialization}>'


class Scan(db.Model):
    """MRI Scan records"""
    __tablename__ = 'scans'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)

    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed

    # Metadata
    image_metadata = db.Column(db.Text, nullable=True)  # JSON string
    dicom_tags = db.Column(db.Text, nullable=True)  # JSON string for DICOM files
    file_size = db.Column(db.Integer, nullable=True)
    file_hash = db.Column(db.String(64), nullable=True)  # SHA-256 for integrity

    # Relationships
    diagnosis = db.relationship('Diagnosis', backref='scan', uselist=False, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Scan {self.id} - Patient {self.patient_id}>'


class Diagnosis(db.Model):
    """AI Diagnosis results with doctor verification"""
    __tablename__ = 'diagnoses'

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('scans.id'), nullable=False, unique=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=True)

    # AI predictions
    ai_prediction = db.Column(db.String(50), nullable=False)  # glioma/meningioma/pituitary/normal
    ai_confidence = db.Column(db.Float, nullable=False)
    model_predictions = db.Column(db.Text, nullable=False)  # JSON with all 3 model outputs
    ensemble_confidence = db.Column(db.Float, nullable=False)

    # Doctor verification
    doctor_verified = db.Column(db.Boolean, default=False)
    doctor_notes = db.Column(db.Text, nullable=True)
    doctor_diagnosis = db.Column(db.String(50), nullable=True)
    verification_date = db.Column(db.DateTime, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    treatment = db.relationship('Treatment', backref='diagnosis', uselist=False, cascade='all, delete-orphan')
    report = db.relationship('Report', backref='diagnosis', uselist=False, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Diagnosis {self.id} - {self.ai_prediction}>'


class Treatment(db.Model):
    """Treatment plan prescribed by doctor (CRITICAL FEATURE)"""
    __tablename__ = 'treatments'

    id = db.Column(db.Integer, primary_key=True)
    diagnosis_id = db.Column(db.Integer, db.ForeignKey('diagnoses.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)

    # Severity levels (CRITICAL FEATURE)
    severity_level = db.Column(db.String(20), nullable=False)  # critical, urgent_surgery_needed, normal_monitoring

    # Treatment details (encrypted)
    _prescription = db.Column('prescription', db.LargeBinary, nullable=True)
    _recommended_therapies = db.Column('recommended_therapies', db.LargeBinary, nullable=True)
    _lifestyle_modifications = db.Column('lifestyle_modifications', db.LargeBinary, nullable=True)

    follow_up_date = db.Column(db.DateTime, nullable=True)
    referred_specialist = db.Column(db.String(200), nullable=True)
    additional_notes = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def prescription(self):
        return decrypt_data(self._prescription) if self._prescription else None

    @prescription.setter
    def prescription(self, value):
        self._prescription = encrypt_data(value) if value else None

    @property
    def recommended_therapies(self):
        return decrypt_data(self._recommended_therapies) if self._recommended_therapies else None

    @recommended_therapies.setter
    def recommended_therapies(self, value):
        self._recommended_therapies = encrypt_data(json.dumps(value)) if value else None

    def get_recommended_therapies_list(self):
        if self.recommended_therapies:
            return json.loads(self.recommended_therapies)
        return []

    @property
    def recommended_therapies_list(self):
        return self.get_recommended_therapies_list()

    @property
    def lifestyle_modifications(self):
        return decrypt_data(self._lifestyle_modifications) if self._lifestyle_modifications else None

    @lifestyle_modifications.setter
    def lifestyle_modifications(self, value):
        self._lifestyle_modifications = encrypt_data(json.dumps(value)) if value else None

    def get_lifestyle_modifications_list(self):
        if self.lifestyle_modifications:
            return json.loads(self.lifestyle_modifications)
        return []

    @property
    def lifestyle_modifications_list(self):
        return self.get_lifestyle_modifications_list()


class Report(db.Model):
    """Generated PDF reports"""
    __tablename__ = 'reports'

    id = db.Column(db.Integer, primary_key=True)
    diagnosis_id = db.Column(db.Integer, db.ForeignKey('diagnoses.id'), nullable=False)
    pdf_path = db.Column(db.String(500), nullable=False)
    generated_date = db.Column(db.DateTime, default=datetime.utcnow)

    digital_signature_hash = db.Column(db.String(256), nullable=True)
    access_count = db.Column(db.Integer, default=0)
    last_accessed = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f'<Report for Diagnosis {self.diagnosis_id}>'


class Notification(db.Model):
    """User notifications"""
    __tablename__ = 'notifications'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    type = db.Column(db.String(20), nullable=False)  # email, sms, in_app
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)

    related_entity_type = db.Column(db.String(50), nullable=True)  # scan, diagnosis, treatment
    related_entity_id = db.Column(db.Integer, nullable=True)

    read_status = db.Column(db.Boolean, default=False)
    sent_status = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sent_at = db.Column(db.DateTime, nullable=True)
    delivered_at = db.Column(db.DateTime, nullable=True)


class AuditLog(db.Model):
    """Complete audit trail for security compliance"""
    __tablename__ = 'audit_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    entity_type = db.Column(db.String(50), nullable=True)
    entity_id = db.Column(db.Integer, nullable=True)

    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(200), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    details = db.Column(db.Text, nullable=True)  # JSON string

    def __repr__(self):
        return f'<AuditLog {self.action} at {self.timestamp}>'
