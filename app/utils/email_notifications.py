# ============================================
# FILE: app/utils/email_notifications.py (FIXED)
# ============================================
from flask import render_template, current_app
from flask_mail import Message
from app import mail
from app.utils.confidence_display import diagnosis_headline_confidence
from threading import Thread
from datetime import datetime
import logging


def send_async_email(app, msg):
    """Send email asynchronously"""
    with app.app_context():
        try:
            # Check if email is configured
            if app.config.get('MAIL_SUPPRESS_SEND', False):
                logging.info(f"Email suppressed: {msg.subject}")
                return

            mail.send(msg)
            logging.info(f"Email sent: {msg.subject}")
        except Exception as e:
            logging.error(f"Email sending failed: {e}")


def send_email(subject, recipients, text_body, html_body, sender=None):
    """Send email using Flask-Mail"""
    from flask import current_app

    # Check if email is configured
    if current_app.config.get('MAIL_SUPPRESS_SEND', False):
        logging.info(f"Email would be sent: {subject} to {recipients}")
        return

    # Get sender from config if not provided
    if not sender:
        sender = current_app.config.get('MAIL_USERNAME')
        if not sender:
            logging.error("No sender configured for email")
            return

    msg = Message(
        subject,
        sender=sender,
        recipients=recipients if isinstance(recipients, list) else [recipients]
    )
    msg.body = text_body
    msg.html = html_body

    # Send in background thread
    Thread(target=send_async_email, args=(current_app._get_current_object(), msg)).start()


# ============= PATIENT EMAIL TEMPLATES =============

def send_upload_confirmation(patient, scan):
    """Email when patient uploads MRI scan"""
    try:
        subject = f"MRI Scan Upload Confirmation - Scan #{scan.id}"

        text_body = f"""
        Dear {patient.user.username},

        Your MRI scan has been successfully uploaded to the Brain Tumor Detection System.

        Scan Details:
        - Scan ID: {scan.id}
        - Filename: {scan.filename}
        - Upload Date: {scan.upload_date.strftime('%Y-%m-%d %H:%M')}
        - Status: Pending Analysis

        Our AI system will now analyze your scan. You will receive another email when the analysis is complete.

        You can check the status anytime by logging into your patient portal.

        Thank you for using our service.

        Best regards,
        Brain Tumor Detection System Team
        """

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .header {{ background: #007bff; color: white; padding: 10px; text-align: center; border-radius: 5px 5px 0 0; }}
                .content {{ padding: 20px; }}
                .scan-details {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .footer {{ text-align: center; padding: 10px; color: #666; font-size: 0.9em; border-top: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>MRI Scan Upload Confirmation</h2>
                </div>
                <div class="content">
                    <p>Dear <strong>{patient.user.username}</strong>,</p>
                    <p>Your MRI scan has been successfully uploaded to the Brain Tumor Detection System.</p>

                    <div class="scan-details">
                        <h3>Scan Details:</h3>
                        <p><strong>Scan ID:</strong> {scan.id}</p>
                        <p><strong>Filename:</strong> {scan.filename}</p>
                        <p><strong>Upload Date:</strong> {scan.upload_date.strftime('%Y-%m-%d %H:%M')}</p>
                        <p><strong>Status:</strong> <span style="color: #ffc107;">Pending Analysis</span></p>
                    </div>

                    <p>Our AI system will now analyze your scan. You will receive another email when the analysis is complete.</p>

                    <p>Thank you for using our service.</p>
                    <p>Best regards,<br>Brain Tumor Detection System Team</p>
                </div>
                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """

        send_email(subject, patient.user.email, text_body, html_body)

        # Create notification record (avoid circular import)
        try:
            from app import db
            from app.database.models import Notification
            notification = Notification(
                user_id=patient.user.id,
                type='email',
                subject=subject,
                message=f"Scan #{scan.id} uploaded successfully",
                related_entity_type='scan',
                related_entity_id=scan.id,
                sent_status=True,
                sent_at=datetime.utcnow()
            )
            db.session.add(notification)
            db.session.commit()
        except Exception as e:
            logging.error(f"Error creating notification: {e}")

    except Exception as e:
        logging.error(f"Error in send_upload_confirmation: {e}")


def send_ai_analysis_complete(patient, scan, diagnosis):
    """Email when AI analysis is complete"""
    try:
        subject = f"AI Analysis Complete - Scan #{scan.id}"

        # Map prediction to readable format
        prediction_map = {
            'glioma': 'Glioma Tumor',
            'meningioma': 'Meningioma Tumor',
            'pituitary': 'Pituitary Tumor',
            'notumor': 'No Tumor Detected',
            'normal': 'No Tumor Detected'
        }

        prediction_text = prediction_map.get(diagnosis.ai_prediction, diagnosis.ai_prediction)
        hc = diagnosis_headline_confidence(diagnosis)

        text_body = f"""
        Dear {patient.user.username},

        The AI analysis of your MRI scan (Scan #{scan.id}) is now complete.

        Results:
        - AI Prediction: {prediction_text}
        - Confidence: {hc:.1f}%

        Next Steps:
        Your results have been sent to your doctor for verification.

        Best regards,
        Brain Tumor Detection System Team
        """

        # Color based on prediction
        color = '#dc3545' if diagnosis.ai_prediction not in ['notumor', 'normal'] else '#28a745'

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .header {{ background: {color}; color: white; padding: 10px; text-align: center; border-radius: 5px 5px 0 0; }}
                .content {{ padding: 20px; }}
                .result-box {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; text-align: center; }}
                .prediction {{ font-size: 24px; font-weight: bold; color: {color}; }}
                .confidence {{ font-size: 18px; margin: 10px 0; }}
                .footer {{ text-align: center; padding: 10px; color: #666; font-size: 0.9em; border-top: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>AI Analysis Complete</h2>
                </div>
                <div class="content">
                    <p>Dear <strong>{patient.user.username}</strong>,</p>
                    <p>The AI analysis of your MRI scan (Scan #{scan.id}) is now complete.</p>

                    <div class="result-box">
                        <h3>Analysis Results:</h3>
                        <div class="prediction">{prediction_text}</div>
                        <div class="confidence">Confidence: {hc:.1f}%</div>
                    </div>

                    <p>Your results have been sent to your doctor for verification.</p>

                    <p>Best regards,<br>Brain Tumor Detection System Team</p>
                </div>
                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """

        send_email(subject, patient.user.email, text_body, html_body)

        # Create notification
        try:
            from app import db
            from app.database.models import Notification
            notification = Notification(
                user_id=patient.user.id,
                type='email',
                subject=subject,
                message=f"AI analysis complete for scan #{scan.id}. Result: {prediction_text}",
                related_entity_type='diagnosis',
                related_entity_id=diagnosis.id,
                sent_status=True,
                sent_at=datetime.utcnow()
            )
            db.session.add(notification)
            db.session.commit()
        except Exception as e:
            logging.error(f"Error creating notification: {e}")

    except Exception as e:
        logging.error(f"Error in send_ai_analysis_complete: {e}")


def send_treatment_advice(patient, doctor, diagnosis, treatment):
    """Email when doctor provides treatment advice"""
    try:
        subject = f"Treatment Advice Available - Scan #{diagnosis.scan.id}"

        severity_map = {
            'critical': 'CRITICAL - Immediate Attention Required',
            'urgent_surgery_needed': 'URGENT - Surgery Needed',
            'normal_monitoring': 'Normal - Monitoring Required'
        }

        severity_text = severity_map.get(treatment.severity_level, treatment.severity_level)

        text_body = f"""
        Dear {patient.user.username},

        Dr. {doctor.user.username} has reviewed your MRI scan and provided treatment advice.

        Severity Level: {severity_text}

        Please log in to your patient portal to view the complete treatment plan.

        Best regards,
        Brain Tumor Detection System Team
        """

        send_email(subject, patient.user.email, text_body, text_body)

        # Create notification
        try:
            from app import db
            from app.database.models import Notification
            notification = Notification(
                user_id=patient.user.id,
                type='email',
                subject=subject,
                message=f"Treatment advice from Dr. {doctor.user.username} available. Severity: {severity_text}",
                related_entity_type='treatment',
                related_entity_id=treatment.id,
                sent_status=True,
                sent_at=datetime.utcnow()
            )
            db.session.add(notification)
            db.session.commit()
        except Exception as e:
            logging.error(f"Error creating notification: {e}")

    except Exception as e:
        logging.error(f"Error in send_treatment_advice: {e}")


def send_new_scan_to_doctor(doctor, patient, scan):
    """Email doctor when new scan is uploaded by their patient"""
    try:
        subject = f"New MRI Scan Ready for Review - Patient: {patient.user.username}"

        text_body = f"""
        Dear Dr. {doctor.user.username},

        A new MRI scan has been uploaded by your patient {patient.user.username} and is ready for your review.

        Scan ID: {scan.id}
        Upload Date: {scan.upload_date.strftime('%Y-%m-%d %H:%M')}

        Please log in to your doctor portal to review the case.

        Best regards,
        Brain Tumor Detection System
        """

        send_email(subject, doctor.user.email, text_body, text_body)

        # Create notification
        try:
            from app import db
            from app.database.models import Notification
            notification = Notification(
                user_id=doctor.user.id,
                type='email',
                subject=subject,
                message=f"New scan from patient {patient.user.username} ready for review",
                related_entity_type='scan',
                related_entity_id=scan.id,
                sent_status=True,
                sent_at=datetime.utcnow()
            )
            db.session.add(notification)
            db.session.commit()
        except Exception as e:
            logging.error(f"Error creating notification: {e}")

    except Exception as e:
        logging.error(f"Error in send_new_scan_to_doctor: {e}")


def send_urgent_case_alert(doctor, patient, diagnosis):
    """Email doctor for urgent/critical cases"""
    try:
        subject = f"URGENT: Critical Case Requires Immediate Attention - Patient: {patient.user.username}"

        _hc = diagnosis_headline_confidence(diagnosis)
        text_body = f"""
        URGENT - CRITICAL CASE

        Dear Dr. {doctor.user.username},

        An MRI scan flagged as POTENTIALLY CRITICAL requires your immediate attention.

        Patient: {patient.user.username}
        Scan ID: {diagnosis.scan.id}
        AI Prediction: {diagnosis.ai_prediction.upper()}
        Confidence: {_hc:.1f}%

        Please review immediately.

        Best regards,
        Brain Tumor Detection System
        """

        send_email(subject, doctor.user.email, text_body, text_body)

        # Create notification
        try:
            from app import db
            from app.database.models import Notification
            notification = Notification(
                user_id=doctor.user.id,
                type='email',
                subject=subject,
                message=f"URGENT: Critical case for patient {patient.user.username}",
                related_entity_type='diagnosis',
                related_entity_id=diagnosis.id,
                sent_status=True,
                sent_at=datetime.utcnow()
            )
            db.session.add(notification)
            db.session.commit()
        except Exception as e:
            logging.error(f"Error creating notification: {e}")

    except Exception as e:
        logging.error(f"Error in send_urgent_case_alert: {e}")