# ============================================
# FILE: app/reporting/pdf_generator.py
# ============================================
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.barcode import qr
from datetime import datetime
import os
from flask import current_app
import json

from app.utils.confidence_display import diagnosis_headline_confidence


class PDFGenerator:
    """Generates professional medical PDF reports"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#3498db'),
            spaceBefore=20,
            spaceAfter=10,
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5,
            borderRadius=5
        ))

        self.styles.add(ParagraphStyle(
            name='SeverityCritical',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.white,
            backColor=colors.HexColor('#dc3545'),
            alignment=TA_CENTER,
            spaceAfter=20,
            borderPadding=10
        ))

        self.styles.add(ParagraphStyle(
            name='SeverityUrgent',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.white,
            backColor=colors.HexColor('#fd7e14'),
            alignment=TA_CENTER,
            spaceAfter=20,
            borderPadding=10
        ))

        self.styles.add(ParagraphStyle(
            name='SeverityNormal',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.white,
            backColor=colors.HexColor('#28a745'),
            alignment=TA_CENTER,
            spaceAfter=20,
            borderPadding=10
        ))

        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER
        ))

    def generate_patient_report(self, diagnosis_id):
        """Generate comprehensive patient report"""
        from app import db
        from app.database.models import Diagnosis, Patient, Doctor, Treatment, Scan

        # Get data
        diagnosis = Diagnosis.query.get_or_404(diagnosis_id)
        scan = diagnosis.scan
        patient = scan.patient
        doctor = Doctor.query.get(diagnosis.doctor_id) if diagnosis.doctor_id else None
        treatment = Treatment.query.filter_by(diagnosis_id=diagnosis.id).first()

        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{patient.user.username}_{scan.id}_{timestamp}.pdf"

        # Ensure reports directory exists
        reports_dir = os.path.join(current_app.root_path, 'static', 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        filepath = os.path.join(reports_dir, filename)

        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Build story (content)
        story = []

        # Header
        story.extend(self._create_header(patient))

        # Severity Banner (if treatment exists)
        if treatment:
            severity_banner = self._create_severity_banner(treatment.severity_level)
            story.extend(severity_banner)

        # Patient Information
        story.extend(self._create_patient_info(patient))

        # Scan Information
        story.extend(self._create_scan_info(scan))

        # AI Analysis Results
        story.extend(self._create_ai_results(diagnosis))

        # Doctor Verification
        if doctor:
            story.extend(self._create_doctor_verification(doctor, diagnosis))

        # Treatment Plan
        if treatment:
            story.extend(self._create_treatment_plan(treatment))

        # Recommendations
        story.extend(self._create_recommendations(diagnosis, treatment))

        # Footer with QR code
        story.extend(self._create_footer(patient, scan))

        # Build PDF
        doc.build(story)

        return filepath

    def _create_header(self, patient):
        """Create report header with logo and title"""
        elements = []

        # Hospital Name
        elements.append(Paragraph(
            "BRAIN TUMOR DETECTION SYSTEM",
            self.styles['CustomTitle']
        ))

        elements.append(Paragraph(
            "Poornima University, Jaipur",
            self.styles['Normal']
        ))

        elements.append(Paragraph(
            f"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            self.styles['Normal']
        ))

        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_severity_banner(self, severity):
        """Create severity level banner"""
        elements = []

        severity_texts = {
            'critical': '⚠️ CRITICAL - IMMEDIATE ATTENTION REQUIRED ⚠️',
            'urgent_surgery_needed': '🚨 URGENT SURGERY NEEDED 🚨',
            'normal_monitoring': '✓ NORMAL - MONITORING REQUIRED ✓'
        }

        style_map = {
            'critical': self.styles['SeverityCritical'],
            'urgent_surgery_needed': self.styles['SeverityUrgent'],
            'normal_monitoring': self.styles['SeverityNormal']
        }

        text = severity_texts.get(severity, severity)
        style = style_map.get(severity, self.styles['SeverityNormal'])

        elements.append(Paragraph(text, style))
        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_patient_info(self, patient):
        """Create patient information table"""
        elements = []

        elements.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeader']))

        data = [
            ['Patient Name:', patient.user.username],
            ['Patient ID:', f"P{patient.id}"],
            ['Email:', patient.user.email],
            ['Phone:', patient.phone or 'Not provided'],
            ['Date of Birth:', patient.date_of_birth or 'Not provided'],
            ['Blood Group:', patient.blood_group or 'Not provided'],
        ]

        table = Table(data, colWidths=[2 * inch, 3 * inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.gray),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_scan_info(self, scan):
        """Create scan information section"""
        elements = []

        elements.append(Paragraph("SCAN INFORMATION", self.styles['SectionHeader']))

        data = [
            ['Scan ID:', f"#{scan.id}"],
            ['Upload Date:', scan.upload_date.strftime('%B %d, %Y at %I:%M %p')],
            ['Filename:', scan.filename],
            ['File Size:', f"{scan.file_size / (1024 * 1024):.2f} MB"],
        ]

        table = Table(data, colWidths=[2 * inch, 3 * inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.gray),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_ai_results(self, diagnosis):
        """Create AI analysis results section"""
        elements = []

        elements.append(Paragraph("AI ANALYSIS RESULTS", self.styles['SectionHeader']))

        # Main prediction
        prediction_color = colors.HexColor('#dc3545') if diagnosis.ai_prediction != 'normal' else colors.HexColor(
            '#28a745')

        data = [
            ['Final Prediction:', diagnosis.ai_prediction.upper()],
            ['Confidence:', f"{diagnosis_headline_confidence(diagnosis):.1f}%"],
        ]

        table = Table(data, colWidths=[2 * inch, 3 * inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TEXTCOLOR', (1, 0), (1, 0), prediction_color),
            ('TEXTCOLOR', (1, 1), (1, 1), colors.blue),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.1 * inch))

        # Model breakdown
        elements.append(Paragraph("Individual Model Predictions:", self.styles['Normal']))

        if diagnosis.model_predictions:
            models_data = json.loads(diagnosis.model_predictions)
            if isinstance(models_data, dict) and "model_details" in models_data:
                models_data = models_data.get("model_details") or {}
            data = [['Model', 'Prediction', 'Confidence']]

            for model_name, model_data in models_data.items():
                if not isinstance(model_data, dict) or "prediction" not in model_data:
                    continue
                model_display = {
                    'resnet50': 'ResNet50',
                    'mobilenetv2': 'MobileNetV2',
                    'custom_cnn': 'Custom CNN'
                }.get(model_name, model_name)

                data.append([
                    model_display,
                    model_data['prediction'].capitalize(),
                    f"{model_data['confidence']:.1f}%"
                ])

            table = Table(data, colWidths=[2 * inch, 2 * inch, 1.5 * inch])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (2, 0), (2, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))

            elements.append(table)

        elements.append(Spacer(1, 0.2 * inch))
        return elements

    def _create_doctor_verification(self, doctor, diagnosis):
        """Create doctor verification section"""
        elements = []

        elements.append(Paragraph("DOCTOR VERIFICATION", self.styles['SectionHeader']))

        data = [
            ['Verified By:', f"Dr. {doctor.user.username}"],
            ['Specialization:', doctor.specialization],
            ['License Number:', doctor.license_number],
            ['Verification Date:', diagnosis.verification_date.strftime(
                '%B %d, %Y at %I:%M %p') if diagnosis.verification_date else 'Pending'],
            ['Clinical Notes:', diagnosis.doctor_notes or 'No additional notes'],
        ]

        table = Table(data, colWidths=[1.5 * inch, 3.5 * inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.gray),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_treatment_plan(self, treatment):
        """Create treatment plan section"""
        elements = []

        elements.append(Paragraph("TREATMENT PLAN", self.styles['SectionHeader']))

        # Prescription
        if treatment.prescription:
            data = [['Prescription:', treatment.prescription]]
            table = Table(data, colWidths=[1.5 * inch, 3.5 * inch])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (0, 0), colors.gray),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(table)

        # Therapies
        if treatment.recommended_therapies:
            therapies = json.loads(treatment.recommended_therapies)
            elements.append(Paragraph("Recommended Therapies:", self.styles['Normal']))
            for therapy in therapies:
                elements.append(Paragraph(f"• {therapy}", self.styles['Normal']))

        # Lifestyle Modifications
        if treatment.lifestyle_modifications:
            modifications = json.loads(treatment.lifestyle_modifications)
            elements.append(Paragraph("Lifestyle Modifications:", self.styles['Normal']))
            for mod in modifications:
                elements.append(Paragraph(f"• {mod}", self.styles['Normal']))

        # Follow-up
        if treatment.follow_up_date:
            elements.append(Paragraph(
                f"Follow-up Date: {treatment.follow_up_date.strftime('%B %d, %Y at %I:%M %p')}",
                self.styles['Normal']
            ))

        # Specialist Referral
        if treatment.referred_specialist:
            elements.append(Paragraph(
                f"Referred to: {treatment.referred_specialist}",
                self.styles['Normal']
            ))

        # Additional Notes
        if treatment.additional_notes:
            elements.append(Paragraph(f"Notes: {treatment.additional_notes}", self.styles['Normal']))

        elements.append(Spacer(1, 0.2 * inch))
        return elements

    def _create_recommendations(self, diagnosis, treatment):
        """Create recommendations section"""
        elements = []

        elements.append(Paragraph("RECOMMENDATIONS", self.styles['SectionHeader']))

        if diagnosis.ai_prediction != 'normal':
            elements.append(Paragraph("• Immediate consultation with neuro-oncologist", self.styles['Normal']))
            elements.append(Paragraph("• Follow-up MRI with contrast within 2 weeks", self.styles['Normal']))
            elements.append(Paragraph("• Biopsy may be required for confirmation", self.styles['Normal']))

            if treatment and treatment.severity_level == 'critical':
                elements.append(Paragraph("• URGENT: Admit to hospital immediately", self.styles['Normal']))
        else:
            elements.append(Paragraph("• Regular monitoring as recommended by doctor", self.styles['Normal']))
            elements.append(Paragraph("• Follow-up scan in 6 months", self.styles['Normal']))
            elements.append(Paragraph("• Maintain healthy lifestyle", self.styles['Normal']))

        elements.append(Spacer(1, 0.2 * inch))
        return elements

    def _create_footer(self, patient, scan):
        """Create footer with QR code and confidentiality notice"""
        elements = []

        # QR Code (for secure link to online report)
        qr_data = f"https://braintumordetection.com/secure-report?pid={patient.id}&scan={scan.id}"
        qr_code = qr.QrCodeWidget(qr_data)
        qr_drawing = Drawing(1 * inch, 1 * inch)
        qr_drawing.add(qr_code)
        elements.append(qr_drawing)

        elements.append(Spacer(1, 0.1 * inch))

        # Confidentiality notice
        elements.append(Paragraph(
            "This report contains confidential medical information. "
            "If you are not the intended recipient, please destroy this document immediately.",
            self.styles['Footer']
        ))

        elements.append(Paragraph(
            f"Report ID: {patient.id}-{scan.id}-{datetime.now().strftime('%Y%m%d')} | Page 1 of 1",
            self.styles['Footer']
        ))

        return elements


# Singleton instance
_pdf_generator = None


def get_pdf_generator():
    """Get or create PDF generator instance"""
    global _pdf_generator
    if _pdf_generator is None:
        _pdf_generator = PDFGenerator()
    return _pdf_generator


def generate_patient_report(diagnosis_id):
    """Generate patient report (convenience function)"""
    generator = get_pdf_generator()
    return generator.generate_patient_report(diagnosis_id)
