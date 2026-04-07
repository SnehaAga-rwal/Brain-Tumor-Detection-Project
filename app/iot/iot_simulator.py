# ============================================
# FILE: app/iot/iot_simulator.py
# ============================================
import threading
import time
import random
import json
import os
import logging
from datetime import datetime
from queue import Queue
from flask import current_app

logger = logging.getLogger(__name__)


class IoTSimulator:
    """
    Simulates IoT MRI devices sending images to the cloud
    with edge computing capabilities
    """

    def __init__(self, app=None):
        self.devices = {}
        self.running = False
        self.thread = None
        self.data_queue = Queue()
        self.app = app
        self.edge_cache = {}
        self.initialize_devices()

    def initialize_devices(self):
        """Create simulated IoT devices"""
        device_types = ['Siemens MRI', 'GE MRI', 'Philips MRI', 'Canon MRI']
        hospitals = ['City Hospital', 'University Medical Center', 'Community Health', 'Neuro Specialty Clinic']
        locations = ['Ward A', 'ICU', 'Radiology Dept', 'Emergency']

        for i in range(1, 11):  # Simulate 10 devices
            device_id = f"MRI-{str(i).zfill(3)}"
            self.devices[device_id] = {
                'id': device_id,
                'name': f"{random.choice(device_types)}-{i}",
                'hospital': random.choice(hospitals),
                'location': random.choice(locations),
                'status': random.choice(['online', 'online', 'online', 'idle', 'maintenance']),
                'last_seen': datetime.utcnow().isoformat(),
                'firmware': f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                'ip_address': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                'total_scans': random.randint(100, 5000),
                'error_rate': round(random.uniform(0.1, 2.0), 2),
                'edge_computing': random.choice([True, False])
            }

    def start(self):
        """Start the IoT simulator in background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_simulation)
            self.thread.daemon = True
            self.thread.start()
            logger.info("IoT Simulator started")

    def stop(self):
        """Stop the IoT simulator"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            logger.info("IoT Simulator stopped")

    def _run_simulation(self):
        """Main simulation loop"""
        while self.running:
            try:
                # Randomly select a device to send data
                device_id = random.choice(list(self.devices.keys()))
                device = self.devices[device_id]

                if device['status'] == 'online':
                    # Simulate image transmission
                    self._simulate_transmission(device_id)

                    # Update device stats
                    device['last_seen'] = datetime.utcnow().isoformat()
                    device['total_scans'] += 1

                # Random status changes
                if random.random() < 0.05:  # 5% chance
                    device['status'] = random.choice(['online', 'online', 'online', 'idle', 'maintenance'])

                # Sleep between 5-30 seconds
                time.sleep(random.randint(5, 30))

            except Exception as e:
                logger.error(f"IoT simulation error: {e}")

    def _simulate_transmission(self, device_id):
        """Simulate image transmission with edge computing"""
        device = self.devices[device_id]

        # Simulate network latency (50ms - 2s)
        latency = random.uniform(0.05, 2.0)
        time.sleep(latency)

        # Generate simulated image data
        image_data = self._generate_simulated_image(device)

        # Edge computing simulation
        if device.get('edge_computing', False):
            # Process at edge first
            edge_result = self._edge_processing(image_data)
            self.edge_cache[device_id] = edge_result

            # Send to cloud with edge result
            transmission = {
                'device_id': device_id,
                'device_name': device['name'],
                'hospital': device['hospital'],
                'timestamp': datetime.utcnow().isoformat(),
                'image_data': image_data,
                'edge_processed': True,
                'edge_result': edge_result,
                'latency_ms': round(latency * 1000, 2),
                'network_type': random.choice(['5G', 'WiFi6', 'Fiber', '4G']),
                'packet_loss': round(random.uniform(0, 0.5), 3)
            }
        else:
            # Direct cloud transmission
            transmission = {
                'device_id': device_id,
                'device_name': device['name'],
                'hospital': device['hospital'],
                'timestamp': datetime.utcnow().isoformat(),
                'image_data': image_data,
                'edge_processed': False,
                'latency_ms': round(latency * 1000, 2),
                'network_type': random.choice(['5G', 'WiFi6', 'Fiber', '4G']),
                'packet_loss': round(random.uniform(0, 0.5), 3)
            }

        # Add to queue for processing
        self.data_queue.put(transmission)

        # If Flask app context available, process the transmission
        if self.app:
            with self.app.app_context():
                self._process_transmission(transmission)

        logger.debug(f"Device {device_id} transmitted image with {latency * 1000:.0f}ms latency")

    def _generate_simulated_image(self, device):
        """Generate simulated image metadata"""
        tumor_types = ['glioma', 'meningioma', 'pituitary', 'normal']

        return {
            'image_id': f"IMG-{random.randint(10000, 99999)}",
            'patient_id': f"P{random.randint(1000, 9999)}",
            'modality': 'MRI',
            'sequence': random.choice(['T1', 'T2', 'FLAIR', 'DWI']),
            'tumor_type': random.choice(tumor_types),
            'size_kb': random.randint(500, 5000),
            'resolution': f"{random.choice([256, 512, 1024])}x{random.choice([256, 512, 1024])}",
            'contrast': random.choice([True, False]),
            'quality_score': round(random.uniform(0.7, 1.0), 2)
        }

    def _edge_processing(self, image_data):
        """Simulate edge computing processing"""
        # Simulate basic preprocessing at edge
        processing_time = random.uniform(0.1, 0.5)
        time.sleep(processing_time)  # Simulate processing delay

        return {
            'processed_at': datetime.utcnow().isoformat(),
            'processing_time_ms': round(processing_time * 1000, 2),
            'compression_ratio': round(random.uniform(0.5, 0.9), 2),
            'quality_check_passed': random.random() > 0.02,  # 98% pass rate
            'preliminary_analysis': {
                'tumor_detected': image_data['tumor_type'] != 'normal',
                'confidence': round(random.uniform(0.6, 0.95), 2)
            }
        }

    def _process_transmission(self, transmission):
        """Process the transmission (would trigger AI analysis)"""
        # In real implementation, this would queue for AI processing
        from app.database.models import Scan, Patient
        from app.models.ensemble_model import EnsemblePredictor

        try:
            # Simulate creating a scan record
            logger.info(f"Processing IoT transmission from {transmission['device_id']}")

            # Here you would:
            # 1. Create a scan record
            # 2. Trigger AI analysis
            # 3. Notify doctors if urgent

        except Exception as e:
            logger.error(f"Error processing IoT transmission: {e}")

    def get_device_status(self):
        """Get current status of all simulated devices"""
        devices_list = []
        for device_id, device in self.devices.items():
            device_copy = device.copy()
            # Add queue status
            device_copy['pending_transmissions'] = self.data_queue.qsize()
            device_copy['has_edge_cache'] = device_id in self.edge_cache
            devices_list.append(device_copy)
        return devices_list

    def control_device(self, device_id, action):
        """Control a simulated device"""
        if device_id in self.devices:
            device = self.devices[device_id]

            if action == 'start':
                device['status'] = 'online'
                return {'success': True, 'message': f"Device {device_id} started"}
            elif action == 'stop':
                device['status'] = 'maintenance'
                return {'success': True, 'message': f"Device {device_id} stopped"}
            elif action == 'reset':
                device['total_scans'] = 0
                device['error_rate'] = 0.1
                return {'success': True, 'message': f"Device {device_id} reset"}
            elif action == 'simulate_error':
                device['status'] = 'error'
                device['error_rate'] = 100
                return {'success': True, 'message': f"Error simulated on {device_id}"}

        return {'success': False, 'message': f"Device {device_id} not found"}


# Singleton instance
_simulator_instance = None


def get_simulator(app=None):
    """Get or create simulator instance"""
    global _simulator_instance
    if _simulator_instance is None and app:
        _simulator_instance = IoTSimulator(app)
    return _simulator_instance


def start_simulator(app):
    """Start the IoT simulator"""
    simulator = get_simulator(app)
    simulator.start()
    return simulator


def stop_simulator():
    """Stop the IoT simulator"""
    if _simulator_instance:
        _simulator_instance.stop()


def get_device_status():
    """Get device status (for API)"""
    if _simulator_instance:
        return _simulator_instance.get_device_status()
    return []


def control_device(device_id, action):
    """Control device (for API)"""
    if _simulator_instance:
        return _simulator_instance.control_device(device_id, action)
    return {'success': False, 'message': 'Simulator not initialized'}
