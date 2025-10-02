import os
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import threading
import time

import pyrealsense2 as rs
import numpy as np
from threading import Thread
import copy
import cv2

from .realsense_reader import RealSenseReader


class RealSenseReaderServer(RealSenseReader):
    def __init__(self, camera_config, server_url=None, upload_interval=1.0, render_depth=False):
        """
        Extended RealSenseReader that also uploads images to a server.
        
        Args:
            camera_config: Camera configuration dictionary
            server_url: URL of the server to upload images to (e.g., "http://localhost:8000/upload")
            upload_interval: Interval in seconds between uploads
            render_depth: Whether to render depth images
        """
        super().__init__(camera_config, render_depth)
        
        self.server_url = server_url
        self.upload_interval = upload_interval
        self.upload_thread = None
        self.upload_shutdown = False
        
        if self.server_url:
            self.start_upload_thread()
    
    def start_upload_thread(self):
        """Start the upload thread."""
        if self.upload_thread is None:
            self.upload_thread = Thread(target=self._upload_loop)
            self.upload_thread.daemon = True
            self.upload_thread.start()
    
    def _upload_loop(self):
        """Main upload loop that runs in a separate thread."""
        while not self.upload_shutdown:
            try:
                self._upload_images()
                time.sleep(self.upload_interval)
            except Exception as e:
                print(f"Error in upload loop: {e}")
                time.sleep(1.0)  # Wait before retrying
    
    def _upload_images(self):
        """Upload current images to the server."""
        if not self.server_url or not self.images:
            return
        
        try:
            # Prepare data for upload
            upload_data = {
                'timestamp': time.time(),
                'cameras': {}
            }
            
            for name, camera_data in self.images.items():
                if 'color' in camera_data:
                    # Convert color image to base64
                    color_img = camera_data['color']
                    color_img_pil = Image.fromarray(color_img)
                    color_buffer = BytesIO()
                    color_img_pil.save(color_buffer, format='JPEG', quality=85)
                    color_base64 = base64.b64encode(color_buffer.getvalue()).decode('utf-8')
                    
                    camera_upload_data = {
                        'color': color_base64,
                        'timestamp': camera_data.get('timestamp', time.time())
                    }
                    
                    # Add depth data if available
                    if 'depth' in camera_data:
                        depth_img = camera_data['depth']
                        # Normalize depth for visualization
                        depth_normalized = np.clip(depth_img / 1000.0, 0, 3.0) / 3.0 * 255
                        depth_img_pil = Image.fromarray(depth_normalized.astype(np.uint8))
                        depth_buffer = BytesIO()
                        depth_img_pil.save(depth_buffer, format='PNG')
                        depth_base64 = base64.b64encode(depth_buffer.getvalue()).decode('utf-8')
                        
                        camera_upload_data['depth'] = depth_base64
                        camera_upload_data['depth_units'] = camera_data.get('depth_units', 0.001)
                    
                    upload_data['cameras'][name] = camera_upload_data
            
            # Send HTTP POST request
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                self.server_url,
                data=json.dumps(upload_data),
                headers=headers,
                timeout=5.0
            )
            
            if response.status_code != 200:
                print(f"Upload failed with status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Network error during upload: {e}")
        except Exception as e:
            print(f"Error preparing upload data: {e}")
    
    def get_status(self):
        """Get current camera status (inherited from parent)."""
        return super().get_status()
    
    def render(self):
        """Render images (inherited from parent)."""
        return super().render()
    
    def shutdown(self):
        """Shutdown the reader and upload thread."""
        self.upload_shutdown = True
        self.shutdown = True
        
        if self.upload_thread and self.upload_thread.is_alive():
            self.upload_thread.join(timeout=2.0)
        
        # Call parent shutdown if it exists
        if hasattr(super(), 'shutdown'):
            super().shutdown()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.shutdown()


# Example server endpoint handler (for reference)
def create_server_endpoint():
    """
    Example Flask server endpoint to receive uploaded images.
    You can use this as a reference for implementing your server.
    
    from flask import Flask, request, jsonify
    import base64
    from PIL import Image
    import io
    
    app = Flask(__name__)
    
    @app.route('/upload', methods=['POST'])
    def upload_images():
        try:
            data = request.json
            timestamp = data.get('timestamp')
            cameras = data.get('cameras', {})
            
            # Process each camera's data
            for camera_name, camera_data in cameras.items():
                if 'color' in camera_data:
                    # Decode color image
                    color_base64 = camera_data['color']
                    color_bytes = base64.b64decode(color_base64)
                    color_image = Image.open(io.BytesIO(color_bytes))
                    
                    # Save or process the image as needed
                    # color_image.save(f'uploads/{camera_name}_{timestamp}.jpg')
                
                if 'depth' in camera_data:
                    # Decode depth image
                    depth_base64 = camera_data['depth']
                    depth_bytes = base64.b64decode(depth_base64)
                    depth_image = Image.open(io.BytesIO(depth_bytes))
                    
                    # Process depth data as needed
            
            return jsonify({'status': 'success', 'message': 'Images uploaded successfully'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8000, debug=True)
    """
    pass


if __name__ == "__main__":
    import cv2
    import time
    from omegaconf import OmegaConf
    
    # Example usage
    camera_config = OmegaConf.load("configs/ros_control/real_papras_7dof_2arm_table.yaml")
    
    # Create reader with server upload
    reader = RealSenseReaderServer(
        camera_config.cameras,
        server_url="http://localhost:8000/upload",  # Replace with your server URL
        upload_interval=1.0  # Upload every second
    )
    
    try:
        while True:
            # Display images locally
            view_im = reader.render()
            if view_im is not None:
                cv2.imshow("RealSense with Server Upload", view_im[:,:,::-1])
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        reader.shutdown()
        cv2.destroyAllWindows() 