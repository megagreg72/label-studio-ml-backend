import os
import logging
import boto3
from typing import List, Dict, Optional
from urllib.parse import urlparse

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)


class AWSRekognitionObjectDetector(LabelStudioMLBase):
    """
    Label Studio ML Backend for AWS Rekognition Custom Labels object detection.
    
    This backend connects to a custom AWS Rekognition project for object detection
    and returns predictions in Label Studio format.
    """

    def setup(self):
        """Configure AWS Rekognition client and model parameters"""
        
        # AWS Rekognition configuration
        self.project_arn = os.getenv(
            'REKOGNITION_PROJECT_ARN',
            'arn:aws:rekognition:us-east-1:030179562632:project/sdi-object-detector/version/sdi-object-detector.2025-12-11T19.13.51/1765502031650'
        )
        
        # Extract region from ARN
        arn_parts = self.project_arn.split(':')
        self.region = arn_parts[3] if len(arn_parts) > 3 else 'us-east-1'
        
        # Initialize AWS Rekognition client
        self.rekognition_client = boto3.client(
            'rekognition',
            region_name=self.region,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Initialize S3 client for fetching images
        self.s3_client = boto3.client(
            's3',
            region_name=self.region,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Confidence threshold for predictions
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', '50'))
        
        # Set model version
        self.set("model_version", "aws-rekognition-1.0.0")
        
        logger.info(f"AWS Rekognition ML Backend initialized")
        logger.info(f"Project ARN: {self.project_arn}")
        logger.info(f"Region: {self.region}")
        logger.info(f"Min Confidence: {self.min_confidence}")

    def _get_image_bytes(self, url: str) -> bytes:
        """
        Fetch image bytes from URL or S3.
        
        Args:
            url: Image URL (can be http/https or s3://)
            
        Returns:
            Image bytes
        """
        parsed_url = urlparse(url)
        
        if parsed_url.scheme == 's3':
            # S3 URL
            bucket = parsed_url.netloc
            key = parsed_url.path.lstrip('/')
            logger.info(f"Fetching image from S3: bucket={bucket}, key={key}")
            
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        
        elif parsed_url.scheme in ['http', 'https']:
            # HTTP URL - download locally first
            import requests
            logger.info(f"Downloading image from URL: {url}")
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        
        else:
            # Local file path
            logger.info(f"Reading local image file: {url}")
            with open(url, 'rb') as f:
                return f.read()

    def _detect_custom_labels(self, image_bytes: bytes) -> List[Dict]:
        """
        Call AWS Rekognition DetectCustomLabels API.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            List of detected custom labels
        """
        try:
            response = self.rekognition_client.detect_custom_labels(
                ProjectVersionArn=self.project_arn,
                Image={'Bytes': image_bytes},
                MinConfidence=self.min_confidence
            )
            
            return response.get('CustomLabels', [])
        
        except Exception as e:
            logger.error(f"Error calling AWS Rekognition: {str(e)}")
            raise

    def _convert_to_label_studio_format(
        self, 
        detections: List[Dict], 
        image_width: int, 
        image_height: int,
        from_name: str,
        to_name: str
    ) -> List[Dict]:
        """
        Convert AWS Rekognition detections to Label Studio format.
        
        Args:
            detections: List of detections from Rekognition
            image_width: Original image width
            image_height: Original image height
            from_name: Label control tag name
            to_name: Image object tag name
            
        Returns:
            List of predictions in Label Studio format
        """
        results = []
        
        for detection in detections:
            # Get bounding box (if available)
            if 'Geometry' in detection and 'BoundingBox' in detection['Geometry']:
                bbox = detection['Geometry']['BoundingBox']
                
                # Rekognition returns normalized coordinates (0-1)
                # Label Studio expects percentages (0-100)
                x = bbox['Left'] * 100
                y = bbox['Top'] * 100
                width = bbox['Width'] * 100
                height = bbox['Height'] * 100
                
                result = {
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "rectanglelabels",
                    "value": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "rectanglelabels": [detection['Name']]
                    },
                    "score": detection['Confidence'] / 100.0  # Convert to 0-1 range
                }
                
                results.append(result)
        
        return results

    def _get_label_config_params(self) -> tuple:
        """
        Extract label configuration parameters.
        
        Returns:
            Tuple of (from_name, to_name, labels)
        """
        # Get the first RectangleLabels control tag
        for control in self.label_interface.controls:
            if control.tag == 'RectangleLabels':
                from_name = control.name
                to_name = control.to_name[0] if control.to_name else None
                labels = [label.value for label in control.labels] if hasattr(control, 'labels') else []
                return from_name, to_name, labels
        
        # Default fallback
        return "label", "image", []

    def predict(
        self, 
        tasks: List[Dict], 
        context: Optional[Dict] = None, 
        **kwargs
    ) -> ModelResponse:
        """
        Run AWS Rekognition predictions on tasks.
        
        Args:
            tasks: List of Label Studio tasks in JSON format
            context: Optional context dictionary
            **kwargs: Additional parameters
            
        Returns:
            ModelResponse with predictions
        """
        predictions = []
        
        # Get label config parameters
        from_name, to_name, labels = self._get_label_config_params()
        
        logger.info(f"Processing {len(tasks)} tasks")
        logger.info(f"Label config: from_name={from_name}, to_name={to_name}")
        
        for task in tasks:
            # Get image URL from task data
            image_url = None
            
            if 'data' in task:
                # Try common image field names
                for field in ['image', 'img', 'image_url']:
                    if field in task['data']:
                        image_url = task['data'][field]
                        break
            
            if not image_url:
                logger.warning(f"No image URL found in task: {task.get('id', 'unknown')}")
                predictions.append({
                    "result": [],
                    "score": 0,
                    "model_version": str(self.model_version)
                })
                continue
            
            try:
                # Get image from URL
                logger.info(f"Processing image: {image_url}")
                
                # For Label Studio local file storage, resolve the path
                if not image_url.startswith(('http://', 'https://', 's3://')):
                    image_url = self.get_local_path(
                        image_url,
                        task_id=task.get('id')
                    )
                
                image_bytes = self._get_image_bytes(image_url)
                
                # Get image dimensions (for Label Studio coordinate conversion)
                from PIL import Image
                import io
                pil_image = Image.open(io.BytesIO(image_bytes))
                image_width, image_height = pil_image.size
                
                # Call AWS Rekognition
                detections = self._detect_custom_labels(image_bytes)
                logger.info(f"Found {len(detections)} detections")
                
                # Convert to Label Studio format
                result = self._convert_to_label_studio_format(
                    detections,
                    image_width,
                    image_height,
                    from_name,
                    to_name
                )
                
                predictions.append({
                    "result": result,
                    "score": max([d['Confidence'] / 100.0 for d in detections], default=0),
                    "model_version": str(self.model_version)
                })
                
            except Exception as e:
                logger.error(f"Error processing task {task.get('id', 'unknown')}: {str(e)}")
                predictions.append({
                    "result": [],
                    "score": 0,
                    "model_version": str(self.model_version)
                })
        
        return ModelResponse(predictions=predictions)
