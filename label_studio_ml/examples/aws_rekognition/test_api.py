import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from model import AWSRekognitionObjectDetector


@pytest.fixture
def mock_label_config():
    """Sample label configuration for testing"""
    return """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="car" background="red"/>
        <Label value="person" background="blue"/>
      </RectangleLabels>
    </View>
    """


@pytest.fixture
def mock_task():
    """Sample task for testing"""
    return {
        "id": 1,
        "data": {
            "image": "https://example.com/test.jpg"
        }
    }


@pytest.fixture
def mock_rekognition_response():
    """Mock AWS Rekognition response"""
    return {
        'CustomLabels': [
            {
                'Name': 'car',
                'Confidence': 95.5,
                'Geometry': {
                    'BoundingBox': {
                        'Left': 0.1,
                        'Top': 0.2,
                        'Width': 0.3,
                        'Height': 0.4
                    }
                }
            },
            {
                'Name': 'person',
                'Confidence': 87.3,
                'Geometry': {
                    'BoundingBox': {
                        'Left': 0.5,
                        'Top': 0.6,
                        'Width': 0.2,
                        'Height': 0.3
                    }
                }
            }
        ]
    }


@pytest.fixture
def model_instance(mock_label_config):
    """Create a model instance with mocked AWS clients"""
    with patch('boto3.client') as mock_boto:
        # Mock the boto3 clients
        mock_rekognition = Mock()
        mock_s3 = Mock()
        
        def client_selector(service_name, **kwargs):
            if service_name == 'rekognition':
                return mock_rekognition
            elif service_name == 's3':
                return mock_s3
            return Mock()
        
        mock_boto.side_effect = client_selector
        
        model = AWSRekognitionObjectDetector(
            project_id='test_project',
            label_config=mock_label_config
        )
        
        # Attach mocked clients for use in tests
        model.rekognition_client = mock_rekognition
        model.s3_client = mock_s3
        
        return model


def test_model_initialization(model_instance):
    """Test that the model initializes correctly"""
    assert model_instance is not None
    assert model_instance.project_arn is not None
    assert model_instance.min_confidence == 50.0
    assert model_instance.region == 'us-east-1'


def test_get_image_bytes_http(model_instance):
    """Test fetching image from HTTP URL"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.content = b'fake_image_data'
        mock_get.return_value = mock_response
        
        result = model_instance._get_image_bytes('https://example.com/test.jpg')
        
        assert result == b'fake_image_data'
        mock_get.assert_called_once()


def test_get_image_bytes_s3(model_instance):
    """Test fetching image from S3"""
    mock_body = Mock()
    mock_body.read.return_value = b's3_image_data'
    
    model_instance.s3_client.get_object.return_value = {
        'Body': mock_body
    }
    
    result = model_instance._get_image_bytes('s3://my-bucket/path/to/image.jpg')
    
    assert result == b's3_image_data'
    model_instance.s3_client.get_object.assert_called_once_with(
        Bucket='my-bucket',
        Key='path/to/image.jpg'
    )


def test_detect_custom_labels(model_instance, mock_rekognition_response):
    """Test calling AWS Rekognition API"""
    model_instance.rekognition_client.detect_custom_labels.return_value = mock_rekognition_response
    
    result = model_instance._detect_custom_labels(b'fake_image')
    
    assert len(result) == 2
    assert result[0]['Name'] == 'car'
    assert result[0]['Confidence'] == 95.5
    
    model_instance.rekognition_client.detect_custom_labels.assert_called_once()


def test_convert_to_label_studio_format(model_instance, mock_rekognition_response):
    """Test conversion from Rekognition format to Label Studio format"""
    detections = mock_rekognition_response['CustomLabels']
    
    result = model_instance._convert_to_label_studio_format(
        detections,
        image_width=1000,
        image_height=800,
        from_name='label',
        to_name='image'
    )
    
    assert len(result) == 2
    
    # Check first detection (car)
    assert result[0]['type'] == 'rectanglelabels'
    assert result[0]['from_name'] == 'label'
    assert result[0]['to_name'] == 'image'
    assert result[0]['value']['rectanglelabels'] == ['car']
    assert result[0]['value']['x'] == 10.0  # 0.1 * 100
    assert result[0]['value']['y'] == 20.0  # 0.2 * 100
    assert result[0]['value']['width'] == 30.0  # 0.3 * 100
    assert result[0]['value']['height'] == 40.0  # 0.4 * 100
    assert result[0]['score'] == pytest.approx(0.955, rel=1e-3)


def test_predict_success(model_instance, mock_task, mock_rekognition_response):
    """Test successful prediction"""
    # Mock the image fetching and PIL
    with patch.object(model_instance, '_get_image_bytes', return_value=b'fake_image'), \
         patch('PIL.Image.open') as mock_pil:
        
        # Mock PIL Image
        mock_image = Mock()
        mock_image.size = (1000, 800)
        mock_pil.return_value = mock_image
        
        # Mock Rekognition response
        model_instance.rekognition_client.detect_custom_labels.return_value = mock_rekognition_response
        
        # Run prediction
        response = model_instance.predict([mock_task])
        
        # Check response structure
        assert response is not None
        predictions = response.predictions if hasattr(response, 'predictions') else response
        
        assert len(predictions) == 1
        assert 'result' in predictions[0]
        assert 'score' in predictions[0]
        assert len(predictions[0]['result']) == 2


def test_predict_no_image_url(model_instance):
    """Test prediction with missing image URL"""
    task_no_image = {"id": 1, "data": {}}
    
    response = model_instance.predict([task_no_image])
    predictions = response.predictions if hasattr(response, 'predictions') else response
    
    assert len(predictions) == 1
    assert predictions[0]['result'] == []
    assert predictions[0]['score'] == 0


def test_predict_error_handling(model_instance, mock_task):
    """Test error handling in prediction"""
    with patch.object(model_instance, '_get_image_bytes', side_effect=Exception('Network error')):
        response = model_instance.predict([mock_task])
        predictions = response.predictions if hasattr(response, 'predictions') else response
        
        assert len(predictions) == 1
        assert predictions[0]['result'] == []
        assert predictions[0]['score'] == 0


def test_get_label_config_params(model_instance):
    """Test extraction of label config parameters"""
    from_name, to_name, labels = model_instance._get_label_config_params()
    
    assert from_name == 'label'
    assert to_name == 'image'
    # Labels may or may not be populated depending on label_interface implementation
    assert isinstance(labels, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
