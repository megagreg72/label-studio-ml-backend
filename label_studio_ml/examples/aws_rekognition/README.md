# AWS Rekognition Custom Labels ML Backend for Label Studio

This ML backend connects Label Studio to AWS Rekognition Custom Labels for object detection tasks.

## Prerequisites

1. AWS Account with Rekognition Custom Labels access
2. Trained AWS Rekognition Custom Labels model
3. AWS credentials (Access Key ID and Secret Access Key) or IAM role
4. Label Studio instance (local or cloud)

## Setup

### 1. Configure AWS Credentials

You can configure AWS credentials in several ways:

**Option A: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**Option B: AWS Credentials File** (~/.aws/credentials)
```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

**Option C: IAM Roles** (Recommended for production)
- Use IAM roles when running on AWS services (EC2, ECS, EKS)

### 2. Start the AWS Rekognition Model (if not already running)

Before using this backend, ensure your AWS Rekognition Custom Labels model is running:

```bash
# Start the model using AWS CLI
aws rekognition start-project-version \
    --project-version-arn "arn:aws:rekognition:us-east-1:030179562632:project/sdi-object-detector/version/sdi-object-detector.2025-12-11T19.13.51/1765502031650" \
    --min-inference-units 1
```

**Note:** Starting a model can take several minutes. Check the status:

```bash
aws rekognition describe-project-versions \
    --project-arn "arn:aws:rekognition:us-east-1:030179562632:project/sdi-object-detector" \
    --version-names "sdi-object-detector.2025-12-11T19.13.51"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-base.txt
```

### 4. Run the ML Backend

**Using Python directly:**
```bash
python _wsgi.py
```

**Using Docker:**
```bash
docker-compose up --build
```

The backend will start on port 9090 by default.

### 5. Configure Label Studio

1. Open your Label Studio project
2. Go to Settings → Machine Learning
3. Add ML Backend with URL: `http://localhost:9090`
4. Click "Validate and Save"

## Label Studio Configuration

Use a labeling configuration like this:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <!-- Add your custom labels here that match your Rekognition model -->
    <Label value="YourLabel1" background="red"/>
    <Label value="YourLabel2" background="blue"/>
    <!-- Add more labels as needed -->
  </RectangleLabels>
</View>
```

**Important:** The label values in Label Studio should match the class names from your AWS Rekognition Custom Labels model.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REKOGNITION_PROJECT_ARN` | ARN of your Rekognition Custom Labels project version | Required |
| `AWS_ACCESS_KEY_ID` | AWS Access Key ID | Required (unless using IAM) |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Access Key | Required (unless using IAM) |
| `AWS_DEFAULT_REGION` | AWS Region | `us-east-1` |
| `MIN_CONFIDENCE` | Minimum confidence threshold (0-100) | `50` |
| `LABEL_STUDIO_URL` | Label Studio instance URL | `http://localhost:8080` |
| `LABEL_STUDIO_API_KEY` | Label Studio API key | Optional |
| `PORT` | Port for the ML backend | `9090` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Usage

Once configured, the ML backend will:

1. Receive image data from Label Studio
2. Send the image to AWS Rekognition Custom Labels for inference
3. Convert the predictions to Label Studio format
4. Return bounding boxes with labels and confidence scores

The predictions will appear as pre-annotations in Label Studio, which annotators can review and correct.

## Cost Considerations

⚠️ **Important:** AWS Rekognition Custom Labels charges for:
- Training models
- Running inference on a deployed model (per hour)
- API calls

Make sure to **stop your model** when not in use to avoid unnecessary charges:

```bash
aws rekognition stop-project-version \
    --project-version-arn "your-project-version-arn"
```

## Troubleshooting

### Model Not Running
If you get errors about the model not being available, ensure it's in "RUNNING" state:
```bash
aws rekognition describe-project-versions \
    --project-arn "your-project-arn"
```

### Authentication Errors
- Verify your AWS credentials are correctly configured
- Ensure your IAM user/role has `rekognition:DetectCustomLabels` permission
- Check if the region is correct

### Image Format Issues
AWS Rekognition supports:
- JPEG
- PNG
- Maximum image size: 15MB
- Minimum dimensions: 64x64 pixels

### Connection Issues
- Verify the ML backend is running on port 9090
- Check firewall rules if running in Docker
- Ensure Label Studio can reach the ML backend URL

## Testing

Run the test suite:
```bash
pytest test_api.py
```

## Additional Resources

- [AWS Rekognition Custom Labels Documentation](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/)
- [Label Studio ML Backend Documentation](https://labelstud.io/guide/ml.html)
- [AWS Rekognition Pricing](https://aws.amazon.com/rekognition/pricing/)

## Support

For issues specific to:
- AWS Rekognition: Refer to AWS Support
- Label Studio: Check Label Studio documentation or GitHub issues
- This integration: Open an issue in this repository
