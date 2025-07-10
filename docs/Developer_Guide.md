# Developer Guide: Visual-Text Matching System

## Overview

This guide is intended for developers who want to contribute to, extend, or integrate the Visual-Text Matching System. It covers the internal architecture, development setup, testing strategies, and contribution guidelines.

## Table of Contents
- [Development Environment Setup](#development-environment-setup)
- [Project Architecture](#project-architecture)
- [Core Components](#core-components)
- [Adding New Features](#adding-new-features)
- [Testing](#testing)
- [Performance Optimization](#performance-optimization)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## Development Environment Setup

### Prerequisites

1. **Python Environment**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

2. **Development Tools**:
```bash
# Install development tools
pip install pytest black flake8 mypy pre-commit jupyter
```

3. **Pre-commit Hooks**:
```bash
# Setup pre-commit hooks
pre-commit install
```

### Project Structure

```
visual-text-matcher/
├── src/
│   ├── visual_text_matcher/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py
│   │   │   ├── cnn_model.py
│   │   │   ├── lstm_model.py
│   │   │   └── gru_model.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── preprocessors.py
│   │   │   ├── datasets.py
│   │   │   └── augmentation.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py
│   │   │   ├── losses.py
│   │   │   └── metrics.py
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── api.py
│   │   │   └── utils.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── config.py
│   │       └── exceptions.py
├── tests/
├── docs/
├── examples/
├── scripts/
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── README.md
```

---

## Project Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Layer   │    │  Processing     │    │  Output Layer   │
│                 │    │     Layer       │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Image Input  │ │────│ │Image        │ │    │ │Similarity   │ │
│ │Preprocessing│ │    │ │Encoder      │ │    │ │Score        │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Text Input   │ │────│ │Text         │ │────│ │Match        │ │
│ │Preprocessing│ │    │ │Encoder      │ │    │ │Decision     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Input Processing**:
   - Images: Resize → Normalize → Tensor conversion
   - Text: Tokenize → Character encoding → Tensor conversion

2. **Feature Extraction**:
   - Image features: CNN/Vision Transformer
   - Text features: Character-level embeddings + Sequential model

3. **Similarity Computation**:
   - Cosine similarity between embeddings
   - Contrastive distance metrics

---

## Core Components

### Base Model Interface

All models must inherit from `BaseModel`:

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Base class for all visual-text matching models."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode image to feature vector."""
        pass
    
    @abstractmethod
    def encode_text(self, text_tensor: torch.Tensor) -> torch.Tensor:
        """Encode text to feature vector."""
        pass
    
    def forward(self, image_tensor: torch.Tensor, text_tensor: torch.Tensor) -> torch.Tensor:
        """Compute similarity between image and text."""
        image_features = self.encode_image(image_tensor)
        text_features = self.encode_text(text_tensor)
        
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(image_features * text_features, dim=1)
        return similarity
```

### Model Implementation Example

```python
class CNNTextMatcher(BaseModel):
    """CNN-based visual-text matcher."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Image encoder (CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, config.embedding_dim)
        )
        
        # Text encoder (Character-level CNN)
        self.char_embedding = nn.Embedding(
            config.vocab_size, 
            config.char_embedding_dim
        )
        self.text_encoder = nn.Sequential(
            nn.Conv1d(config.char_embedding_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, config.embedding_dim)
        )
    
    def encode_image(self, image_tensor):
        return self.image_encoder(image_tensor)
    
    def encode_text(self, text_tensor):
        # text_tensor shape: (batch_size, sequence_length)
        embedded = self.char_embedding(text_tensor)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        return self.text_encoder(embedded)
```

### Configuration Management

```python
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    
    # Model architecture
    model_type: str = "lstm"
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout_rate: float = 0.1
    
    # Text processing
    max_sequence_length: int = 50
    vocab_size: int = 128  # ASCII characters
    char_embedding_dim: int = 64
    
    # Image processing
    image_size: Tuple[int, int] = (128, 32)
    image_channels: int = 3
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_epochs: int = 100
    
    # Loss function
    loss_type: str = "contrastive"  # or "triplet"
    margin: float = 1.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
```

---

## Adding New Features

### Adding a New Model Architecture

1. **Create the model class**:
```python
# src/visual_text_matcher/models/transformer_model.py
class TransformerTextMatcher(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Implementation...
    
    def encode_image(self, image_tensor):
        # Vision Transformer implementation
        pass
    
    def encode_text(self, text_tensor):
        # Text Transformer implementation
        pass
```

2. **Register the model**:
```python
# src/visual_text_matcher/models/__init__.py
from .transformer_model import TransformerTextMatcher

MODEL_REGISTRY = {
    'cnn': CNNTextMatcher,
    'lstm': LSTMTextMatcher,
    'gru': GRUTextMatcher,
    'transformer': TransformerTextMatcher,  # Add new model
}

def create_model(config):
    model_class = MODEL_REGISTRY.get(config.model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {config.model_type}")
    return model_class(config)
```

3. **Add tests**:
```python
# tests/test_transformer_model.py
def test_transformer_model():
    config = ModelConfig(model_type="transformer")
    model = create_model(config)
    
    # Test forward pass
    batch_size = 4
    image_tensor = torch.randn(batch_size, 3, 32, 128)
    text_tensor = torch.randint(0, 128, (batch_size, 20))
    
    similarities = model(image_tensor, text_tensor)
    assert similarities.shape == (batch_size,)
```

### Adding New Loss Functions

```python
# src/visual_text_matcher/training/losses.py
class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        ce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Register the loss function
LOSS_REGISTRY = {
    'contrastive': ContrastiveLoss,
    'triplet': TripletLoss,
    'focal': FocalLoss,  # Add new loss
}
```

### Adding Data Augmentation

```python
# src/visual_text_matcher/data/augmentation.py
class AdvancedImageAugmentation:
    """Advanced image augmentation techniques."""
    
    def __init__(self, config):
        self.transforms = self._build_transforms(config)
    
    def _build_transforms(self, config):
        transforms = []
        
        if config.use_rotation:
            transforms.append(
                torchvision.transforms.RandomRotation(degrees=5)
            )
        
        if config.use_perspective:
            transforms.append(
                torchvision.transforms.RandomPerspective(
                    distortion_scale=0.1, p=0.5
                )
            )
        
        if config.use_noise:
            transforms.append(self._add_noise)
        
        return torchvision.transforms.Compose(transforms)
    
    def _add_noise(self, image):
        """Add Gaussian noise to image."""
        noise = torch.randn_like(image) * 0.05
        return torch.clamp(image + noise, 0, 1)
    
    def __call__(self, image):
        return self.transforms(image)
```

---

## Testing

### Unit Testing Strategy

```python
# tests/conftest.py
import pytest
import torch
from visual_text_matcher.utils.config import ModelConfig

@pytest.fixture
def sample_config():
    return ModelConfig(
        model_type="lstm",
        embedding_dim=128,
        hidden_dim=256,
        max_sequence_length=20
    )

@pytest.fixture
def sample_data():
    batch_size = 4
    return {
        'images': torch.randn(batch_size, 3, 32, 128),
        'texts': torch.randint(0, 128, (batch_size, 20)),
        'labels': torch.randint(0, 2, (batch_size,)).float()
    }
```

### Model Testing

```python
# tests/test_models.py
import pytest
from visual_text_matcher.models import create_model

@pytest.mark.parametrize("model_type", ["cnn", "lstm", "gru"])
def test_model_forward_pass(model_type, sample_config, sample_data):
    """Test forward pass for all model types."""
    sample_config.model_type = model_type
    model = create_model(sample_config)
    
    similarities = model(sample_data['images'], sample_data['texts'])
    
    assert similarities.shape == (4,)
    assert torch.all(similarities >= -1) and torch.all(similarities <= 1)

def test_model_gradient_flow(sample_config, sample_data):
    """Test that gradients flow correctly."""
    model = create_model(sample_config)
    loss_fn = ContrastiveLoss()
    
    similarities = model(sample_data['images'], sample_data['texts'])
    loss = loss_fn(similarities, sample_data['labels'])
    loss.backward()
    
    # Check that gradients are computed
    for param in model.parameters():
        assert param.grad is not None
```

### Integration Testing

```python
# tests/test_integration.py
def test_end_to_end_training():
    """Test complete training pipeline."""
    config = ModelConfig(num_epochs=2, batch_size=8)
    
    # Create synthetic dataset
    dataset = create_synthetic_dataset(size=100)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    
    # Initialize components
    model = create_model(config)
    trainer = Trainer(model, dataloader, config)
    
    # Train for few epochs
    history = trainer.train()
    
    assert len(history['train_loss']) == config.num_epochs
    assert history['train_loss'][-1] < history['train_loss'][0]  # Loss should decrease
```

### Performance Testing

```python
# tests/test_performance.py
import time

def test_inference_speed():
    """Test inference speed requirements."""
    config = ModelConfig()
    model = create_model(config)
    model.eval()
    
    # Single image inference
    image = torch.randn(1, 3, 32, 128)
    text = torch.randint(0, 128, (1, 20))
    
    start_time = time.time()
    with torch.no_grad():
        _ = model(image, text)
    end_time = time.time()
    
    inference_time = end_time - start_time
    assert inference_time < 0.1  # Should be faster than 100ms

def test_memory_usage():
    """Test memory usage stays within bounds."""
    config = ModelConfig(batch_size=32)
    model = create_model(config)
    
    initial_memory = torch.cuda.memory_allocated()
    
    # Forward pass
    images = torch.randn(32, 3, 32, 128).cuda()
    texts = torch.randint(0, 128, (32, 20)).cuda()
    _ = model(images, texts)
    
    peak_memory = torch.cuda.memory_allocated()
    memory_usage = peak_memory - initial_memory
    
    # Should use less than 500MB for batch of 32
    assert memory_usage < 500 * 1024 * 1024
```

---

## Performance Optimization

### Profiling and Benchmarking

```python
# scripts/profile_model.py
import torch.profiler
from visual_text_matcher.models import create_model

def profile_model():
    config = ModelConfig()
    model = create_model(config).cuda()
    
    images = torch.randn(32, 3, 32, 128).cuda()
    texts = torch.randint(0, 128, (32, 20)).cuda()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(100):
            _ = model(images, texts)
    
    print(prof.key_averages().table(sort_by="cuda_time_total"))

if __name__ == "__main__":
    profile_model()
```

### Optimization Techniques

1. **Model Quantization**:
```python
# scripts/quantize_model.py
def quantize_model(model_path, output_path):
    model = torch.load(model_path)
    model.eval()
    
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    torch.save(quantized_model, output_path)
```

2. **ONNX Export**:
```python
# scripts/export_onnx.py
def export_to_onnx(model_path, onnx_path):
    model = torch.load(model_path)
    model.eval()
    
    dummy_image = torch.randn(1, 3, 32, 128)
    dummy_text = torch.randint(0, 128, (1, 20))
    
    torch.onnx.export(
        model,
        (dummy_image, dummy_text),
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['image', 'text'],
        output_names=['similarity']
    )
```

3. **TensorRT Optimization**:
```python
# scripts/tensorrt_optimize.py
def convert_to_tensorrt(onnx_path, engine_path):
    import tensorrt as trt
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    
    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(onnx_path)
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    engine = builder.build_engine(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

---

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose API port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "visual_text_matcher.api.server"]
```

### API Server

```python
# src/visual_text_matcher/api/server.py
from fastapi import FastAPI, UploadFile, HTTPException
from visual_text_matcher.inference.api import MatchingAPI

app = FastAPI(title="Visual-Text Matching API")

# Initialize model
matching_api = MatchingAPI("models/pretrained_model.pth")

@app.post("/match")
async def match_image_text(
    image: UploadFile,
    text: str,
    threshold: float = 0.8
):
    """Match uploaded image with text."""
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{image.filename}"
        with open(temp_path, "wb") as f:
            f.write(await image.read())
        
        # Perform matching
        result = matching_api.match_single(temp_path, text, threshold)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: visual-text-matcher
spec:
  replicas: 3
  selector:
    matchLabels:
      app: visual-text-matcher
  template:
    metadata:
      labels:
        app: visual-text-matcher
    spec:
      containers:
      - name: api
        image: visual-text-matcher:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        env:
        - name: MODEL_PATH
          value: "/app/models/pretrained_model.pth"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

---

## Contributing

### Code Style

1. **Use Black for formatting**:
```bash
black src/ tests/
```

2. **Follow PEP 8 guidelines**:
```bash
flake8 src/ tests/
```

3. **Type hints are required**:
```python
def process_image(image_path: str, config: ModelConfig) -> torch.Tensor:
    """Process image with proper type hints."""
    pass
```

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Write tests** for your changes
4. **Run the test suite**: `pytest tests/`
5. **Update documentation** if needed
6. **Submit a pull request** with a clear description

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are written and passing
- [ ] Documentation is updated
- [ ] Performance impact is considered
- [ ] Backward compatibility is maintained
- [ ] Security implications are reviewed

### Release Process

1. **Update version** in `setup.py`
2. **Update CHANGELOG.md**
3. **Create release tag**: `git tag v1.0.0`
4. **Build and publish**: `python setup.py sdist bdist_wheel`
5. **Deploy to PyPI**: `twine upload dist/*`

---

## Debugging and Troubleshooting

### Common Development Issues

1. **CUDA out of memory**:
```python
# Reduce batch size or use gradient accumulation
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

2. **Model not converging**:
```python
# Add debugging prints
def debug_forward_pass(model, batch):
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            print(f"{name}: weight norm = {module.weight.norm()}")
```

3. **Slow training**:
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = model(images, texts)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Logging Configuration

```python
# src/visual_text_matcher/utils/logging.py
import logging
import sys

def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
```

This developer guide provides a comprehensive foundation for contributing to and extending the Visual-Text Matching System. It covers all aspects from setup to deployment, ensuring maintainable and scalable development practices.