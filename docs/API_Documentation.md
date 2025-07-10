# API Documentation

## Overview
This document provides comprehensive documentation for all public APIs, functions, and components in the Visual-Text Matching project.

## Table of Contents
- [Core Models](#core-models)
- [Data Processing](#data-processing)
- [Training Components](#training-components)
- [Inference APIs](#inference-apis)
- [Utilities](#utilities)
- [Configuration](#configuration)

---

## Core Models

### TextImageMatcher
**Purpose**: Main model class for exact visual-text matching without semantic understanding.

#### Constructor
```python
TextImageMatcher(
    encoder_type: str = "cnn",  # "cnn", "rnn", "lstm", "gru"
    max_sequence_length: int = 50,
    character_embedding_dim: int = 128,
    hidden_dim: int = 256,
    dropout_rate: float = 0.1
)
```

**Parameters**:
- `encoder_type`: Type of encoder architecture to use
- `max_sequence_length`: Maximum length of text sequences
- `character_embedding_dim`: Dimension of character embeddings
- `hidden_dim`: Hidden layer dimensions
- `dropout_rate`: Dropout rate for regularization

**Example Usage**:
```python
from visual_text_matcher import TextImageMatcher

# Initialize model
matcher = TextImageMatcher(
    encoder_type="lstm",
    max_sequence_length=30,
    hidden_dim=512
)

# Load pretrained weights
matcher.load_weights("path/to/model.pth")

# Compare image and text
similarity_score = matcher.compare(image_path="word_image.png", text="hello")
```

#### Methods

##### `compare(image_path: str, text: str) -> float`
Compares a word image with a text string for exact character matching.

**Parameters**:
- `image_path`: Path to the word image file
- `text`: Text string to compare against

**Returns**: Similarity score (0.0 to 1.0) where 1.0 indicates exact match

**Example**:
```python
score = matcher.compare("hello_image.png", "hello")  # Expected: ~1.0
score = matcher.compare("hello_image.png", "helo")   # Expected: ~0.0
```

##### `batch_compare(image_paths: List[str], texts: List[str]) -> List[float]`
Performs batch comparison for multiple image-text pairs.

**Parameters**:
- `image_paths`: List of paths to word image files
- `texts`: List of text strings to compare

**Returns**: List of similarity scores

---

## Data Processing

### ImagePreprocessor
**Purpose**: Handles preprocessing of word images for model input.

#### Constructor
```python
ImagePreprocessor(
    target_size: Tuple[int, int] = (128, 32),
    normalize: bool = True,
    augment: bool = False
)
```

#### Methods

##### `preprocess(image_path: str) -> torch.Tensor`
Preprocesses a single image for model input.

**Example**:
```python
preprocessor = ImagePreprocessor(target_size=(128, 32))
tensor = preprocessor.preprocess("word_image.png")
```

### TextPreprocessor
**Purpose**: Handles preprocessing of text strings for model input.

#### Methods

##### `encode_text(text: str) -> torch.Tensor`
Converts text to character-level tensor representation.

##### `decode_text(tensor: torch.Tensor) -> str`
Converts tensor back to text string.

---

## Training Components

### ContrastiveLoss
**Purpose**: Implements contrastive loss for training the exact matching model.

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        self.margin = margin
    
    def forward(self, 
                image_embeddings: torch.Tensor,
                text_embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        # Implementation details...
```

### TripletLoss
**Purpose**: Implements triplet loss as an alternative training objective.

### Trainer
**Purpose**: Handles the training loop and validation.

#### Constructor
```python
Trainer(
    model: TextImageMatcher,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str = "cuda"
)
```

#### Methods

##### `train(epochs: int, save_path: str = None) -> Dict[str, List[float]]`
Trains the model for specified number of epochs.

**Returns**: Dictionary containing training and validation losses

**Example**:
```python
trainer = Trainer(model, train_loader, val_loader, optimizer, loss_fn)
history = trainer.train(epochs=100, save_path="model_checkpoint.pth")
```

---

## Inference APIs

### MatchingAPI
**Purpose**: High-level API for inference and deployment.

#### Constructor
```python
MatchingAPI(model_path: str, device: str = "cpu")
```

#### Methods

##### `match_single(image_path: str, text: str, threshold: float = 0.8) -> Dict`
Performs single image-text matching with confidence threshold.

**Returns**:
```python
{
    "match": bool,          # True if similarity > threshold
    "similarity": float,    # Similarity score
    "confidence": float,    # Model confidence
    "processing_time": float # Time taken in seconds
}
```

##### `match_batch(pairs: List[Tuple[str, str]]) -> List[Dict]`
Performs batch matching for multiple pairs.

##### `find_best_match(image_path: str, candidate_texts: List[str]) -> Dict`
Finds the best matching text from a list of candidates.

**Example**:
```python
api = MatchingAPI("trained_model.pth")

# Single match
result = api.match_single("word.png", "hello", threshold=0.9)

# Find best match from candidates
candidates = ["hello", "help", "world", "held"]
best_match = api.find_best_match("word.png", candidates)
```

---

## Utilities

### DatasetBuilder
**Purpose**: Utilities for creating training datasets.

#### Methods

##### `create_synthetic_dataset(words: List[str], fonts: List[str], output_dir: str)`
Creates synthetic word images for training.

##### `create_contrastive_pairs(dataset_path: str) -> List[Tuple]`
Creates positive and negative pairs for contrastive learning.

### Metrics
**Purpose**: Evaluation metrics for model performance.

#### Functions

##### `exact_match_accuracy(predictions: List[float], labels: List[int], threshold: float) -> float`
Calculates exact match accuracy given predictions and labels.

##### `precision_recall_curve(predictions: List[float], labels: List[int]) -> Tuple`
Generates precision-recall curve data.

---

## Configuration

### Config
**Purpose**: Configuration management for the entire system.

```python
@dataclass
class Config:
    # Model configuration
    model_type: str = "lstm"
    max_sequence_length: int = 50
    embedding_dim: int = 128
    hidden_dim: int = 256
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    
    # Data configuration
    image_size: Tuple[int, int] = (128, 32)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
```

**Usage**:
```python
from config import Config

config = Config(
    model_type="cnn",
    batch_size=64,
    learning_rate=0.0005
)
```

---

## Error Handling

### Common Exceptions

#### `InvalidImageFormatError`
Raised when image format is not supported.

#### `TextTooLongError`
Raised when input text exceeds maximum sequence length.

#### `ModelNotTrainedError`
Raised when attempting inference with untrained model.

**Example Error Handling**:
```python
try:
    score = matcher.compare("image.png", "text")
except InvalidImageFormatError:
    print("Unsupported image format")
except TextTooLongError:
    print("Text too long for model")
```

---

## Installation and Setup

### Requirements
```bash
pip install torch torchvision opencv-python pillow numpy
```

### Quick Start
```python
# 1. Install dependencies
pip install -r requirements.txt

# 2. Import and initialize
from visual_text_matcher import TextImageMatcher, MatchingAPI

# 3. Load pretrained model or train new one
matcher = TextImageMatcher()
# ... training code ...

# 4. Use for inference
api = MatchingAPI("trained_model.pth")
result = api.match_single("word_image.png", "target_text")
```

---

## Performance Considerations

### Memory Usage
- Image preprocessing: ~10MB per batch of 32 images
- Model inference: ~50MB GPU memory for LSTM model
- Batch processing recommended for large datasets

### Speed Benchmarks
- Single image comparison: ~10ms on GPU, ~50ms on CPU
- Batch of 100 comparisons: ~100ms on GPU, ~2s on CPU

### Optimization Tips
- Use GPU acceleration when available
- Batch process multiple comparisons
- Cache preprocessed images for repeated use
- Use appropriate image resolution (128x32 recommended)

---

## Contributing

### Adding New Model Architectures
1. Inherit from `BaseModel` class
2. Implement required methods: `forward()`, `encode_image()`, `encode_text()`
3. Add configuration options to `Config` class
4. Update this documentation

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run performance benchmarks
python benchmarks/run_benchmarks.py
```