# User Guide: Visual-Text Matching System

## Introduction

The Visual-Text Matching System is designed to compare word images with text strings for exact character matching, without any semantic understanding. This guide will help you get started with using the system effectively.

## Table of Contents
- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Getting Started

### Installation

1. **Install Python dependencies**:
```bash
pip install torch torchvision opencv-python pillow numpy matplotlib
```

2. **Clone and setup the project**:
```bash
git clone <repository-url>
cd visual-text-matcher
pip install -r requirements.txt
```

3. **Verify installation**:
```python
from visual_text_matcher import TextImageMatcher
print("Installation successful!")
```

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended)
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space

---

## Basic Usage

### Quick Start Example

```python
from visual_text_matcher import MatchingAPI

# Initialize the API with a pretrained model
api = MatchingAPI("models/pretrained_model.pth")

# Compare a word image with text
result = api.match_single(
    image_path="examples/hello_word.png",
    text="hello",
    threshold=0.8
)

print(f"Match: {result['match']}")
print(f"Similarity: {result['similarity']:.3f}")
```

### Step-by-Step Tutorial

#### 1. Preparing Your Images

Word images should be:
- Clear and readable
- Preferably on a light background
- Single words (no phrases)
- Common formats: PNG, JPG, JPEG

```python
from visual_text_matcher import ImagePreprocessor

preprocessor = ImagePreprocessor(
    target_size=(128, 32),  # Width x Height
    normalize=True
)

# Check if image is suitable
tensor = preprocessor.preprocess("your_word_image.png")
print(f"Image tensor shape: {tensor.shape}")
```

#### 2. Basic Matching

```python
# Single comparison
score = api.match_single("word_image.png", "target_text")

# Batch comparison
image_paths = ["img1.png", "img2.png", "img3.png"]
texts = ["hello", "world", "test"]
results = api.match_batch(list(zip(image_paths, texts)))

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['similarity']:.3f}")
```

#### 3. Finding Best Matches

```python
# Find best match from candidates
candidates = ["hello", "hell", "help", "held", "hero"]
best_match = api.find_best_match("word_image.png", candidates)

print(f"Best match: '{best_match['text']}'")
print(f"Similarity: {best_match['similarity']:.3f}")
```

---

## Advanced Features

### Custom Model Configuration

```python
from visual_text_matcher import TextImageMatcher, Config

# Create custom configuration
config = Config(
    model_type="lstm",          # or "cnn", "gru"
    max_sequence_length=30,
    hidden_dim=512,
    learning_rate=0.0005
)

# Initialize model with custom config
matcher = TextImageMatcher(config=config)
```

### Batch Processing for Large Datasets

```python
import os
from pathlib import Path

def process_image_directory(image_dir, target_text):
    """Process all images in a directory"""
    results = []
    image_paths = list(Path(image_dir).glob("*.png"))
    
    # Process in batches of 32
    batch_size = 32
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_texts = [target_text] * len(batch_paths)
        
        batch_results = api.match_batch(
            list(zip(batch_paths, batch_texts))
        )
        results.extend(batch_results)
    
    return results

# Usage
results = process_image_directory("path/to/images", "hello")
```

### Performance Monitoring

```python
import time

def benchmark_performance(image_path, text, iterations=100):
    """Benchmark matching performance"""
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        result = api.match_single(image_path, text)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"Average processing time: {avg_time*1000:.2f}ms")
    return avg_time

# Usage
benchmark_performance("test_image.png", "test")
```

---

## Common Use Cases

### 1. Document Verification

```python
def verify_document_fields(document_images, expected_values):
    """Verify specific fields in document images"""
    verification_results = {}
    
    for field_name, (image_path, expected_text) in document_images.items():
        result = api.match_single(image_path, expected_text, threshold=0.9)
        verification_results[field_name] = {
            'verified': result['match'],
            'confidence': result['similarity']
        }
    
    return verification_results

# Usage
document_fields = {
    'name': ('extracted_name.png', 'John Doe'),
    'id_number': ('extracted_id.png', '123456789'),
    'address': ('extracted_address.png', '123 Main St')
}

results = verify_document_fields(document_fields)
```

### 2. OCR Quality Assessment

```python
def assess_ocr_quality(image_path, ocr_text):
    """Assess how well OCR extracted text matches the image"""
    result = api.match_single(image_path, ocr_text)
    
    quality_score = result['similarity']
    if quality_score >= 0.95:
        quality = "Excellent"
    elif quality_score >= 0.85:
        quality = "Good"
    elif quality_score >= 0.70:
        quality = "Fair"
    else:
        quality = "Poor"
    
    return {
        'quality': quality,
        'score': quality_score,
        'needs_review': quality_score < 0.85
    }
```

### 3. Handwriting vs Typed Text Classification

```python
def classify_text_type(image_path, known_text):
    """Classify if text in image is handwritten or typed"""
    result = api.match_single(image_path, known_text)
    
    # High similarity suggests typed text (exact match)
    # Lower similarity might indicate handwriting variations
    if result['similarity'] >= 0.95:
        return "typed"
    elif result['similarity'] >= 0.80:
        return "printed"
    else:
        return "handwritten"
```

### 4. Brand Logo/Text Verification

```python
def verify_brand_text(logo_image, brand_name):
    """Verify brand text in logo images"""
    # Try different case variations
    variations = [
        brand_name.lower(),
        brand_name.upper(),
        brand_name.title()
    ]
    
    best_match = None
    best_score = 0
    
    for variation in variations:
        result = api.match_single(logo_image, variation)
        if result['similarity'] > best_score:
            best_score = result['similarity']
            best_match = variation
    
    return {
        'brand_verified': best_score >= 0.9,
        'best_match': best_match,
        'confidence': best_score
    }
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Low Similarity Scores for Correct Text

**Symptoms**: Getting low scores (< 0.5) even when text matches the image.

**Solutions**:
1. **Check image quality**:
   ```python
   # Ensure image is clear and high contrast
   preprocessor = ImagePreprocessor(normalize=True)
   tensor = preprocessor.preprocess("image.png")
   ```

2. **Verify text preprocessing**:
   ```python
   # Check for extra spaces or special characters
   clean_text = text.strip().lower()
   result = api.match_single(image_path, clean_text)
   ```

3. **Adjust threshold**:
   ```python
   # Use lower threshold for noisy images
   result = api.match_single(image_path, text, threshold=0.6)
   ```

#### Issue: Memory Errors with Large Batches

**Solutions**:
```python
# Reduce batch size
def process_large_dataset(image_paths, texts, batch_size=16):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_images = image_paths[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        batch_results = api.match_batch(list(zip(batch_images, batch_texts)))
        results.extend(batch_results)
    return results
```

#### Issue: Slow Processing Speed

**Solutions**:
1. **Use GPU acceleration**:
   ```python
   # Initialize with GPU
   api = MatchingAPI("model.pth", device="cuda")
   ```

2. **Enable batch processing**:
   ```python
   # Process multiple images at once
   results = api.match_batch(pairs)  # Instead of multiple single calls
   ```

### Error Messages and Fixes

| Error | Cause | Solution |
|-------|-------|----------|
| `InvalidImageFormatError` | Unsupported image format | Convert to PNG/JPG |
| `TextTooLongError` | Text exceeds max length | Truncate text or increase model's max_sequence_length |
| `ModelNotTrainedError` | Using untrained model | Load pretrained weights or train model |
| `CUDA out of memory` | Batch too large for GPU | Reduce batch size |

---

## Best Practices

### Image Preparation

1. **Optimal Image Characteristics**:
   - Resolution: 128x32 pixels (width x height)
   - Format: PNG with transparent background preferred
   - Text color: Dark text on light background
   - Font: Clear, non-decorative fonts work best

2. **Image Quality Tips**:
   ```python
   # Check image quality before processing
   def check_image_quality(image_path):
       img = cv2.imread(image_path)
       if img is None:
           return False, "Cannot read image"
       
       height, width = img.shape[:2]
       if width < 50 or height < 20:
           return False, "Image too small"
       
       return True, "Good quality"
   ```

### Performance Optimization

1. **Batch Processing**:
   ```python
   # Good: Process in batches
   results = api.match_batch(pairs)
   
   # Avoid: Multiple single calls
   # results = [api.match_single(img, txt) for img, txt in pairs]
   ```

2. **Caching for Repeated Queries**:
   ```python
   import functools
   
   @functools.lru_cache(maxsize=128)
   def cached_match(image_path, text):
       return api.match_single(image_path, text)
   ```

### Threshold Selection

1. **Use Case Specific Thresholds**:
   - **Exact matching required**: 0.95+
   - **High precision needed**: 0.85-0.95
   - **Balance precision/recall**: 0.75-0.85
   - **High recall needed**: 0.60-0.75

2. **Dynamic Threshold Adjustment**:
   ```python
   def adaptive_threshold(image_quality_score):
       if image_quality_score >= 0.9:
           return 0.9
       elif image_quality_score >= 0.7:
           return 0.8
       else:
           return 0.7
   ```

### Model Selection Guidelines

| Use Case | Recommended Model | Rationale |
|----------|------------------|-----------|
| Short words (< 10 chars) | CNN | Fast, good for simple patterns |
| Medium words (10-20 chars) | LSTM | Balance of speed and accuracy |
| Long text (20+ chars) | GRU | Better for longer sequences |
| High-noise images | Ensemble | Multiple models for robustness |

### Monitoring and Logging

```python
import logging

# Setup logging for production use
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_match(image_path, text):
    start_time = time.time()
    result = api.match_single(image_path, text)
    end_time = time.time()
    
    logger.info(f"Matched '{text}' with similarity {result['similarity']:.3f} "
                f"in {(end_time - start_time)*1000:.2f}ms")
    
    return result
```

---

## Next Steps

After mastering the basics:

1. **Explore Training**: Learn to train custom models for your specific use case
2. **Integration**: Integrate the API into your applications
3. **Optimization**: Fine-tune performance for your specific requirements
4. **Monitoring**: Set up production monitoring and logging

For advanced topics, see:
- [Developer Guide](Developer_Guide.md)
- [Architecture Documentation](Architecture.md)
- [Training Tutorial](Training_Guide.md)