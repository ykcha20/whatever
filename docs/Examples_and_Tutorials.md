# Examples and Tutorials

## Overview

This document provides comprehensive examples and step-by-step tutorials for using the Visual-Text Matching System. Each example includes complete code, expected outputs, and explanations of the underlying concepts.

## Table of Contents
- [Basic Examples](#basic-examples)
- [Advanced Usage](#advanced-usage)
- [Training Tutorials](#training-tutorials)
- [Real-World Applications](#real-world-applications)
- [Integration Examples](#integration-examples)
- [Troubleshooting Examples](#troubleshooting-examples)

---

## Basic Examples

### Example 1: Simple Image-Text Matching

```python
"""
Basic example: Compare a word image with a text string
"""
from visual_text_matcher import MatchingAPI
import torch

# Initialize the API
api = MatchingAPI("models/pretrained_model.pth", device="cuda")

# Compare single image with text
result = api.match_single(
    image_path="examples/hello_world.png",
    text="hello",
    threshold=0.8
)

print(f"Match result: {result['match']}")
print(f"Similarity score: {result['similarity']:.4f}")
print(f"Processing time: {result['processing_time']:.2f}ms")

# Expected output:
# Match result: True
# Similarity score: 0.9234
# Processing time: 12.34ms
```

### Example 2: Batch Processing

```python
"""
Process multiple image-text pairs efficiently
"""
import os
from pathlib import Path

# Prepare batch data
image_dir = Path("examples/word_images")
image_paths = list(image_dir.glob("*.png"))
texts = ["hello", "world", "test", "example", "match"]

# Create pairs
pairs = list(zip(image_paths, texts))

# Process batch
results = api.match_batch(pairs)

# Display results
for i, (image_path, text) in enumerate(pairs):
    result = results[i]
    print(f"{image_path.name} vs '{text}': "
          f"similarity={result['similarity']:.3f}, "
          f"match={result['match']}")

# Expected output:
# hello.png vs 'hello': similarity=0.956, match=True
# world.png vs 'world': similarity=0.923, match=True
# test.png vs 'test': similarity=0.891, match=True
# example.png vs 'example': similarity=0.234, match=False
# match.png vs 'match': similarity=0.876, match=True
```

### Example 3: Finding Best Match

```python
"""
Find the best matching text from a list of candidates
"""

# Test image
image_path = "examples/hello_handwritten.png"

# Candidate texts
candidates = [
    "hello",
    "hallo",  # Similar but different
    "hell",   # Partial match
    "world",  # Different word
    "HELLO",  # Case variation
]

# Find best match
best_match = api.find_best_match(image_path, candidates)

print(f"Best match: '{best_match['text']}'")
print(f"Similarity: {best_match['similarity']:.4f}")
print(f"All scores:")
for candidate, score in best_match['all_scores'].items():
    print(f"  {candidate}: {score:.4f}")

# Expected output:
# Best match: 'hello'
# Similarity: 0.8976
# All scores:
#   hello: 0.8976
#   hallo: 0.7234
#   hell: 0.6789
#   world: 0.1234
#   HELLO: 0.8456
```

---

## Advanced Usage

### Example 4: Custom Configuration

```python
"""
Using custom model configuration for specific requirements
"""
from visual_text_matcher import TextImageMatcher, Config

# Create custom configuration
config = Config(
    model_type="lstm",
    max_sequence_length=40,  # Longer sequences
    hidden_dim=512,          # More capacity
    dropout_rate=0.2,        # More regularization
    embedding_dim=256,
    device="cuda"
)

# Initialize model with custom config
matcher = TextImageMatcher(config=config)

# Load pretrained weights (optional)
matcher.load_weights("models/custom_model.pth")

# Use the custom model
similarity = matcher.compare("long_text_image.png", "this is a longer text string")
print(f"Custom model similarity: {similarity:.4f}")
```

### Example 5: Performance Monitoring

```python
"""
Monitor performance and optimize for your use case
"""
import time
import statistics
from typing import List

class PerformanceMonitor:
    def __init__(self, api):
        self.api = api
        self.times = []
        self.similarities = []
    
    def benchmark_single_inference(self, image_path: str, text: str, iterations: int = 100):
        """Benchmark single inference performance"""
        print(f"Benchmarking {iterations} iterations...")
        
        for i in range(iterations):
            start_time = time.time()
            result = self.api.match_single(image_path, text)
            end_time = time.time()
            
            self.times.append((end_time - start_time) * 1000)  # Convert to ms
            self.similarities.append(result['similarity'])
        
        self.print_stats()
    
    def benchmark_batch_processing(self, pairs: List, batch_sizes: List[int]):
        """Compare different batch sizes"""
        print("Benchmarking batch processing:")
        
        for batch_size in batch_sizes:
            # Split pairs into batches
            batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
            
            start_time = time.time()
            results = []
            for batch in batches:
                batch_results = self.api.match_batch(batch)
                results.extend(batch_results)
            end_time = time.time()
            
            total_time = (end_time - start_time) * 1000
            per_item_time = total_time / len(pairs)
            
            print(f"  Batch size {batch_size}: "
                  f"{total_time:.2f}ms total, "
                  f"{per_item_time:.2f}ms per item")
    
    def print_stats(self):
        """Print performance statistics"""
        print(f"Performance Statistics:")
        print(f"  Mean time: {statistics.mean(self.times):.2f}ms")
        print(f"  Median time: {statistics.median(self.times):.2f}ms")
        print(f"  Min time: {min(self.times):.2f}ms")
        print(f"  Max time: {max(self.times):.2f}ms")
        print(f"  Std dev: {statistics.stdev(self.times):.2f}ms")
        print(f"  Mean similarity: {statistics.mean(self.similarities):.4f}")

# Usage
monitor = PerformanceMonitor(api)
monitor.benchmark_single_inference("test_image.png", "test_text")

# Expected output:
# Benchmarking 100 iterations...
# Performance Statistics:
#   Mean time: 12.34ms
#   Median time: 11.89ms
#   Min time: 10.23ms
#   Max time: 18.67ms
#   Std dev: 1.45ms
#   Mean similarity: 0.8923
```

### Example 6: Error Handling and Validation

```python
"""
Robust error handling and input validation
"""
from visual_text_matcher.utils.exceptions import (
    InvalidImageFormatError,
    TextTooLongError,
    ModelNotTrainedError
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustMatcher:
    def __init__(self, model_path: str):
        try:
            self.api = MatchingAPI(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def safe_match(self, image_path: str, text: str, threshold: float = 0.8):
        """Safely perform matching with comprehensive error handling"""
        try:
            # Validate inputs
            self._validate_inputs(image_path, text)
            
            # Perform matching
            result = self.api.match_single(image_path, text, threshold)
            logger.info(f"Successful match: {result['similarity']:.3f}")
            return result
            
        except InvalidImageFormatError as e:
            logger.error(f"Invalid image format: {e}")
            return {"error": "invalid_image_format", "message": str(e)}
            
        except TextTooLongError as e:
            logger.error(f"Text too long: {e}")
            return {"error": "text_too_long", "message": str(e)}
            
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return {"error": "file_not_found", "message": f"Image {image_path} not found"}
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": "unexpected", "message": str(e)}
    
    def _validate_inputs(self, image_path: str, text: str):
        """Validate input parameters"""
        import os
        
        # Check if image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check image format
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
            raise InvalidImageFormatError(f"Unsupported image format: {image_path}")
        
        # Check text length
        if len(text) > 50:  # Assuming max length of 50
            raise TextTooLongError(f"Text too long ({len(text)} chars), max 50 allowed")
        
        # Check for empty text
        if not text.strip():
            raise ValueError("Text cannot be empty")

# Usage example
matcher = RobustMatcher("models/pretrained_model.pth")

# Test various scenarios
test_cases = [
    ("valid_image.png", "hello"),
    ("nonexistent.png", "test"),
    ("valid_image.txt", "test"),  # Wrong format
    ("valid_image.png", "this is a very long text that exceeds the maximum allowed length"),
    ("valid_image.png", ""),  # Empty text
]

for image_path, text in test_cases:
    result = matcher.safe_match(image_path, text)
    if "error" in result:
        print(f"Error: {result['error']} - {result['message']}")
    else:
        print(f"Success: similarity = {result['similarity']:.3f}")
```

---

## Training Tutorials

### Tutorial 1: Training from Scratch

```python
"""
Complete tutorial for training a model from scratch
"""
from visual_text_matcher import (
    TextImageMatcher, Config, Trainer, 
    DatasetBuilder, ContrastiveLoss
)
from torch.utils.data import DataLoader
import torch.optim as optim

# Step 1: Create configuration
config = Config(
    model_type="lstm",
    embedding_dim=256,
    hidden_dim=512,
    max_sequence_length=30,
    batch_size=32,
    learning_rate=0.001,
    num_epochs=50,
    device="cuda"
)

# Step 2: Create synthetic dataset
print("Creating synthetic dataset...")
dataset_builder = DatasetBuilder()

# Generate word list
common_words = [
    "hello", "world", "test", "example", "image", "text", "match",
    "computer", "python", "machine", "learning", "vision", "neural",
    "network", "training", "validation", "accuracy", "precision"
]

# Create synthetic images with different fonts and styles
dataset_builder.create_synthetic_dataset(
    words=common_words,
    fonts=["Arial", "Times", "Helvetica", "Courier"],
    styles=["normal", "bold", "italic"],
    sizes=[20, 24, 28, 32],
    output_dir="data/synthetic",
    images_per_word=50  # 50 variations per word
)

# Step 3: Create contrastive pairs
print("Creating contrastive pairs...")
pairs = dataset_builder.create_contrastive_pairs("data/synthetic")
print(f"Created {len(pairs)} training pairs")

# Step 4: Setup data loaders
from visual_text_matcher.data.datasets import ContrastiveDataset

train_dataset = ContrastiveDataset(pairs[:int(0.8 * len(pairs))], config)
val_dataset = ContrastiveDataset(pairs[int(0.8 * len(pairs)):], config)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Step 5: Initialize model and training components
model = TextImageMatcher(config)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
loss_fn = ContrastiveLoss(margin=1.0)

# Step 6: Setup trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    config=config
)

# Step 7: Train the model
print("Starting training...")
history = trainer.train(save_path="models/custom_trained_model.pth")

# Step 8: Evaluate results
print("Training completed!")
print(f"Final train loss: {history['train_loss'][-1]:.4f}")
print(f"Final val loss: {history['val_loss'][-1]:.4f}")
print(f"Best val accuracy: {max(history['val_accuracy']):.4f}")

# Step 9: Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
```

### Tutorial 2: Fine-tuning a Pretrained Model

```python
"""
Fine-tune a pretrained model for your specific domain
"""

# Step 1: Load pretrained model
base_model_path = "models/pretrained_general_model.pth"
model = TextImageMatcher.load_from_checkpoint(base_model_path)

# Step 2: Prepare domain-specific data
# Assume you have medical text images
medical_terms = [
    "diagnosis", "treatment", "patient", "symptoms", "medication",
    "prescription", "therapy", "examination", "clinical", "medical"
]

# Create domain-specific dataset
domain_dataset_builder = DatasetBuilder()
domain_dataset_builder.create_synthetic_dataset(
    words=medical_terms,
    fonts=["Arial", "Times"],  # Medical documents often use standard fonts
    output_dir="data/medical_synthetic"
)

# Step 3: Setup fine-tuning with lower learning rate
fine_tune_config = Config(
    learning_rate=0.0001,  # Lower learning rate for fine-tuning
    num_epochs=20,         # Fewer epochs
    batch_size=16,         # Smaller batch size
    weight_decay=1e-5      # Add regularization
)

# Step 4: Freeze early layers (optional)
# Freeze image encoder layers
for param in model.image_encoder[:3].parameters():
    param.requires_grad = False

# Step 5: Setup training for fine-tuning
domain_pairs = domain_dataset_builder.create_contrastive_pairs("data/medical_synthetic")
domain_train_dataset = ContrastiveDataset(domain_pairs, fine_tune_config)
domain_train_loader = DataLoader(domain_train_dataset, batch_size=16, shuffle=True)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=fine_tune_config.learning_rate
)

# Step 6: Fine-tune
fine_tune_trainer = Trainer(
    model=model,
    train_loader=domain_train_loader,
    optimizer=optimizer,
    loss_fn=ContrastiveLoss(margin=0.8),  # Smaller margin for fine-tuning
    config=fine_tune_config
)

print("Starting fine-tuning...")
fine_tune_history = fine_tune_trainer.train(
    save_path="models/medical_finetuned_model.pth"
)

print("Fine-tuning completed!")
```

---

## Real-World Applications

### Application 1: Document Verification System

```python
"""
Complete document verification system using visual-text matching
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

class DocumentVerifier:
    def __init__(self, model_path: str, confidence_threshold: float = 0.9):
        self.api = MatchingAPI(model_path)
        self.threshold = confidence_threshold
        self.verification_history = []
    
    def verify_document(self, document_config: Dict) -> Dict:
        """
        Verify a document against expected field values
        
        Args:
            document_config: Dictionary containing field definitions
                {
                    "document_id": "DOC_123",
                    "fields": [
                        {
                            "name": "name",
                            "image_path": "extracted_name.png",
                            "expected_value": "John Doe",
                            "required": True
                        },
                        {
                            "name": "id_number",
                            "image_path": "extracted_id.png",
                            "expected_value": "123456789",
                            "required": True
                        }
                    ]
                }
        """
        results = {
            "document_id": document_config["document_id"],
            "overall_status": "pending",
            "field_results": [],
            "confidence_score": 0.0,
            "timestamp": time.time()
        }
        
        total_confidence = 0
        verified_fields = 0
        required_fields_verified = 0
        total_required_fields = 0
        
        for field in document_config["fields"]:
            field_result = self._verify_field(field)
            results["field_results"].append(field_result)
            
            if field_result["verified"]:
                verified_fields += 1
                total_confidence += field_result["confidence"]
                
                if field.get("required", False):
                    required_fields_verified += 1
            
            if field.get("required", False):
                total_required_fields += 1
        
        # Calculate overall confidence
        if verified_fields > 0:
            results["confidence_score"] = total_confidence / verified_fields
        
        # Determine overall status
        if required_fields_verified == total_required_fields:
            if results["confidence_score"] >= self.threshold:
                results["overall_status"] = "verified"
            else:
                results["overall_status"] = "low_confidence"
        else:
            results["overall_status"] = "failed"
        
        # Store in history
        self.verification_history.append(results)
        
        return results
    
    def _verify_field(self, field: Dict) -> Dict:
        """Verify a single document field"""
        try:
            result = self.api.match_single(
                image_path=field["image_path"],
                text=field["expected_value"],
                threshold=self.threshold
            )
            
            return {
                "field_name": field["name"],
                "verified": result["match"],
                "confidence": result["similarity"],
                "expected": field["expected_value"],
                "processing_time": result["processing_time"],
                "error": None
            }
            
        except Exception as e:
            return {
                "field_name": field["name"],
                "verified": False,
                "confidence": 0.0,
                "expected": field["expected_value"],
                "processing_time": 0,
                "error": str(e)
            }
    
    def generate_report(self, document_id: str) -> str:
        """Generate a verification report for a document"""
        # Find document in history
        doc_result = None
        for result in self.verification_history:
            if result["document_id"] == document_id:
                doc_result = result
                break
        
        if not doc_result:
            return f"No verification record found for document {document_id}"
        
        report = f"""
Document Verification Report
===========================
Document ID: {doc_result['document_id']}
Overall Status: {doc_result['overall_status'].upper()}
Overall Confidence: {doc_result['confidence_score']:.3f}
Verification Time: {time.ctime(doc_result['timestamp'])}

Field Details:
"""
        
        for field in doc_result["field_results"]:
            status = "✓ VERIFIED" if field["verified"] else "✗ FAILED"
            report += f"""
  {field['field_name']}:
    Status: {status}
    Expected: "{field['expected']}"
    Confidence: {field['confidence']:.3f}
    Processing Time: {field['processing_time']:.2f}ms
"""
            if field["error"]:
                report += f"    Error: {field['error']}\n"
        
        return report

# Usage example
verifier = DocumentVerifier("models/document_model.pth", confidence_threshold=0.85)

# Define document structure
id_document = {
    "document_id": "ID_001",
    "fields": [
        {
            "name": "full_name",
            "image_path": "extracted_fields/name.png",
            "expected_value": "John Doe",
            "required": True
        },
        {
            "name": "id_number",
            "image_path": "extracted_fields/id_number.png",
            "expected_value": "123456789",
            "required": True
        },
        {
            "name": "date_of_birth",
            "image_path": "extracted_fields/dob.png",
            "expected_value": "01/01/1990",
            "required": False
        }
    ]
}

# Verify document
verification_result = verifier.verify_document(id_document)

# Print results
print(f"Document Status: {verification_result['overall_status']}")
print(f"Confidence: {verification_result['confidence_score']:.3f}")

# Generate detailed report
report = verifier.generate_report("ID_001")
print(report)

# Expected output:
# Document Status: verified
# Confidence: 0.923
# 
# Document Verification Report
# ===========================
# Document ID: ID_001
# Overall Status: VERIFIED
# Overall Confidence: 0.923
# ...
```

### Application 2: OCR Quality Assessment Tool

```python
"""
Tool for assessing OCR quality using visual-text matching
"""
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class OCRAssessmentResult:
    text_confidence: float
    character_accuracy: float
    word_accuracy: float
    processing_time: float
    quality_grade: str
    recommendations: List[str]

class OCRQualityAssessor:
    def __init__(self, model_path: str):
        self.api = MatchingAPI(model_path)
        self.quality_thresholds = {
            'excellent': 0.95,
            'good': 0.85,
            'fair': 0.70,
            'poor': 0.50
        }
    
    def assess_ocr_quality(self, 
                          image_path: str, 
                          ocr_text: str,
                          ground_truth: Optional[str] = None) -> OCRAssessmentResult:
        """
        Assess the quality of OCR output
        
        Args:
            image_path: Path to the original image
            ocr_text: Text extracted by OCR
            ground_truth: Known correct text (optional)
        """
        start_time = time.time()
        
        # Primary assessment using visual-text matching
        result = self.api.match_single(image_path, ocr_text)
        text_confidence = result['similarity']
        
        # Character and word accuracy (if ground truth available)
        char_accuracy = 1.0
        word_accuracy = 1.0
        
        if ground_truth:
            char_accuracy = self._calculate_character_accuracy(ocr_text, ground_truth)
            word_accuracy = self._calculate_word_accuracy(ocr_text, ground_truth)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Determine quality grade
        quality_grade = self._determine_quality_grade(text_confidence)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            text_confidence, char_accuracy, word_accuracy, ocr_text
        )
        
        return OCRAssessmentResult(
            text_confidence=text_confidence,
            character_accuracy=char_accuracy,
            word_accuracy=word_accuracy,
            processing_time=processing_time,
            quality_grade=quality_grade,
            recommendations=recommendations
        )
    
    def _calculate_character_accuracy(self, ocr_text: str, ground_truth: str) -> float:
        """Calculate character-level accuracy using edit distance"""
        import editdistance
        
        distance = editdistance.eval(ocr_text, ground_truth)
        max_length = max(len(ocr_text), len(ground_truth))
        
        if max_length == 0:
            return 1.0
        
        return 1.0 - (distance / max_length)
    
    def _calculate_word_accuracy(self, ocr_text: str, ground_truth: str) -> float:
        """Calculate word-level accuracy"""
        ocr_words = ocr_text.split()
        gt_words = ground_truth.split()
        
        if len(gt_words) == 0:
            return 1.0 if len(ocr_words) == 0 else 0.0
        
        correct_words = sum(1 for ocr_word, gt_word in zip(ocr_words, gt_words) 
                           if ocr_word == gt_word)
        
        return correct_words / len(gt_words)
    
    def _determine_quality_grade(self, confidence: float) -> str:
        """Determine quality grade based on confidence"""
        for grade, threshold in self.quality_thresholds.items():
            if confidence >= threshold:
                return grade
        return 'very_poor'
    
    def _generate_recommendations(self, 
                                text_conf: float, 
                                char_acc: float, 
                                word_acc: float,
                                ocr_text: str) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if text_conf < 0.7:
            recommendations.append("Consider image preprocessing (denoising, contrast enhancement)")
            recommendations.append("Try different OCR engines or models")
        
        if char_acc < 0.8:
            recommendations.append("Check for character-level errors (I/l, O/0, etc.)")
            recommendations.append("Consider character-level post-processing")
        
        if word_acc < 0.7:
            recommendations.append("Apply dictionary-based spell checking")
            recommendations.append("Use language models for context correction")
        
        if len(ocr_text.split()) < 3:
            recommendations.append("Short text detected - consider manual verification")
        
        if not recommendations:
            recommendations.append("OCR quality is good - no specific improvements needed")
        
        return recommendations
    
    def batch_assessment(self, 
                        image_texts: List[Tuple[str, str, Optional[str]]]) -> List[OCRAssessmentResult]:
        """Assess multiple OCR results"""
        results = []
        
        for image_path, ocr_text, ground_truth in image_texts:
            try:
                result = self.assess_ocr_quality(image_path, ocr_text, ground_truth)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Add error result
                results.append(OCRAssessmentResult(
                    text_confidence=0.0,
                    character_accuracy=0.0,
                    word_accuracy=0.0,
                    processing_time=0.0,
                    quality_grade='error',
                    recommendations=[f"Error: {str(e)}"]
                ))
        
        return results
    
    def generate_quality_report(self, results: List[OCRAssessmentResult]) -> str:
        """Generate comprehensive quality report"""
        if not results:
            return "No results to report"
        
        # Calculate statistics
        confidences = [r.text_confidence for r in results if r.quality_grade != 'error']
        char_accuracies = [r.character_accuracy for r in results if r.quality_grade != 'error']
        word_accuracies = [r.word_accuracy for r in results if r.quality_grade != 'error']
        
        report = f"""
OCR Quality Assessment Report
============================
Total samples assessed: {len(results)}
Successful assessments: {len(confidences)}
Failed assessments: {len(results) - len(confidences)}

Overall Statistics:
  Average text confidence: {np.mean(confidences):.3f}
  Average character accuracy: {np.mean(char_accuracies):.3f}
  Average word accuracy: {np.mean(word_accuracies):.3f}
  Average processing time: {np.mean([r.processing_time for r in results]):.2f}ms

Quality Distribution:
"""
        
        # Count quality grades
        grade_counts = {}
        for result in results:
            grade = result.quality_grade
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        for grade, count in sorted(grade_counts.items()):
            percentage = (count / len(results)) * 100
            report += f"  {grade.title()}: {count} ({percentage:.1f}%)\n"
        
        # Common recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        from collections import Counter
        common_recs = Counter(all_recommendations).most_common(5)
        
        report += "\nMost Common Recommendations:\n"
        for rec, count in common_recs:
            report += f"  • {rec} ({count} times)\n"
        
        return report

# Usage example
assessor = OCRQualityAssessor("models/ocr_quality_model.pth")

# Test data: (image_path, ocr_result, ground_truth)
test_data = [
    ("test_images/doc1.png", "Hello World", "Hello World"),
    ("test_images/doc2.png", "The quick brown fox", "The quick brown fox"),
    ("test_images/doc3.png", "Th3 qu1ck br0wn f0x", "The quick brown fox"),  # OCR errors
    ("test_images/doc4.png", "Machine Learning", None),  # No ground truth
]

# Batch assessment
results = assessor.batch_assessment(test_data)

# Print individual results
for i, result in enumerate(results):
    print(f"\nSample {i+1}:")
    print(f"  Quality Grade: {result.quality_grade}")
    print(f"  Text Confidence: {result.text_confidence:.3f}")
    print(f"  Character Accuracy: {result.character_accuracy:.3f}")
    print(f"  Recommendations: {', '.join(result.recommendations[:2])}")

# Generate comprehensive report
report = assessor.generate_quality_report(results)
print(report)
```

---

## Integration Examples

### Example 7: REST API Integration

```python
"""
Integrate visual-text matching into a REST API service
"""
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import uuid
from io import BytesIO
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the matching API
matcher_api = MatchingAPI("models/production_model.pth", device="cuda")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/api/match', methods=['POST'])
def match_image_text():
    """
    Match uploaded image with provided text
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image_data",
        "text": "text_to_match",
        "threshold": 0.8  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data or 'text' not in data:
            return jsonify({'error': 'Missing image or text data'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        
        # Save temporary file
        temp_filename = f"{uuid.uuid4().hex}.png"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        # Perform matching
        threshold = data.get('threshold', 0.8)
        result = matcher_api.match_single(temp_path, data['text'], threshold)
        
        # Cleanup temporary file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/match/batch', methods=['POST'])
def match_batch():
    """
    Batch matching for multiple image-text pairs
    
    Expected JSON payload:
    {
        "pairs": [
            {"image": "base64_data", "text": "text1"},
            {"image": "base64_data", "text": "text2"}
        ],
        "threshold": 0.8  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'pairs' not in data:
            return jsonify({'error': 'Missing pairs data'}), 400
        
        threshold = data.get('threshold', 0.8)
        temp_files = []
        pairs = []
        
        # Process each pair
        for pair in data['pairs']:
            if 'image' not in pair or 'text' not in pair:
                continue
            
            # Decode and save image
            image_data = base64.b64decode(pair['image'])
            temp_filename = f"{uuid.uuid4().hex}.png"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            with open(temp_path, 'wb') as f:
                f.write(image_data)
            
            temp_files.append(temp_path)
            pairs.append((temp_path, pair['text']))
        
        # Perform batch matching
        results = matcher_api.match_batch(pairs)
        
        # Cleanup temporary files
        for temp_file in temp_files:
            os.remove(temp_file)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Simple test to ensure model is responsive
        test_result = matcher_api.match_single("test_images/health_check.png", "test")
        return jsonify({'status': 'healthy', 'model_responsive': True})
    except:
        return jsonify({'status': 'unhealthy', 'model_responsive': False}), 503

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get API performance metrics"""
    # This would typically pull from a monitoring system
    return jsonify({
        'requests_processed': 1234,
        'average_response_time': 45.6,
        'success_rate': 0.987,
        'model_accuracy': 0.923
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Example 8: Webhook Integration

```python
"""
Process documents via webhook notifications
"""
import requests
import json
from typing import Dict, Any

class WebhookProcessor:
    def __init__(self, model_path: str, webhook_url: str):
        self.api = MatchingAPI(model_path)
        self.webhook_url = webhook_url
    
    def process_webhook(self, webhook_data: Dict[str, Any]):
        """
        Process incoming webhook for document verification
        
        Expected webhook format:
        {
            "document_id": "DOC_123",
            "callback_url": "https://client.com/webhook/result",
            "fields": [
                {
                    "name": "field_name",
                    "image_url": "https://storage.com/image.png",
                    "expected_value": "expected text"
                }
            ]
        }
        """
        try:
            document_id = webhook_data['document_id']
            callback_url = webhook_data['callback_url']
            fields = webhook_data['fields']
            
            results = []
            
            for field in fields:
                # Download image
                image_path = self._download_image(field['image_url'], document_id)
                
                # Perform verification
                result = self.api.match_single(
                    image_path, 
                    field['expected_value']
                )
                
                results.append({
                    'field_name': field['name'],
                    'verified': result['match'],
                    'confidence': result['similarity'],
                    'expected_value': field['expected_value']
                })
                
                # Cleanup downloaded image
                os.remove(image_path)
            
            # Send results back via callback
            self._send_callback(callback_url, {
                'document_id': document_id,
                'status': 'completed',
                'results': results,
                'timestamp': time.time()
            })
            
        except Exception as e:
            # Send error callback
            self._send_callback(webhook_data.get('callback_url'), {
                'document_id': webhook_data.get('document_id'),
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            })
    
    def _download_image(self, image_url: str, document_id: str) -> str:
        """Download image from URL"""
        response = requests.get(image_url)
        response.raise_for_status()
        
        temp_path = f"temp_{document_id}_{uuid.uuid4().hex}.png"
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        return temp_path
    
    def _send_callback(self, callback_url: str, data: Dict[str, Any]):
        """Send results back to client via webhook"""
        try:
            response = requests.post(
                callback_url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send callback: {e}")

# Usage with Flask
from flask import Flask

app = Flask(__name__)
processor = WebhookProcessor("models/production_model.pth", "")

@app.route('/webhook/process', methods=['POST'])
def handle_webhook():
    """Handle incoming webhook requests"""
    webhook_data = request.get_json()
    
    # Process asynchronously (in production, use Celery or similar)
    import threading
    thread = threading.Thread(
        target=processor.process_webhook,
        args=(webhook_data,)
    )
    thread.start()
    
    return jsonify({'status': 'accepted', 'message': 'Processing started'}), 202
```

This comprehensive documentation provides complete examples and tutorials covering all aspects of the Visual-Text Matching System, from basic usage to advanced applications and integrations. Each example includes full working code, expected outputs, and practical explanations to help users effectively implement the system in their projects.