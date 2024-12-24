import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    AdamW,
    get_linear_schedule_with_warmup
)
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
import json
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn.utils import clip_grad_norm_
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training and evaluation"""
    model_name: str = "huggingface/CodeBERTa-small-v1"
    max_length: int = 512
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    validation_split: float = 0.1
    freeze_base_model: bool = True
    unfreeze_last_n_layers: int = 2
    weight_decay: float = 0.01
    dropout: float = 0.1
    gradient_accumulation_steps: int = 1

class CodeDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer: RobertaTokenizer, max_length: int):
        if not texts or not labels or len(texts) != len(labels):
            raise ValueError("Invalid input data format")
            
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        try:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
        except Exception as e:
            logger.error(f"Error tokenizing text at index {idx}: {str(e)}")
            raise

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CodeBertaClassifier:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=len(self.get_language_mapping()),
                hidden_dropout_prob=config.dropout,
                attention_probs_dropout_prob=config.dropout
            )
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        self._setup_model_parameters()
        logger.info(f"Model initialized with config: {config}")

    def _setup_model_parameters(self):
        """Configure model parameters freezing/unfreezing"""
        if self.config.freeze_base_model:
            for param in self.model.roberta.parameters():
                param.requires_grad = False

            if self.config.unfreeze_last_n_layers > 0:
                for layer in self.model.roberta.encoder.layer[-self.config.unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,d} ({100 * trainable_params / total_params:.2f}%)")

    @staticmethod
    def get_language_mapping() -> Dict[str, int]:
        return {
            'cpp': 0, 'groovy': 1, 'java': 2, 'javascript': 3,
            'json': 4, 'python': 5, 'xml': 6, 'yml': 7
        }

    def prepare_data(self, data_dir):
        """Original data loading method kept as is"""
        texts = []
        labels = []
        file_counts = {lang: 0 for lang in self.get_language_mapping().keys()}

        for lang_path in Path(data_dir).glob('*'):
            if lang_path.is_dir():
                lang = lang_path.name
                if lang in self.get_language_mapping():
                    for file_path in lang_path.glob('*'):
                        try:
                            text = file_path.read_text(encoding='utf-8')
                            texts.append(text)
                            labels.append(self.get_language_mapping()[lang])
                            file_counts[lang] += 1
                        except UnicodeDecodeError:
                            continue

        logger.info("\nFiles processed per language:")
        for lang, count in file_counts.items():
            logger.info(f"{lang}: {count} files")

        return texts, labels

    def train_epoch(self, train_loader: DataLoader, optimizer: AdamW, 
                   scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                   scaler: GradScaler, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with automatic mixed precision
            with autocast():
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                scaler.unscale_(optimizer)
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler:
                    scheduler.step()

            # Metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_correct += (predictions == batch['labels']).sum().item()
            total_samples += len(batch['labels'])

            # Log progress
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f'Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | '
                    f'Loss: {loss.item():.4f} | '
                    f'Accuracy: {total_correct/total_samples:.4f} | '
                    f'Time: {elapsed:.2f}s'
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        logger.info(f'Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}')
        
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                total_correct += (predictions == batch['labels']).sum().item()
                total_samples += len(batch['labels'])

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        logger.info(f'Validation | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}')
        
        return avg_loss, accuracy

    def train_and_evaluate(self, train_texts: List[str], train_labels: List[int],
                          test_texts: List[str], test_labels: List[int]) -> Tuple[np.ndarray, str]:
        # Create validation split
        val_size = int(len(train_texts) * self.config.validation_split)
        train_texts, val_texts = train_texts[:-val_size], train_texts[-val_size:]
        train_labels, val_labels = train_labels[:-val_size], train_labels[-val_size:]

        # Create datasets
        train_dataset = CodeDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        val_dataset = CodeDataset(val_texts, val_labels, self.tokenizer, self.config.max_length)
        test_dataset = CodeDataset(test_texts, test_labels, self.tokenizer, self.config.max_length)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

        # Initialize training components
        optimizer = self._setup_optimizer()
        num_training_steps = len(train_loader) * self.config.epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        scaler = GradScaler()

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, scaler, epoch)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(best_model_path, epoch, optimizer, scheduler, scaler, val_loss)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

        # Load best model and evaluate
        self._load_checkpoint(best_model_path)
        test_results = self._evaluate(test_loader)
        
        # Save final results
        results_path = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'confusion_matrix': test_results['confusion_matrix'].tolist(),
                'classification_report': test_results['classification_report'],
                'test_accuracy': test_results['accuracy']
            }, f, indent=2)
        
        logger.info(f"Final results saved to {results_path}")
        return test_results['confusion_matrix'], test_results['classification_report']

    def _evaluate(self, test_loader: DataLoader) -> Dict:
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                probs = F.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_probs.extend(probs.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        return {
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
            'classification_report': classification_report(
                all_labels, 
                all_predictions,
                target_names=list(self.get_language_mapping().keys()),
                digits=4
            ),
            'accuracy': (all_predictions == all_labels).mean()
        }

    def _setup_optimizer(self) -> AdamW:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def _save_checkpoint(self, path: str, epoch: int, optimizer: AdamW,
                        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                        scaler: GradScaler, loss: float):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }, path)

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def _evaluate(self, test_loader: DataLoader) -> Dict:
        self.model.eval()
        predictions = []
        true_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        return {
            'confusion_matrix': confusion_matrix(true_labels, predictions),
            'classification_report': classification_report(
                true_labels, predictions,
                target_names=list(self.get_language_mapping().keys()),
                digits=4
            ),
            'accuracy': (predictions == true_labels).mean()
        }

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict with confidence score"""
        self.model.eval()
        try:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.config.max_length,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=-1)
                predicted_idx = torch.argmax(outputs.logits, dim=-1).item()
                confidence = probs[0][predicted_idx].item()

            idx_to_lang = {v: k for k, v in self.get_language_mapping().items()}
            return idx_to_lang[predicted_idx], confidence

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize configuration
    config = ModelConfig()
    
    # Initialize classifier
    logger.info("Initializing CodeBERTa classifier...")
    classifier = CodeBertaClassifier(config)
    
    # Get data_dir from command-line arguments
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "FileTypeData"  # Default value if not provided

    # Prepare data
    logger.info("Preparing data from FileTypeData directory...")
    data_dir = "FileTypeData"
    texts, labels = classifier.prepare_data(data_dir)
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Testing samples: {len(test_texts)}")
    
    # Train and evaluate
    conf_matrix, class_report = classifier.train_and_evaluate(
        train_texts, train_labels,
        test_texts, test_labels
    )
    
    # Log final results
    logger.info("\nFinal Confusion Matrix:")
    logger.info(conf_matrix)
    logger.info("\nFinal Classification Report:")
    logger.info(class_report)