# From Basic ML to Deep Learning with PyTorch: A Complete Guide

## Table of Contents

1. Basic Machine Learning Concepts
2. Introduction to Neural Networks
3. Understanding PyTorch Basics
4. Deep Learning Fundamentals
5. Advanced PyTorch Concepts
6. Understanding Transformers
7. Practical Implementation

## 1. Basic Machine Learning Concepts

### 1.1 Classification Basics

Remember your basic ML course? Classification is like sorting things into different boxes. In traditional ML, we used algorithms like:

```python
# Basic classification example using scikit-learn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Key concepts:

- Features (input data)
- Labels (output classes)
- Training data vs Test data
- Model fitting

### 1.2 From Traditional ML to Neural Networks

Traditional ML vs Neural Networks:

```python
# Traditional ML (Logistic Regression)
y = w * x + b

# Neural Network (Single Neuron)
y = activation_function(w * x + b)
```

## 2. Introduction to Neural Networks

### 2.1 The Basic Building Block: Neuron

Think of a neuron as a mathematical function:

```python
import numpy as np

class SimpleNeuron:
    def __init__(self):
        self.weight = np.random.randn()
        self.bias = np.random.randn()

    def forward(self, x):
        return self.relu(self.weight * x + self.bias)

    def relu(self, x):
        return max(0, x)
```

### 2.2 Activation Functions

Common activation functions:

```python
def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)
```

## 3. Understanding PyTorch Basics

### 3.1 Tensors: The Foundation

PyTorch's basic building block is the tensor:

```python
import torch

# Creating tensors
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 3)
z = torch.randn(3, 3)  # Random normal distribution

# Basic operations
a = x + y
b = torch.matmul(y, z)
```

### 3.2 Autograd: Automatic Differentiation

PyTorch's magic sauce - automatic gradient calculation:

```python
# Creating tensors with gradients
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # Will print 4.0 (derivative of x^2 at x=2)
```

## 4. Deep Learning Fundamentals

### 4.1 Building a Simple Neural Network

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)  # 10 inputs, 5 outputs
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(5, 2)   # 5 inputs, 2 outputs

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

### 4.2 Training Loop Basics

```python
# Basic training loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
```

## 5. Advanced PyTorch Concepts

### 5.1 Dataset and DataLoader

Creating custom datasets:

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Using DataLoader
dataset = CustomDataset(data, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

### 5.2 Loss Functions and Optimizers

```python
# Common loss functions
criterion = nn.CrossEntropyLoss()  # For classification
criterion = nn.MSELoss()          # For regression

# Optimizers
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

## 6. Understanding Transformers

### 6.1 Self-Attention Basics

Simplified self-attention implementation:

```python
def self_attention(query, key, value):
    # Calculate attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Scale scores
    scores = scores / torch.sqrt(torch.tensor(key.shape[-1]))

    # Apply softmax
    attention_weights = torch.softmax(scores, dim=-1)

    # Get weighted sum
    output = torch.matmul(attention_weights, value)
    return output
```

### 6.2 Understanding BERT/RoBERTa Architecture

Basic transformer block:

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        # Self attention
        attention_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attention_out)

        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x
```

## 7. Practical Implementation

### 7.1 Putting It All Together

Here's how we can build a simple code classifier:

```python
class SimpleCodeClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.transformer = TransformerBlock(128, 8)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # Take the [CLS] token output
        x = x[:, 0, :]
        x = self.classifier(x)
        return x
```

### 7.2 Training with Best Practices

```python
def train_with_best_practices(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=epochs)

    for epoch in range(epochs):
        # Training
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'])
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                outputs = model(batch['input_ids'])
                val_loss += criterion(outputs, batch['labels'])
```

## Practical Tips and Tricks

1. **Memory Management**

   - Use gradient accumulation for large models
   - Clear cache regularly: `torch.cuda.empty_cache()`
   - Use mixed precision training when possible

2. **Training Stability**

   - Start with small learning rates
   - Use learning rate warmup
   - Implement gradient clipping
   - Monitor training metrics

3. **Debugging Tips**

   - Print tensor shapes during forward pass
   - Use `torch.autograd.detect_anomaly()`
   - Monitor GPU memory usage

4. **Performance Optimization**
   - Use DataLoader with appropriate batch size
   - Enable cudnn benchmarking
   - Utilize multiple GPUs when available

## Conclusion

Understanding deep learning requires building from the basics up to advanced concepts. The journey from basic ML to transformers is significant, but each step builds on the previous ones:

1. Start with basic ML concepts
2. Understand neural networks
3. Learn PyTorch fundamentals
4. Master deep learning concepts
5. Explore advanced architectures

Remember: Practice is key! Start with simple implementations and gradually move to more complex models.

Want to dive deeper into any particular topic? The best way to learn is by implementing these concepts in small, manageable projects.

---

_Note: This tutorial assumes basic Python knowledge and familiarity with machine learning concepts. For absolute beginners, it's recommended to first study Python basics and fundamental machine learning concepts._
