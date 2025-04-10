---
title: 'Finding Interpretable Features in Llama 3 from Scratch'
date: 2015-04-10
permalink: /posts/2025/04/finding-features/
tags:
  - interpretability
  - mechanistic interpretability
  - saes
---

Headings are cool
======

You can have many headings
======

Aren't headings cool?
------

Here's some example code in Python:
```python
def train_autoencoder(data, hidden_size=64, epochs=100):
    # Define a simple autoencoder model
    model = nn.Sequential(
        nn.Linear(data.shape[1], hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, data.shape[1])
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        encoded = model(data)
        loss = criterion(encoded, data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    return model
```


We can express the autoencoder mathematically. For an input vector $x$, the encoder maps it to a hidden representation $h$ through:

$h = \sigma(W_1x + b_1)$

where $W_1$ is the weight matrix, $b_1$ is the bias vector, and $\sigma$ is the ReLU activation function.

The decoder then maps this hidden representation back to reconstruct the input:

$\hat{x} = W_2h + b_2$

The training objective is to minimize the reconstruction error:

$\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \|x_i - \hat{x}_i\|^2$

where $n$ is the number of training examples.

In our code above, this is implemented using PyTorch's MSE loss:


