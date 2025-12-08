# SimpleNN

SimpleNN is a C# native neural network library for Unity, heavily inspired by PyTorch.
It is designed to run deep learning inference and training directly within Unity using C# scripts, without external dependencies like Python.

> [!NOTE]
> This project is currently under active development and is partially implemented. APIs are subject to change.

## Key Features

*   **C# Native**: Runs entirely in C# / Unity.
*   **PyTorch-like API**: Familiar intuitive API for PyTorch users.
*   **Autograd**: Automatic differentiation support via dynamic computational graphs.
*   **Tensor Operations**: `Tensor` class handling multi-dimensional array operations (broadcasting, reshaping, etc.).
*   **Neural Modules**: Pre-built layers like `Linear`, `Conv2d`, `Sequential`.
*   **Optimizers**: Common optimizers like `SGD`, `Adam`, `Adagrad` included.

## Core Concepts

### 1. Tensor
The `Tensor` class is the fundamental data structure, representing a multi-dimensional array of floats. It supports:
*   Standard arithmetic (+, -, *, /)
*   Matrix multiplication (`MatMul`)
*   Broadcasting (similar to NumPy/PyTorch)
*   Shape manipulation (`Reshape`, `Permute`)

### 2. TensorBox (Autograd)
`TensorBox` is a wrapper around `Tensor` that enables automatic differentiation.
*   It records the history of operations to build a computation graph.
*   Calling `Backward()` on a scalar `TensorBox` computes gradients for all tensors in the graph that have `RequireGrad = true`.

## Feature Overview

| Category | Available Components |
| :--- | :--- |
| **Containers** | `Sequential`, `ModuleList` |
| **Linear Layers** | `Linear` |
| **Convolution Layers** | `Conv2d`, `TransposedConv2d` |
| **Activations** | `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax` |
| **Loss Functions** | `MSELoss`, `L1Loss` |
| **Optimizers** | `SGD`, `Adam`, `RAdam`, `Adagrad`, `Adadelta` |
| **Math Ops** | `MatMul`, `Pow`, `Exp`, `Log`, `Sin`, `Cos`, etc. |

## Current Limitations

Since this library is under development, there are some known limitations that are planned to be addressed in future updates:

*   **Indexing and Slicing**: Python-style text indexing (e.g., `tensor[:, 0]`) and general slicing are not currently supported. Please use `Split`, `Reshape`, or `Permute` for manipulating tensor dimensions instead.
*   **Module Instance Reuse**: Reusing the same module instance multiple times (e.g., calling `AddModule(layerA)` twice for the same instance) will cause issues with the gradient graph and parameter management. Please instantiate new modules for each layer use.

## Usage Example

Here is a simple example showing how to simple neural network training loop in Unity.

```csharp
using SimpleNN;
using SimpleNN.Module;
using SimpleNN.Optimizer;
using UnityEngine;
using System.Collections.Generic;

public class SimpleExample : MonoBehaviour
{
    void Start()
    {
        // 1. Define Network Model
        // A simple MLP: Input(10) -> ReLU -> Hidden(20) -> Output(1)
        var model = new Sequential(
            new Linear(10, 20),
            new ReLU(),
            new Linear(20, 1)
        );

        // 2. Setup Optimizer (Adam, lr=0.01)
        var optimizer = new Adam(0.01f, model.Parameters());

        // 3. Setup Loss Function (Mean Squared Error)
        var criterion = new MSELoss();

        // 4. Dummy Data (Batch Size: 5, Input Features: 10)
        var input = new TensorBox(Tensor.Random(new int[] { 5, 10 }));
        var target = new TensorBox(Tensor.Random(new int[] { 5, 1 }));

        Debug.Log("Start Training...");

        // 5. Training Loop
        for (int i = 0; i < 100; i++)
        {
            // Zero gradients from previous step
            optimizer.ZeroGrad();

            // Forward Pass
            var output = model.Forward(input);

            // Compute Loss
            var loss = criterion.Forward(output, target);

            // Backward Pass (Compute Gradients)
            loss.Backward();

            // Update Parameters
            optimizer.Step();

            if (i % 20 == 0)
            {
                Debug.Log($"Step {i}, Loss: {loss.GetTensor().Item()}");
            }
        }
    }
}
```

## Directory Structure

*   `Tensor/`: Core tensor implementation and math operations.
*   `Graph/`: Autograd engine and `TensorBox`.
*   `Module/`: Neural network layers and loss functions.
*   `Optimizer/`: Optimization algorithms.
