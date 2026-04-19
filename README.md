# Masked Language Model Training Animation

This project provides an animation that visually demonstrates how a masked language model (MLM) like BERT learns to fill in missing words in sentences. It is designed as an educational tool for students to understand the core idea behind MLM training.

## Project Purpose

Masked language modeling is a foundational technique in modern natural language processing. In this approach, a model is trained to predict missing words in a sentence based on the surrounding context. This project helps students visualize this process by showing how the model's predictions improve over time.

## "Wheel of Fortune" Analogy

Think of MLM training as a game similar to "Wheel of Fortune," but instead of guessing missing letters, the model tries to guess entire words. Just like in the game, the model uses clues from the rest of the sentence to make its best guess for the masked word. Over many training examples, the model gets better at using context to fill in the blanks.

## Features
- High-quality animation (MP4) showing the training process
- Visualizes how word embeddings cluster and move as the model learns
- Progress bar for rendering frames
- Designed for classroom and presentation use

## Usage
Run the animation script in your pixi environment:

```bash
pixi run python animation.py
```

The output video will be saved as `bert_training.mp4` in the project directory.

## Requirements
- Python 3.14+
- matplotlib
- numpy
- ffmpeg
- tqdm

All dependencies are managed via `pixi.toml` for cross-platform reproducibility (Linux, macOS, Windows).

## License
MIT
