# Music Generation with a PyTorch LSTM

This project demonstrates how to generate music using a character-level Recurrent Neural Network (RNN), specifically a Long Short-Term Memory (LSTM) model built with PyTorch. The model is trained on a collection of folk songs written in ABC notation and learns to generate new music sequences character by character.

The complete workflow is implemented in the `Music Generation.ipynb` notebook, including preprocessing, model definition, training logic, and audio playback.

---

## Project Overview

### 1. Data Loading
The dataset consists of 817 folk songs in ABC notation, loaded using the `mitdeeplearning` library.

### 2. Preprocessing
- The entire corpus is merged into one long text string.
- A character-level vocabulary (83 unique characters) is constructed.
- Mappings are created:
  - `char2idx`: maps each character to an integer.
  - `idx2char`: maps each integer back to its corresponding character.
- The full dataset is vectorized by converting characters into integer indices.

### 3. Data Batching
A custom `get_batch` function:
- Creates input sequences (`x`)
- Creates target sequences (`y`), which are simply the input shifted one character ahead  
This trains the model to predict the next character in a sequence.

---

## Model Architecture

The LSTM model (`LSTMModel`) contains:

- **Embedding Layer (`nn.Embedding`)**  
  Converts input character indices into dense vector embeddings.

- **LSTM Layer (`nn.LSTM`)**  
  The main recurrent layer that captures temporal patterns across sequences.

- **Fully Connected Layer (`nn.Linear`)**  
  Maps the LSTM output to logits over the vocabulary, predicting the next character.

---

## Training

- **Loss Function:**  
  `nn.CrossEntropyLoss`, used for next-character prediction.

- **Optimizer:**  
  `torch.optim.Adam`, used to update model parameters.

> Note: The training notebook defines a `train_step` function, but a complete multi-epoch training loop is not included in the provided file.

---

## Music Generation

The `generate_text` function performs the following:
1. Takes a starting prompt in ABC notation.
2. Runs it through the model to generate predictions.
3. Samples the next character using `torch.multinomial`.
4. Appends the generated character and feeds it back into the model.
5. Repeats this process to generate a full music sequence.

---

## Playback

The generated ABC string is converted back into audio using:

- `abcmidi`
- `timidity`

The synthesized audio waveform can be played directly in the notebook or saved as a `.wav` file.

---

## Technologies Used

- PyTorch
- NumPy
- mitdeeplearning
- abcmidi / timidity for audio synthesis

---

