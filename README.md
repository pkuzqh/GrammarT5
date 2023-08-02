# A Pytorch implemenmtation of GrammarT5.
A PyTorch Implementation of "GrammarT5: Grammar-Integrated Pre-trained Encoder-Decoder
Neural Model for Code"

# Introduction
Pre-trained models for code have exhibited promising performance across various code-related tasks, such as code summarization, code completion, code translation, and bug detection. These accomplishments have substantially contributed to the advancement of AI-assisted programming and developer tools. However, despite their success, the majority of current models still represent code as a token sequence in the fine-tuning phase, which may not adequately capture the essence of the underlying code structure.

In this work, we propose GrammarT5, a grammar-integrated encoder-decoder pre-trained model for code. GrammarT5 employs a novel grammar-integrated representation, Tokenized Grammar Rule List (TGRL), for code. TGRL is constructed based on the grammar rule list utilized in syntax-guided code generation and integrates syntax information with code tokens within an appropriate input length. Furthermore, we suggest attaching language flags to help GrammarT5 differentiate between grammar rules of various programming languages. Finally, we introduce three novel pre-training objectives—Edge Prediction (EP), Identifier Prediction (IP), and Sub-Tree Prediction (STP)—for GrammarT5 to learn syntax from TGRL.

Experiments were conducted on five code-related tasks using ten datasets, demonstrating that GrammarT5 achieves state-of-the-art performance on all tasks in comparison to models of the same scale. Additionally, the paper illustrates that the proposed pre-training objectives and language flags can enhance GrammarT5's ability to better capture code syntax and semantics.

# Dataset
## Train set
The raw data from Code Search Net (https://zenodo.org/record/7857872)
## Test set
### [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main)

# Usage
## Pre-trained Model
We will publish our pre-trained models after the paper is accepted.

## Fine-tuning Model
Task can be one of the following tasks. ['django', 'concode', 'codetrans', 'repair', 'assert', 'conala', 'test', 'repairme', 'transj2c', 'transc2j', 'commentjava', 'commentpython', 'mbpp', 'searchadv', 'searchcos']
```python
sh run.sh
```
The saved model is ```checkModel[task]```.
## Testing Model
```python
sh eval.sh
```

# Dependency
* Python 3.7
* PyTorch 1.12
* transformers 2.26
* Java 8
* docker
* nvidia-docker


