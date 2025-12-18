# OpenVINO Dependency Parser for bezoku low resource and indigenous language models

Universal CPU-optimized inference tool for BiLSTM dependency parsing with multi-task outputs.

## Features

- **Multi-task parsing**: UPOS, XPOS, DEPREL, FEATS, and HEAD prediction
- **OpenVINO optimization**: Fast CPU inference using Intel OpenVINO toolkit
- **Language-agnostic**: Works with any trained model following the standard format
- **Interactive mode**: Test sentences in real-time
- **CoNLL-U export**: Toggle between table and CoNLL-U format output

## Requirements

```bash
pip install -r requirements_openvino.txt
```

Required packages:
- `torch>=2.0.0`
- `openvino>=2023.0.0`
- `numpy>=1.24.0`

## Required Files

Create a model directory and add these files:
1. **model.pt** - PyTorch model checkpoint (state_dict format)
2. **vocab.pkl** - Vocabulary dictionaries (pickle format)
3. **metadata.json** - Model metadata and performance metrics

## Usage

### Quick Start

```bash
python3 openvino_parser.py
```

The script will prompt you for:
1. Model directory path
2. Which model file to use (if multiple .pt files exist)

### Example Session

```
Enter path to model directory: /home/user/Models and Results/Brazilian Portuguese/
✓ Selected: best_pos_tagger_model_a100(1).pt

[Model loads and converts to OpenVINO...]

> O Brasil é um país da América do Sul.

ID   TOKEN           UPOS     XPOS       DEPREL          FEATS                HEAD 
====================================================================================================
1    O               DET      ART        det             Definite=Def|Gen..   2    
2    Brasil          PROPN    PROPN      nsubj           Number=Sing          3    
3    é               AUX      V          cop             Mood=Ind|Number=..   0    
4    um              DET      ART        det             Definite=Ind|Gen..   5    
5    país            NOUN     N          root            Number=Sing          3    
6    da              ADP      ADP        case            _                    7    
7    América         PROPN    PROPN      nmod            Number=Sing          5    
8    do              ADP      ADP        case            _                    9    
9    Sul             PROPN    PROPN      nmod            Number=Sing          7    
10   .               PUNCT    .          punct           _                    5    

⏱️  Inference time: 23.4ms (10 tokens)
```

## Commands

### Interactive Mode
- `quit` or `exit` - Exit the program
- `info` - Display model information and performance metrics
- `conllu` - Toggle CoNLL-U format output
- `help` - Show command list

### Output Formats

**Table format** (default):
- Human-readable table with columns: ID, TOKEN, UPOS, XPOS, DEPREL, FEATS, HEAD
- Truncates long FEATS strings for readability

**CoNLL-U format** (toggle with `conllu` command):
- Standard Universal Dependencies format
- Compatible with UD tools and validators
- Includes placeholder fields for LEMMA, DEPS, and MISC

## Model Architecture

The parser uses a BiLSTM architecture with:
- Word embeddings (128-512 dimensions)
- Bidirectional LSTM layers (1-4 layers)
- Multi-task classification heads for UPOS, XPOS, DEPREL, FEATS
- Biaffine attention for HEAD prediction

## OpenVINO Optimization

The conversion process:
1. Loads PyTorch checkpoint
2. Infers architecture from tensor shapes
3. Reconstructs model with loaded weights
4. Converts to OpenVINO IR format (.xml + .bin)
5. Compiles for CPU execution

Benefits:
- **2-5× faster** inference on Intel CPUs
- Lower memory footprint
- Optimized for production deployment
- No GPU required

## Performance

Typical inference times on Intel Xeon/Core i7:
- Short sentence (5-10 tokens): 15-30ms
- Medium sentence (20-30 tokens): 30-60ms
- Long sentence (50+ tokens): 80-150ms

Memory usage: ~200-500MB (model dependent)

## File Structure

After first run, the directory will contain:
```
openvino_model/
├── dependency_parser.xml   # OpenVINO IR model
└── dependency_parser.bin   # Model weights
```

The OpenVINO model is cached and reused for subsequent runs.

## Troubleshooting

### "Conversion failed" error
- Check that model checkpoint contains all expected layers
- Verify vocab.pkl has all required vocabulary dictionaries
- Ensure PyTorch and OpenVINO versions are compatible

### "Index out of range" during inference
- Check tokenization - may need to adjust for your language
- Verify vocabulary contains `<UNK>` token for unknown words

### Slow inference
- First run includes conversion time (~10-30 seconds)
- Subsequent runs use cached OpenVINO model
- CPU optimization depends on Intel CPU features (AVX2, AVX-512)

## Citation

If using this tool for research, please cite:
```
@software{bezoku_dependency_parser,
  title={Universal OpenVINO Dependency Parser},
  author={bezoku.ai},
  year={2025},
  url={https://github.com/bezokurepo}
}
```

## License

See model-specific metadata.json for usage terms and licensing.

## Contact

For support, custom models, or enterprise deployment:
- Email: ian.gilmour@bezoku.ai
- Models: https://github.com/bezokurepo/models
