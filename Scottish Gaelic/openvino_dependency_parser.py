
"""
Universal OpenVINO Dependency Parser
Supports multi-task inference: UPOS, XPOS, DEPREL, FEATS, HEADS
Works with model.pt, vocab.pkl, and metadata.json
"""

import torch
import openvino as ov
import numpy as np
import json
import pickle
import os
import re
import time
from pathlib import Path
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class DependencyParser(nn.Module):
    """BiLSTM-based dependency parser with multi-task learning"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 num_upos, num_xpos, num_deprel, num_feats, dropout=0.3):
        super(DependencyParser, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # Classification heads
        self.upos_fc = nn.Linear(lstm_output_dim, num_upos)
        self.xpos_fc = nn.Linear(lstm_output_dim, num_xpos)
        self.deprel_fc = nn.Linear(lstm_output_dim, num_deprel)
        self.feats_fc = nn.Linear(lstm_output_dim, num_feats)
        
        # Head prediction using dot product (simplified to MLP)
        self.head_mlp_dep = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU()
        )
        self.head_mlp_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """Forward pass returning all task outputs"""
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_output, _ = self.lstm(embedded)
        lstm_output = self.dropout(lstm_output)
        
        # Task predictions
        upos_logits = self.upos_fc(lstm_output)
        xpos_logits = self.xpos_fc(lstm_output)
        deprel_logits = self.deprel_fc(lstm_output)
        feats_logits = self.feats_fc(lstm_output)
        
        # Head prediction (dot product scoring)
        dep_repr = self.head_mlp_dep(lstm_output)  # [batch, seq, hidden]
        head_repr = self.head_mlp_head(lstm_output)  # [batch, seq, hidden]
        
        # Compute dot product scores: [batch, seq_dep, seq_head]
        head_scores = torch.matmul(dep_repr, head_repr.transpose(1, 2))
        
        return upos_logits, xpos_logits, deprel_logits, feats_logits, head_scores


# ============================================================================
# FILE LOADING UTILITIES
# ============================================================================

def get_model_directory():
    """Prompt user for model directory containing .pt, .pkl, and .json files"""
    while True:
        model_dir = input("\nEnter path to model directory (contains .pt, vocab.pkl, metadata.json): ").strip()
        model_dir = model_dir.strip('"').strip("'")
        
        if os.path.isdir(model_dir):
            # Check for required files
            pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            has_vocab = os.path.exists(os.path.join(model_dir, 'vocab.pkl'))
            has_metadata = os.path.exists(os.path.join(model_dir, 'metadata.json'))
            
            if not pt_files:
                print("❌ No .pt model file found in directory")
                continue
            if not has_vocab:
                print("❌ vocab.pkl not found in directory")
                continue
            if not has_metadata:
                print("❌ metadata.json not found in directory")
                continue
            
            # If multiple .pt files, let user choose
            if len(pt_files) > 1:
                print(f"\nFound {len(pt_files)} model files:")
                for i, f in enumerate(pt_files, 1):
                    print(f"  {i}. {f}")
                choice = int(input("Select model file number: ")) - 1
                pt_file = pt_files[choice]
            else:
                pt_file = pt_files[0]
            
            model_path = os.path.join(model_dir, pt_file)
            vocab_path = os.path.join(model_dir, 'vocab.pkl')
            metadata_path = os.path.join(model_dir, 'metadata.json')
            
            print(f"✓ Selected: {pt_file}")
            return model_path, vocab_path, metadata_path
        else:
            print(f"❌ Directory not found: {model_dir}")


def load_vocabularies(vocab_path: str) -> Dict:
    """Load vocabulary dictionaries from pickle file"""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def sanitize_json(raw: str) -> str:
    """Remove trailing commas and common JSON formatting issues."""
    # Strip trailing commas before } or ]
    sanitized = re.sub(r',\s*([}\]])', r'\1', raw)
    return sanitized


def load_metadata(metadata_path: str) -> Dict:
    """Load model metadata from JSON file, tolerating trailing commas."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        # Retry after stripping trailing commas
        sanitized = sanitize_json(raw)
        try:
            metadata = json.loads(sanitized)
            print("⚠️  metadata.json had formatting issues (e.g. trailing commas) — auto-corrected")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse {metadata_path} even after sanitization: {e}\n"
                f"Please verify the file contains valid JSON."
            ) from e
    return metadata


# ============================================================================
# MODEL CONVERSION
# ============================================================================

def infer_model_config(checkpoint: Dict, vocab: Dict) -> Dict:
    """Infer model configuration from checkpoint tensor shapes"""
    
    # Extract dimensions from tensor shapes
    embedding_weight = checkpoint['embedding.weight']
    vocab_size, embedding_dim = embedding_weight.shape
    
    # LSTM hidden dim (weight_ih contains 4 * hidden_dim for LSTM gates)
    lstm_weight_ih = checkpoint['lstm.weight_ih_l0']
    hidden_dim = lstm_weight_ih.shape[0] // 4  # 4 gates (i, f, g, o)
    
    # Count LSTM layers
    num_layers = sum(1 for k in checkpoint.keys() if k.startswith('lstm.weight_ih_l')) // 2  # div by 2 for bidirectional
    
    # Output dimensions
    num_upos = vocab['upos_to_idx']['<PAD>'] if '<PAD>' in vocab['upos_to_idx'] else len(vocab['upos_to_idx'])
    num_xpos = vocab['xpos_to_idx']['<PAD>'] if '<PAD>' in vocab['xpos_to_idx'] else len(vocab['xpos_to_idx'])
    num_deprel = vocab['deprel_to_idx']['<PAD>'] if '<PAD>' in vocab['deprel_to_idx'] else len(vocab['deprel_to_idx'])
    num_feats = vocab['feats_to_idx']['<PAD>'] if '<PAD>' in vocab['feats_to_idx'] else len(vocab['feats_to_idx'])
    
    # Use actual output layer sizes
    num_upos = checkpoint['upos_fc.weight'].shape[0]
    num_xpos = checkpoint['xpos_fc.weight'].shape[0]
    num_deprel = checkpoint['deprel_fc.weight'].shape[0]
    num_feats = checkpoint['feats_fc.weight'].shape[0]
    
    config = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_upos': num_upos,
        'num_xpos': num_xpos,
        'num_deprel': num_deprel,
        'num_feats': num_feats,
        'dropout': 0.3  # Default, not critical for inference
    }
    
    return config


def convert_to_openvino(model_path: str, vocab: Dict, output_dir: str = "openvino_model") -> Optional[str]:
    """Convert PyTorch model to OpenVINO format"""
    
    print(f"\nConverting {os.path.basename(model_path)} to OpenVINO...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Infer model configuration
    config = infer_model_config(checkpoint, vocab)
    
    print("✓ Model configuration:")
    print(f"  Vocab size: {config['vocab_size']:,}")
    print(f"  Embedding dim: {config['embedding_dim']}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  UPOS classes: {config['num_upos']}")
    print(f"  XPOS classes: {config['num_xpos']}")
    print(f"  DEPREL classes: {config['num_deprel']}")
    print(f"  FEATS classes: {config['num_feats']}")
    
    # Create model instance
    model = DependencyParser(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_upos=config['num_upos'],
        num_xpos=config['num_xpos'],
        num_deprel=config['num_deprel'],
        num_feats=config['num_feats'],
        dropout=config['dropout']
    )
    
    # Load weights
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✓ Model weights loaded successfully")
    
    # Create dummy input for tracing
    max_seq_len = 128
    dummy_input = torch.randint(0, config['vocab_size'], (1, max_seq_len))
    
    try:
        # Convert to OpenVINO
        ov_model = ov.convert_model(model, example_input=dummy_input)
        
        # Save model
        Path(output_dir).mkdir(exist_ok=True)
        ov_model_path = f"{output_dir}/dependency_parser.xml"
        ov.save_model(ov_model, ov_model_path)
        
        print(f"✓ OpenVINO model saved to {ov_model_path}")
        return ov_model_path
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# INFERENCE
# ============================================================================

def tokenize_sentence(sentence: str) -> List[str]:
    """Simple tokenization with punctuation splitting"""
    # Split punctuation
    for punct in ['.', ',', '!', '?', ':', ';', '(', ')', '"', "'", '«', '»']:
        sentence = sentence.replace(punct, f' {punct} ')
    
    # Do NOT split on accented characters - they are part of words
    # Previous Portuguese-specific rule removed to support all languages
    
    tokens = sentence.split()
    tokens = [t.strip() for t in tokens if t.strip()]
    
    return tokens


def predict_sentence(sentence: str, compiled_model, vocab: Dict, 
                     input_layer, output_layers, max_len: int = 128) -> Dict:
    """
    Predict all dependency parsing tasks for a sentence
    
    Returns:
        Dict with keys: tokens, upos, xpos, deprel, feats, heads
    """
    
    # Tokenize
    tokens = tokenize_sentence(sentence)
    
    # Convert to IDs (lowercase matching training)
    token_ids = []
    for token in tokens:
        token_lower = token.lower()
        token_ids.append(vocab['word_to_idx'].get(token_lower, vocab['word_to_idx'].get('<UNK>', 1)))
    
    # Pad to fixed length
    original_length = len(token_ids)
    
    if len(token_ids) < max_len:
        token_ids.extend([0] * (max_len - len(token_ids)))
    else:
        token_ids = token_ids[:max_len]
        tokens = tokens[:max_len]
        original_length = max_len
    
    # Run inference
    input_data = np.array([token_ids], dtype=np.int64)
    outputs = compiled_model([input_data])
    
    # Extract predictions for each task
    upos_logits = outputs[output_layers[0]]
    xpos_logits = outputs[output_layers[1]]
    deprel_logits = outputs[output_layers[2]]
    feats_logits = outputs[output_layers[3]]
    head_scores = outputs[output_layers[4]]
    
    # Convert logits to predictions
    upos_preds = np.argmax(upos_logits[0], axis=-1)
    xpos_preds = np.argmax(xpos_logits[0], axis=-1)
    deprel_preds = np.argmax(deprel_logits[0], axis=-1)
    feats_preds = np.argmax(feats_logits[0], axis=-1)
    head_preds = np.argmax(head_scores[0], axis=-1)
    
    # Post-process: Enforce single-root constraint
    # Find all tokens predicting HEAD=0 (multiple roots)
    root_candidates = [i for i in range(original_length) if head_preds[i] == 0]
    
    if len(root_candidates) > 1:
        # Multiple roots detected - pick the best one based on head_scores
        best_root_idx = -1
        best_root_score = -np.inf
        
        for candidate_idx in root_candidates:
            # Score for this token pointing to 0 (root)
            score = head_scores[0][candidate_idx, 0]
            if score > best_root_score:
                best_root_score = score
                best_root_idx = candidate_idx
        
        # Reassign non-best roots to point to the best root
        for candidate_idx in root_candidates:
            if candidate_idx != best_root_idx:
                # Find the next-best head (excluding 0)
                scores_copy = head_scores[0][candidate_idx].copy()
                scores_copy[0] = -np.inf  # Mask out root
                head_preds[candidate_idx] = np.argmax(scores_copy)
    
    elif len(root_candidates) == 0:
        # No root predicted - force the highest-scoring token to be root
        best_idx = 0
        best_score = -np.inf
        for i in range(original_length):
            score = head_scores[0][i, 0]
            if score > best_score:
                best_score = score
                best_idx = i
        head_preds[best_idx] = 0
    
    # Build results (only for original tokens, not padding)
    results = {
        'tokens': [],
        'upos': [],
        'xpos': [],
        'deprel': [],
        'feats': [],
        'heads': []
    }
    
    for i in range(original_length):
        results['tokens'].append(tokens[i])
        results['upos'].append(vocab['idx_to_upos'].get(upos_preds[i], 'UNK'))
        results['xpos'].append(vocab['idx_to_xpos'].get(xpos_preds[i], 'UNK'))
        results['deprel'].append(vocab['idx_to_deprel'].get(deprel_preds[i], 'UNK'))
        results['feats'].append(vocab['idx_to_feats'].get(feats_preds[i], '_'))
        results['heads'].append(int(head_preds[i]))
    
    return results


def format_conllu(results: Dict) -> str:
    """Format results as CoNLL-U format"""
    lines = []
    for i, token in enumerate(results['tokens'], 1):
        line = [
            str(i),                      # ID
            token,                       # FORM
            '_',                         # LEMMA (not predicted)
            results['upos'][i-1],        # UPOS
            results['xpos'][i-1],        # XPOS
            results['feats'][i-1],       # FEATS
            str(results['heads'][i-1]),  # HEAD
            results['deprel'][i-1],      # DEPREL
            '_',                         # DEPS
            '_'                          # MISC
        ]
        lines.append('\t'.join(line))
    return '\n'.join(lines)


# ============================================================================
# INTERACTIVE INTERFACE
# ============================================================================

def print_results_table(results: Dict):
    """Print results in a formatted table"""
    print("\n" + "=" * 100)
    print(f"{'ID':<4} {'TOKEN':<15} {'UPOS':<8} {'XPOS':<10} {'DEPREL':<15} {'FEATS':<20} {'HEAD':<5}")
    print("=" * 100)
    
    for i, token in enumerate(results['tokens'], 1):
        feats_str = results['feats'][i-1][:18] + '..' if len(results['feats'][i-1]) > 20 else results['feats'][i-1]
        
        print(f"{i:<4} {token:<15} {results['upos'][i-1]:<8} {results['xpos'][i-1]:<10} "
              f"{results['deprel'][i-1]:<15} {feats_str:<20} {results['heads'][i-1]:<5}")
    
    print("=" * 100)


def print_model_info(metadata: Dict, vocab: Dict):
    """Print model information"""
    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    print(f"Language: {metadata['language']} ({metadata['language_code']})")
    print(f"Version: {metadata['version']}")
    print(f"Trained: {metadata['date_trained']}")
    print(f"\nPerformance:")
    for metric, value in metadata['performance'].items():
        print(f"  {metric.upper():<10} {value*100:5.2f}%")
    print(f"\nVocabulary Sizes:")
    for key, size in metadata['vocabulary_sizes'].items():
        print(f"  {key:<20} {size:>6,}")
    print(f"\nCharacteristics:")
    print(f"  Variant: {metadata['characteristics']['variant']}")
    print(f"  Morphology: {metadata['characteristics']['morphology']}")
    print("=" * 70)


def interactive_mode(compiled_model, vocab: Dict, metadata: Dict, input_layer, output_layers):
    """Interactive testing mode"""
    
    print("\n" + "=" * 70)
    print("INTERACTIVE DEPENDENCY PARSING MODE")
    print("=" * 70)
    print("Commands:")
    print("  'quit' or 'exit' - Exit program")
    print("  'info' - Show model information")
    print("  'conllu' - Toggle CoNLL-U format output")
    print("  'help' - Show this help message")
    print("\nEnter sentences to parse:")
    
    show_conllu = False
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'info':
                print_model_info(metadata, vocab)
            elif user_input.lower() == 'conllu':
                show_conllu = not show_conllu
                print(f"✓ CoNLL-U output: {'ON' if show_conllu else 'OFF'}")
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  'quit'/'exit' - Exit")
                print("  'info' - Model info")
                print("  'conllu' - Toggle CoNLL-U format")
            elif user_input:
                # Parse sentence
                start_time = time.time()
                results = predict_sentence(user_input, compiled_model, vocab, 
                                          input_layer, output_layers)
                inference_time = time.time() - start_time
                
                # Display results
                if show_conllu:
                    print("\n" + format_conllu(results))
                else:
                    print_results_table(results)
                
                print(f"\n⏱️  Inference time: {inference_time*1000:.1f}ms ({len(results['tokens'])} tokens)")
                
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🚀 UNIVERSAL OPENVINO DEPENDENCY PARSER")
    print("=" * 70)
    print("\nThis tool provides CPU-optimized inference for dependency parsing")
    print("with multi-task outputs: UPOS, XPOS, DEPREL, FEATS, and HEAD prediction")
    
    # Step 1: Get model files
    model_path, vocab_path, metadata_path = get_model_directory()
    
    # Step 2: Load vocabularies and metadata
    print("\n" + "=" * 70)
    print("LOADING MODEL ASSETS")
    print("=" * 70)
    
    vocab = load_vocabularies(vocab_path)
    print(f"✓ Loaded vocabularies from {os.path.basename(vocab_path)}")
    
    metadata = load_metadata(metadata_path)
    print(f"✓ Loaded metadata from {os.path.basename(metadata_path)}")
    
    print_model_info(metadata, vocab)
    
    # Step 3: Convert to OpenVINO
    print("\n" + "=" * 70)
    print("CONVERTING TO OPENVINO")
    print("=" * 70)
    
    ov_model_path = convert_to_openvino(model_path, vocab)
    
    if ov_model_path is None:
        print("❌ Conversion failed!")
        return
    
    # Step 4: Initialize OpenVINO runtime
    print("\n" + "=" * 70)
    print("INITIALIZING OPENVINO RUNTIME")
    print("=" * 70)
    
    core = ov.Core()
    model = core.read_model(ov_model_path)
    compiled_model = core.compile_model(model, "CPU")
    
    input_layer = compiled_model.input(0)
    output_layers = [compiled_model.output(i) for i in range(5)]
    
    print(f"✓ Model compiled for CPU inference")
    print(f"✓ Device: {core.get_property('CPU', 'FULL_DEVICE_NAME')}")
    print(f"✓ Ready for parsing!")
    
    # Step 5: Interactive mode
    interactive_mode(compiled_model, vocab, metadata, input_layer, output_layers)
    
    print("\n✓ Session complete!")
    print(f"✓ OpenVINO model saved: {ov_model_path}")


if __name__ == "__main__":
    main()
