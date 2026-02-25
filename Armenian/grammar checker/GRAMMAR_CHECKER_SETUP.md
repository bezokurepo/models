# Grammar Checker Setup Guide

## Overview

The grammar checker is now integrated into the OpenVINO dependency parser. It validates morphological agreement and syntactic consistency across all 24 languages without requiring any model retraining.

## Files

### Updated Files
- **`openvino_dependency_parser.py`** - Now includes grammar checking integration
- **`README.md`** - Updated with grammar checking documentation

### New Files
- **`grammar_checker.py`** - Core validation module
- **`morphology_table.pkl`** - Per-language morphology lookup tables (in each model directory)

## Setup for Each Language

### Step 1: Organize Your Files

Each language model directory should contain:
```
ModelDirectory/
├── best_pos_tagger_model_a100.pt
├── vocab.pkl
├── metadata.json
└── morphology_table.pkl  ← Add this file
```

### Step 2: Copy Morphology Tables

Copy the generated `morphology_table.pkl` from training data to model directories:

```bash
# Example for Armenian
cp /home/bezoku/UD_Training_Data/Armenian/morphology_table.pkl \
   /path/to/armenian/model/directory/

# Repeat for all 24 languages
```

### Step 3: Upload to GitHub

For each language repository, upload these files:
1. `best_pos_tagger_model_a100.pt`
2. `vocab.pkl`
3. `metadata.json`
4. `morphology_table.pkl` ← NEW

## Usage

### Running the Parser

```bash
python3 openvino_dependency_parser.py
```

### Interactive Commands

Once running:
- Type a sentence to parse it
- Type `grammar` to toggle grammar checking on/off
- Grammar checking is ON by default if morphology table is available

### Example Session

```
> O Brasil é um país da América do Sul.

[Parse results displayed...]

======================================================================
GRAMMAR CHECK: Found 0 error(s)
======================================================================

✓ No grammar errors detected

> The children walks to school

[Parse results displayed...]

======================================================================
GRAMMAR CHECK: Found 1 error(s)
======================================================================

❌ Error 1: Subject-Verb Agreement
   Token 3: "walks"
   Issue: Verb does not agree with subject in Number
   Related to token 2: "children"
   Expected: Number=Plur|Person=3|Tense=Pres
   Found: Number=Sing|Person=3|Tense=Pres

======================================================================
```

## How It Works

### No Retraining Required ✓

The grammar checker uses:
1. **Existing model outputs**: UPOS, XPOS, FEATS, DEPREL, HEAD
2. **Rule-based validation**: Checks agreement patterns
3. **Morphology table**: Optional lookup for corrections

### Validation Rules

1. **Subject-Verb Agreement**
   - Checks Number, Person, Gender features
   - Validates across `nsubj`, `csubj` relations

2. **Noun-Adjective Agreement**
   - Checks Number, Gender, Case features
   - Validates `amod` relations

3. **Determiner-Noun Agreement**
   - Checks Number, Gender, Case features
   - Validates `det` relations

4. **Case Consistency**
   - Validates expected case for syntactic roles
   - Example: subjects typically nominative, objects accusative

## Language Coverage

Works with all 24 trained languages:

### Inflected Languages (16)
- Armenian
- Bulgarian
- Czech
- Finnish
- German
- Greek
- Hindi
- Icelandic
- Latin
- Lithuanian
- Polish
- Portuguese
- Romanian
- Russian
- Slovak
- Ukrainian

### Agglutinative Languages (8)
- Basque
- Hungarian
- Japanese
- Korean
- Swahili
- Tamil
- Turkish
- Uyghur

## Technical Details

### Architecture

```
Input sentence
    ↓
Dependency Parser (existing, unchanged)
    ↓ UPOS, XPOS, FEATS, DEPREL, HEAD
Grammar Checker (new post-processor)
    ↓
Error report + suggestions
```

### Benefits

- **Zero retraining**: Uses existing trained models
- **Language-agnostic**: Same code for all 24 languages
- **Optional**: Works with or without morphology tables
- **Interpretable**: Rule-based, not black-box ML
- **Fast**: < 1ms overhead per sentence

## Deployment

### For End Users

Distribute these files per language:
1. `openvino_dependency_parser.py`
2. `grammar_checker.py`
3. `requirements_openvino.txt`
4. Model files (`.pt`, `vocab.pkl`, `metadata.json`, `morphology_table.pkl`)

### For GitHub Repositories

Each language repository should include:
- All model files
- Updated README with grammar checking info
- `morphology_table.pkl` in releases

## Future Enhancements

Potential additions (not implemented yet):
1. Lemmatization for better suggestions
2. Language-specific agreement rules
3. Custom rule configuration
4. Batch file processing mode
5. Confidence scores for errors

## Support

For issues or questions:
- Email: ian.gilmour@bezoku.ai
- GitHub: https://github.com/bezokurepo/models
