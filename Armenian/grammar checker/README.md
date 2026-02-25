# Grammar Checker Version - Experimental

## Overview

This folder contains an **experimental version** of the OpenVINO dependency parser with integrated grammar checking functionality.

⚠️ **Important**: The standard parser in the parent directory (`../openvino_dependency_parser.py`) is the stable, tested version. Use this grammar checker version only after testing with your specific language models.

## Files in This Folder

### Core Files
- **`openvino_dependency_parser_with_grammar.py`** - Parser with grammar checking integration
- **`grammar_checker.py`** - Grammar validation module
- **`requirements_openvino.txt`** - Same as parent directory

### Documentation
- **`README.md`** - This file
- **`GRAMMAR_CHECKER_SETUP.md`** - Detailed setup and usage guide
- **`IMPLEMENTATION_SUMMARY.md`** - Technical implementation details

## Quick Start

### 1. Test with a Language Model

```bash
cd "/home/bezoku/openVINO runners/Grammar_Checker_Version"
python3 openvino_dependency_parser_with_grammar.py
```

### 2. Required Files

Your model directory must contain:
- `best_pos_tagger_model_a100.pt`
- `vocab.pkl`
- `metadata.json`
- `morphology_table.pkl` (optional, enables suggestions)

### 3. Interactive Commands

Once running:
- Type sentences to parse them
- Type `grammar` to toggle grammar checking on/off
- Type `info` to see model information
- Type `quit` to exit

## Features Added

✓ **Subject-verb agreement checking**
✓ **Noun-adjective agreement checking**
✓ **Determiner-noun agreement checking**
✓ **Case consistency validation**
✓ **Detailed error reporting**
✓ **Toggle grammar checking on/off**

## Testing Checklist

Before deploying this version:

- [ ] Test with at least 3 different language models
- [ ] Verify grammar checking accuracy on known errors
- [ ] Compare performance with standard parser
- [ ] Test with and without morphology_table.pkl
- [ ] Validate all interactive commands work
- [ ] Check error messages are helpful and clear

## Comparison with Standard Parser

### Standard Parser (Parent Directory)
- ✓ Stable and tested
- ✓ Fast inference only
- ✓ No external dependencies beyond standard requirements
- Use for: Production deployments, baseline parsing

### Grammar Checker Version (This Directory)
- ⚠️ Experimental
- ✓ Adds grammar validation
- ✓ Optional morphology table support
- ✓ Detailed error reporting
- Use for: Educational tools, grammar checking applications, testing

## If Issues Occur

If you encounter problems with this version:

1. **Fall back to standard parser**:
   ```bash
   cd "/home/bezoku/openVINO runners"
   python3 openvino_dependency_parser.py
   ```

2. **Check logs** for error messages

3. **Test without grammar checking**:
   - Run the grammar version
   - Type `grammar` to turn off checking
   - Verify basic parsing still works

## Deployment Strategy

### Phase 1: Internal Testing
- Test with all 24 language models
- Validate accuracy of grammar checks
- Collect feedback on error messages

### Phase 2: Limited Release
- Release as "beta" or "experimental" feature
- Provide both standard and grammar checker versions
- Gather user feedback

### Phase 3: Production (If Successful)
- Merge into standard parser as optional feature
- Update all documentation
- Replace standard parser in parent directory

## File Locations

```
/home/bezoku/openVINO runners/
├── openvino_dependency_parser.py     ← STABLE VERSION (use this)
├── README.md
├── requirements_openvino.txt
├── History/
│   └── openvino_dependency_parser (HISTORY).py
└── Grammar_Checker_Version/           ← EXPERIMENTAL VERSION (test this)
    ├── openvino_dependency_parser_with_grammar.py
    ├── grammar_checker.py
    ├── README.md (this file)
    ├── GRAMMAR_CHECKER_SETUP.md
    └── IMPLEMENTATION_SUMMARY.md
```

## Morphology Tables

All 24 morphology tables are in:
```
/home/bezoku/UD_Training_Data/[Language]/morphology_table.pkl
```

Copy these to your model directories when deploying the grammar checker version.

## Support

For questions or issues with this experimental version:
- Email: ian.gilmour@bezoku.ai
- Check `GRAMMAR_CHECKER_SETUP.md` for detailed documentation
- Review `IMPLEMENTATION_SUMMARY.md` for technical details

## Version Control

- **Standard Parser**: Stable, recommended for production
- **Grammar Checker Version**: Experimental, test before deploying
- Always keep both versions until grammar checker is fully validated
