# Grammar Checker Version - Experimental

## Overview

This folder contains an **experimental version** of the OpenVINO dependency parser with integrated grammar checking functionality.

⚠️ **Important**: The standard parser in the parent directory (`../openvino_dependency_parser.py`) is the stable, tested version. This version is subject to user testing before it is marked as stable.

## Files in This Folder

### Core Files
- **`openvino_dependency_parser_with_grammar.py`** - Parser with grammar checking integration
- **`grammar_checker.py`** - Grammar validation module

### Documentation
- **`README.md`** - This file
- **`GRAMMAR_CHECKER_SETUP.md`** - Detailed setup and usage guide
- **`IMPLEMENTATION_SUMMARY.md`** - Technical implementation details

## Quick Start

### 1. Follow the steps for the dependency parser to get python and pip working on Windows or Ubuntu.

### 2. Required Files

Your model directory must contain:
- `best_pos_tagger_model_a100.pt` - Same as parent directory (do not forget to unzip)
- `vocab.pkl` - Same as parent directory
- `metadata.json` - Same as parent directory
-  `requirements_openvino.txt` - Same as parent directory
- `morphology_table.pkl` (optional, enables suggestions)

### 3. Interactive Commands

Once running:
- Type sentences to parse them
- Type `grammar` to toggle grammar checking on/off
- Type `info` to see model information
- Type `quit` to exit

## New Features

✓ **Subject-verb agreement checking**
✓ **Noun-adjective agreement checking**
✓ **Determiner-noun agreement checking**
✓ **Case consistency validation**
✓ **Detailed error reporting**
✓ **Toggle grammar checking on/off**

### Grammar Checker Version (This Directory)
- ⚠️ Experimental
- ✓ Adds grammar validation
- ✓ Optional morphology table support
- ✓ Detailed error reporting
- Use for: Educational tools, grammar checking applications, testing

## If Issues Occur

If you encounter problems with this version:

1. **Fall back to standard parser**:

2. **Check logs** for error messages
   Email errors to ian.gilmour@bezoku.ai for bug fixing

4. **Test without grammar checking**:
   - Run the grammar version
   - Type `grammar` to turn off checking
   - Verify basic parsing still works

## Support

For questions or issues with this experimental version:
- Email: ian.gilmour@bezoku.ai
- Check `GRAMMAR_CHECKER_SETUP.md` for detailed documentation
- Review `IMPLEMENTATION_SUMMARY.md` for technical details

## Version Control

- **Standard Parser**: Stable, Android and MAC OS / iOS in development
- **Grammar Checker Version**: Beta
