# Grammar Checker Version - beta release

## Overview

This folder contains an **experimental version** of the OpenVINO dependency parser with integrated grammar checking functionality.

⚠️ **Important**: The standard parser in the parent directory (`../openvino_dependency_parser.py`) is the stable, tested version. This version is subject to user testing before it is marked as stable.

## Files in This Folder

### Core Files
- **`openvino_dependency_parser_with_grammar.py`** - Parser with grammar checking integration
- **`grammar_checker.py`** - Grammar validation module


## Quick Start

### 1. Follow the steps for the dependency parser to get python and pip working on Windows or Ubuntu.

## Start bezoku installation
## Required Files

Create a model directory, e.g. Armenian, and download these files:
1. **best_pos_tagger_model_a100.pt** - PyTorch model checkpoint (state_dict format)
2. **vocab.pkl** - Vocabulary dictionaries (pickle format)
3. **metadata.json** - Model metadata and performance metrics
4. **requirements_openvino.txt** - The libraries needed to run the model
5. **openvino_dependency_parser_with_grammar.py** - This is the model runner that brings all the files together
6. **grammar_checker.py** - This is similar to a library, for example lke numpy

Once the six files are downloaded on to the desktop (if the openvino_dependency_parser.py file is in the directory, it can remain), follow these steps before running the model:

Open Terminal
```bash
cd ~/"the folder name" 
```
For example if the folder you named is called "Armenian", the command in Terminal would be **cd ~/Armenian** 

If you have issues navigating folders, visit this easy to follow blog (https://www.redhat.com/en/blog/navigating-filesystem-linux-terminal).

Copy the command below into Terminal - if you have not already run them when using openvino_dependency_parser.py
```bash
pip install -r requirements_openvino.txt
```

This command loads the required packages:
- `torch>=2.0.0`
- `openvino>=2023.0.0`
- `numpy>=1.24.0`

You are now ready to run the model !

## Usage

### Start

```bash
python3 openvino_dependency_parser_with_grammar.py
```

The script will prompt you for:
1. Model directory path
2. Which model file to use (if multiple .pt files exist)

### 2. Required Files

Your model directory must contain:
- `best_pos_tagger_model_a100.pt` - Same as parent directory (do not forget to unzip)
- `vocab.pkl` - Same as parent directory
- `metadata.json` - Same as parent directory
- `requirements_openvino.txt` - Same as parent directory
- `morphology_table.pkl` - optional, needed to enable suggestions
- `grammar_checker.py` - needed to run openvino_dependency_parser_with_grammar.py

### 3. Interactive Commands

Once running:
- Type sentences to parse them
- Type `grammar` to toggle grammar checking on/off
- Type `info` to see model information
- Type `quit` to exit

## Extra features to the openvino_dependency_parser.py

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
