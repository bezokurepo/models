# Quick Start Guide - OpenVINO Dependency Parser
The model runner folders will have all refactored code to run bezoku models. The runners in each Model folder may be out of date, this is where the golden copy can be found. Email ian.gilmour@bezoku.ai for support.

## 1. Install Python (one-time)
For Windows users, you can visit the Windows Store
https://apps.microsoft.com/detail/9nq7512cxl7t?ocid=webpdpshare

For Ubuntu copy this instruction into Terminal
```bash
$ sudo apt-get update
$ sudo apt-get install python3.6
```
## 2. Install Dependencies (one-time)

```bash
pip install torch openvino numpy
```

## 3. Run the Parser

```bash
python3 openvino_dependency_parser.py
```

## 4. Point to Your Model

When prompted, enter the path to your model directory:
```
Enter path to model directory: /home/xyz/Models and Results/Brazilian Portuguese/
```

## 5. Start Parsing!

```
> O Brasil é um país muito bonito.
> A economia cresceu em 2024.
> conllu          # Toggle CoNLL-U format
> info            # Show model info
> quit            # Exit
```

## Example Output

```
ID   TOKEN           UPOS     XPOS       DEPREL          FEATS                HEAD 
====================================================================================================
1    O               DET      DET        det             Definite=Def|Gen..   2    
2    Brasil          NOUN     NOUN       nsubj           _                    4    
3    é               AUX      AUX        cop             _                    4    
4    um              DET      DET        det             _                    5    
5    país            NOUN     NOUN       root            _                    0    
...

⏱️  Inference time: 15.9ms (10 tokens)
```

## What You Get

- **5 parsing tasks**: UPOS, XPOS, DEPREL, FEATS, HEADS
- **Fast inference**: 15-30ms per sentence on CPU
- **Intel optimized**: 2-5× faster than PyTorch
- **Two output formats**: Table or CoNLL-U
- **Interactive testing**: Real-time parsing

## Files Needed

Your model directory must have:
- ✅ `best_pos_tagger_model_a100.pt` - Model weights
- ✅ `vocab.pkl` - Vocabularies
- ✅ `metadata.json` - Model info


**Ready for production use! 🚀**

---

For detailed documentation, see `README_OPENVINO.md`
