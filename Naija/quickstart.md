# Quick Start Guide - OpenVINO Dependency Parser

## 1. Install Dependencies (one-time)

```bash
pip install torch openvino numpy
```

## 2. Run the Parser

```bash
python3 openVINO_parser.py
```

## 3. Point to Your Model

When prompted, enter the path to your model directory:
```
Enter path to model directory: /home/bezoku/Models and Results/Brazilian Portuguese/
```

## What You Get

- **5 parsing tasks**: UPOS, XPOS, DEPREL, FEATS, HEADS
- **Fast inference**: 15-30ms per sentence on CPU
- **Intel optimized**: 2-5× faster than PyTorch
- **Two output formats**: Table or CoNLL-U
- **Interactive testing**: Real-time parsing

## Files Needed

Your model directory must have:
- ✅ `model.pt` - Model weights
- ✅ `vocab.pkl` - Vocabularies
- ✅ `metadata.json` - Model info

## Performance

**Naija Model:**
- LAS:  0.7979
- Inference: ~16ms/sentence
- CPU: Intel i7-10700

**Ready for production use! 🚀**

---

For detailed documentation, see `README_OPENVINO.md`
