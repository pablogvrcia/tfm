# Quick Start Guide

Get up and running in 5 minutes!

## 🚀 Setup (First Time Only)

```bash
cd /home/pablo/tfm/code
./setup.sh
```

This will install everything automatically. Takes ~10-15 minutes.

## ✅ Verify Installation

```bash
source venv/bin/activate
python test_installation.py
```

Should show: `🎉 All tests passed! 🎉`

## 🎨 Usage Examples

### 1. Segment (Find Objects)

```bash
python main.py --image photo.jpg --prompt "person" --mode segment --visualize
```

**Output:** `output/segmentation.png` with highlighted objects

### 2. Remove (Delete Objects)

```bash
python main.py --image photo.jpg --prompt "person" --mode remove --visualize
```

**Output:** `output/edited.png` with object removed

### 3. Replace (Change Objects)

```bash
python main.py --image photo.jpg --prompt "old TV" --mode replace --edit "modern TV" --visualize
```

**Output:** `output/edited.png` with TV replaced

### 4. Style (Apply Artistic Styles)

```bash
python main.py --image photo.jpg --prompt "building" --mode style --edit "watercolor painting" --visualize
```

**Output:** `output/edited.png` with building styled

## ⚙️ Common Options

```bash
--image PATH      # Input image
--prompt TEXT     # What to find
--mode MODE       # segment/remove/replace/style
--edit TEXT       # For replace/style modes
--config PRESET   # fast/balanced/quality
--top-k N         # Number of results (default: 5)
--visualize       # Show results
--output DIR      # Output directory (default: output/)
```

## 🔧 Troubleshooting

### Out of Memory?

```bash
# Use smaller model
python scripts/download_sam2_checkpoints.py --model sam2_hiera_tiny
```

### Too Slow?

```bash
# Use fast config
python main.py --image photo.jpg --prompt "car" --config fast
```

### SAM 2 Not Found?

```bash
# Download checkpoint
python scripts/download_sam2_checkpoints.py --model sam2_hiera_large
```

## 📚 Documentation

- **README.md** - Overview and examples
- **SETUP.md** - Detailed installation guide
- **CHANGES.md** - What was changed from previous version
- **This file** - Quick commands

## 🎯 Thesis Context

This implements the methodology from Chapter 3:
- **SAM 2**: Automatic mask generation (§3.2.2)
- **CLIP**: Vision-language features (§3.2.1)
- **Alignment**: Mask-text scoring (§3.2.3)
- **Stable Diffusion**: Inpainting (§3.2.4)

See your thesis in `/home/pablo/tfm/overleaf/` for full details.

## 💡 Tips

1. **Start simple**: Try segment mode first
2. **Be specific**: "red car on left" works better than "car"
3. **Check output/**: Results are saved even without `--visualize`
4. **Use --help**: `python main.py --help` for all options
5. **GTX 1060 limits**: Stick to balanced or fast config for best performance

## 📞 Help

**Error messages are your friend!** They tell you exactly what to do:

```
WARNING: SAM 2 checkpoint not found...
Please download with: python scripts/download_sam2_checkpoints.py
```

Follow the instructions in the error message!

---

**Ready to go!** 🚀

Start with: `python main.py --image photo.jpg --prompt "person" --mode segment --visualize`
