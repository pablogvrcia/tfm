# Quick Command Reference

## ✅ Commands That Work (Use These!)

### Activate Environment (Always do this first!)
```bash
cd /home/pablo/tfm/code
source venv/bin/activate
```

### Segmentation
```bash
# Simple segmentation
python segment.py --image photo.jpg --prompt "car"

# With visualization
python segment.py --image photo.jpg --prompt "person" --visualize

# More results
python segment.py --image photo.jpg --prompt "dog" --top-k 10

# Different image
python segment.py --image /path/to/image.jpg --prompt "object"
```

### Generate Figures
```bash
# Quick test figure
python quick_viz.py

# All thesis figures (publication quality)
python create_visualizations.py

# Simple test
python test_segment.py
```

### Help
```bash
python segment.py --help
```

## ❌ Don't Use (Memory Issues on GTX 1060!)

```bash
# ❌ This will run out of memory:
python main.py --image photo.jpg --prompt "car" --mode segment
```

## 📊 Check Status

```bash
# GPU status
nvidia-smi

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test installation
python test_installation.py
```

## 📁 View Results

```bash
# Generated figures
ls -lh figures/
ls -lh thesis_figures/

# Open figure
xdg-open figures/segmentation_car.png
```

## 🎓 For Thesis

### Generate experiment results:
```bash
# Run on multiple prompts
python segment.py -i photo.jpg -p "car" > results_car.txt
python segment.py -i photo.jpg -p "person" > results_person.txt
python segment.py -i photo.jpg -p "tree" > results_tree.txt
```

### Create all figures:
```bash
python create_visualizations.py
cp thesis_figures/* ~/tfm/overleaf/figures/
```

## 💾 Memory Usage

```bash
# Check while running
watch -n 1 nvidia-smi
```

Expected:
- `segment.py`: ~2 GB / 6 GB (safe ✅)
- `main.py`: ~6-7 GB / 6 GB (out of memory ❌)

## 🔧 Troubleshooting

### Out of memory?
→ Use `segment.py` instead of `main.py`

### No matches found?
→ Try different prompts that exist in the image
→ Or use different images with clear objects

### Slow performance?
→ Normal! SAM 2 tiny takes ~8s per image
→ This is expected on GTX 1060

## Summary

**✅ Always use:** `segment.py`, `quick_viz.py`, `test_segment.py`
**❌ Don't use:** `main.py` (out of memory on GTX 1060)
