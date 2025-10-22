# Batch Processing Solution

## Problem with Original Approach

The original `process_all_pdfs.py` script was **failing** because:

1. **Subprocess overhead**: Each file launched a new Python subprocess
2. **Model reloading**: Every subprocess tried to initialize the vllm engine from scratch
3. **Resource conflicts**: The vllm engine failed to initialize repeatedly due to GPU resource conflicts
4. **Error**: `RuntimeError: Engine core initialization failed`

### Why It Failed

```python
# OLD APPROACH (BROKEN)
for each file:
    subprocess.run(['python', 'run_dpsk_ocr_pdf.py'])  # ❌ Tries to load 30GB model each time!
    # vllm engine fails to initialize repeatedly
```

## New Solution: Integrated Processing

Created **`process_all_pdfs_integrated.py`** which:

✅ **Loads the model ONCE** at startup  
✅ **Reuses the same model** for all files  
✅ **No subprocess overhead**  
✅ **Proper resource management**  

### Architecture Comparison

| Aspect | Old (Subprocess) | New (Integrated) |
|--------|-----------------|------------------|
| Model loading | 70 times (once per file) | 1 time (startup) |
| Memory efficiency | ❌ Poor | ✅ Excellent |
| Reliability | ❌ Engine crashes | ✅ Stable |
| Speed | Slow (startup overhead) | Fast (reuse model) |

## Usage

### ✅ Use the Integrated Version

```bash
cd /home/kevin/development/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm
python3 process_all_pdfs_integrated.py
```

### ❌ Don't Use (Broken)

```bash
# DON'T USE - This will fail
python3 process_all_pdfs.py
```

## How It Works

```python
# NEW INTEGRATED APPROACH
# 1. Initialize model ONCE
llm = LLM(model=MODEL_PATH, ...)  # Happens once at startup
processor = DeepseekOCRProcessor()

# 2. Process all files with same model
for each file:
    images = load_document(file)
    results = llm.generate(images)  # ✅ Reuses loaded model!
    save_results(results)
```

## Features

Both scripts maintain the same features:
- ✅ Per-file output directories
- ✅ PDF and EPUB support
- ✅ Progress tracking
- ✅ JSON results file
- ✅ Error handling

## Performance

**Old approach (if it worked):**
- Model loading: ~2 minutes × 70 files = **140 minutes overhead**
- Processing: ~2-3 minutes per file = **140-210 minutes**
- **Total: 280-350 minutes (4.5-6 hours)**

**New integrated approach:**
- Model loading: ~2 minutes (once)
- Processing: ~2-3 minutes per file = **140-210 minutes**
- **Total: 142-212 minutes (2.5-3.5 hours)**

**Savings: ~2-3 hours!** 🚀

## Output Structure

Same as before - each file gets its own directory:

```
output/
├── filename1/
│   ├── filename1.mmd
│   ├── filename1_det.mmd
│   ├── filename1_layouts.pdf
│   └── images/
├── filename2/
│   └── ...
```

## Monitoring

```bash
# Watch progress
watch -n 5 "cat processing_results.json | jq '.processed, .successful, .failed'"

# Check GPU usage
watch -n 2 nvidia-smi
```

## Technical Details

The integrated version:
1. Imports all modules directly (no subprocess)
2. Registers the model with vllm once
3. Creates a single LLM instance
4. Loops through files using the same instance
5. Properly manages GPU memory throughout

## Troubleshooting

If you get OOM (Out of Memory) errors:
- The script processes one file at a time to minimize memory usage
- Consider reducing `gpu_memory_utilization` in the code (currently 0.9)
- Check GPU memory: `nvidia-smi`

## Credits

Based on the original `run_dpsk_ocr_pdf.py` script, refactored for efficient batch processing.





