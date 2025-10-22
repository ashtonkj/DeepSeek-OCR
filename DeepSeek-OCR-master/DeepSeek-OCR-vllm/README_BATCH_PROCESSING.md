# Batch Document Processing Script

## Overview

The `process_all_pdfs.py` script processes all PDF and EPUB files from the input directory, creating a **separate output directory for each file**.

## Configuration

Edit these variables in the script if needed:

```python
INPUT_DIR = '/home/kevin/development/knowledge-base-documents/Documents'
OUTPUT_BASE_DIR = '/home/kevin/development/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/output/'
SCRIPT_PATH = '/home/kevin/development/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py'
RESULTS_FILE = '/home/kevin/development/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/processing_results.json'
```

## Output Structure

Each file gets its own subdirectory in the output folder:

```
output/
├── filename1/
│   ├── filename1.mmd
│   ├── filename1_det.mmd
│   ├── filename1_layouts.pdf
│   └── images/
│       ├── 0_0.jpg
│       └── ...
├── filename2/
│   ├── filename2.mmd
│   ├── filename2_det.mmd
│   ├── filename2_layouts.pdf
│   └── images/
│       └── ...
└── ...
```

### Example:

For a file named `Kung_Fu_Basics.pdf`, the output will be in:
```
output/Kung_Fu_Basics/
  ├── Kung_Fu_Basics.mmd
  ├── Kung_Fu_Basics_det.mmd
  ├── Kung_Fu_Basics_layouts.pdf
  └── images/
```

## Usage

```bash
cd /home/kevin/development/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm
python3 process_all_pdfs.py
```

## Supported Formats

- ✅ **PDF** - Portable Document Format
- ✅ **EPUB** - Electronic Publication

PyMuPDF (fitz) also supports: MOBI, XPS, CBZ, FB2, and more

## Features

- **Per-file directories**: Each document's output is isolated in its own folder
- **Progress tracking**: Shows [X/Y] for each file being processed
- **Error handling**: 10-minute timeout per file, captures errors
- **Persistent results**: JSON file (`processing_results.json`) updated after each file
- **Detailed logging**: Stores stdout/stderr, return codes, timestamps, and output directory paths

## Results File

The `processing_results.json` includes:

```json
{
  "start_time": "2025-10-21T...",
  "output_mode": "per_file_directory",
  "output_base_directory": "/path/to/output/",
  "files": {
    "filename.pdf": {
      "input_path": "/path/to/input/filename.pdf",
      "output_directory": "/path/to/output/filename/",
      "success": true,
      "processed_at": "2025-10-21T..."
    }
  }
}
```

## Statistics

Current document counts:
- **PDFs:** 55 files
- **EPUBs:** 15 files
- **Total:** 70 files

**Estimated runtime:** 2.5-6 hours (2-5 minutes per file)

## Monitoring Progress

```bash
# Watch the results file
watch -n 5 "cat processing_results.json | jq '.processed, .successful, .failed'"

# Count output directories created
ls -1d output/*/ | wc -l

# Check latest processed file
ls -lt output/ | head -5
```

## Benefits of Per-File Directories

1. **Organization**: Easy to find outputs for specific documents
2. **No conflicts**: Image files won't overwrite each other
3. **Easy cleanup**: Delete individual result directories if needed
4. **Parallel processing**: Could potentially run multiple instances safely (future enhancement)
5. **Archive friendly**: Can zip/tar individual results


