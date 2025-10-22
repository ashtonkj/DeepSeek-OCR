#!/usr/bin/env python3
"""
Process all PDF and EPUB files in the directory with DeepSeek OCR
Tracks success/failure status in a JSON file
Each file gets its own output directory
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Configuration
INPUT_DIR = '/home/kevin/development/knowledge-base-documents/Documents'
OUTPUT_BASE_DIR = '/home/kevin/development/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/output/'
SCRIPT_PATH = '/home/kevin/development/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py'
RESULTS_FILE = '/home/kevin/development/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/processing_results.json'

# Supported formats (PyMuPDF/fitz supports these)
SUPPORTED_EXTENSIONS = ('.pdf', '.epub')

def get_document_files(directory):
    """Get all supported document files (PDF, EPUB) in the directory"""
    doc_files = []
    for file in os.listdir(directory):
        if file.lower().endswith(SUPPORTED_EXTENSIONS):
            full_path = os.path.join(directory, file)
            if os.path.isfile(full_path):
                doc_files.append(file)
    return sorted(doc_files)

def process_file(input_path, output_path, script_path):
    """Process a single PDF file with DeepSeek OCR"""
    try:
        # Set environment variables
        env = os.environ.copy()
        env['INPUT_PATH'] = input_path
        env['OUTPUT_PATH'] = output_path
        
        # Run the command
        result = subprocess.run(
            ['python', script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per file
        )
        
        return {
            'success': result.returncode == 0,
            'return_code': result.returncode,
            'stdout': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,  # Last 500 chars
            'stderr': result.stderr[-500:] if len(result.stderr) > 500 else result.stderr,  # Last 500 chars
            'error': None
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'return_code': -1,
            'stdout': '',
            'stderr': '',
            'error': 'Process timeout (>10 minutes)'
        }
    except Exception as e:
        return {
            'success': False,
            'return_code': -1,
            'stdout': '',
            'stderr': '',
            'error': str(e)
        }

def main():
    """Main processing loop"""
    print("=" * 80)
    print("DeepSeek OCR Batch Processor")
    print("=" * 80)
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Base Directory: {OUTPUT_BASE_DIR}")
    print(f"  (Each file will get its own subdirectory)")
    print(f"Results File: {RESULTS_FILE}")
    print(f"Supported Formats: {', '.join(SUPPORTED_EXTENSIONS)}")
    print()
    
    # Get all document files
    doc_files = get_document_files(INPUT_DIR)
    total_files = len(doc_files)
    
    if total_files == 0:
        print("No supported document files found in the directory!")
        return
    
    # Count by type
    pdf_count = sum(1 for f in doc_files if f.lower().endswith('.pdf'))
    epub_count = sum(1 for f in doc_files if f.lower().endswith('.epub'))
    print(f"Found {total_files} files to process:")
    print(f"  - PDFs: {pdf_count}")
    print(f"  - EPUBs: {epub_count}\n")
    
    # Initialize results structure
    results = {
        'start_time': datetime.now().isoformat(),
        'input_directory': INPUT_DIR,
        'output_base_directory': OUTPUT_BASE_DIR,
        'output_mode': 'per_file_directory',
        'total_files': total_files,
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'files': {}
    }
    
    # Process each file
    for idx, filename in enumerate(doc_files, 1):
        input_path = os.path.join(INPUT_DIR, filename)
        
        # Create per-file output directory
        filename_without_ext = os.path.splitext(filename)[0]
        file_output_dir = os.path.join(OUTPUT_BASE_DIR, filename_without_ext)
        
        # Ensure the directory exists
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Ensure OUTPUT_PATH ends with trailing slash (required by the OCR script)
        if not file_output_dir.endswith('/'):
            file_output_dir += '/'
        
        print(f"[{idx}/{total_files}] Processing: {filename}")
        print(f"  Input: {input_path}")
        print(f"  Output: {file_output_dir}")
        
        # Process the file
        file_result = process_file(input_path, file_output_dir, SCRIPT_PATH)
        
        # Update results
        results['processed'] += 1
        if file_result['success']:
            results['successful'] += 1
            print(f"  ✓ SUCCESS")
        else:
            results['failed'] += 1
            print(f"  ✗ FAILED")
            if file_result['error']:
                print(f"    Error: {file_result['error']}")
            if file_result['stderr']:
                print(f"    Stderr: {file_result['stderr'][:200]}")
        
        # Store file-specific results
        results['files'][filename] = {
            'index': idx,
            'input_path': input_path,
            'output_directory': file_output_dir,
            'processed_at': datetime.now().isoformat(),
            **file_result
        }
        
        # Save results after each file (in case of crashes)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        print()
    
    # Final summary
    results['end_time'] = datetime.now().isoformat()
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total Files: {results['total_files']}")
    print(f"Processed: {results['processed']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    print("=" * 80)

if __name__ == '__main__':
    main()

