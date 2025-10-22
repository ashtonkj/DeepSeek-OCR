#!/usr/bin/env python3
"""
Integrated batch processor for PDF and EPUB files with DeepSeek OCR
Loads the model once and processes all files efficiently
"""

import os
import fitz
import img2pdf
import io
import re
import json
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# CUDA setup
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

from config import MODEL_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
import signal
from contextlib import contextmanager
import psutil
import gc

# Configuration
INPUT_DIR = '/home/kevin/development/knowledge-base-documents/Documents'
OUTPUT_BASE_DIR = '/home/kevin/development/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/output/'
RESULTS_FILE = '/home/kevin/development/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/processing_results.json'
SUPPORTED_EXTENSIONS = ('.pdf', '.epub')
BATCH_SIZE = 5  # Process pages in batches to avoid memory issues
MAX_BOUNDING_BOXES = 1000  # Limit boxes per page to prevent hangs
PAGE_TIMEOUT = 30  # Timeout in seconds for processing each page

# Colors for output
class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB


def get_document_files(directory):
    """Get all supported document files (PDF, EPUB) in the directory"""
    doc_files = []
    for file in os.listdir(directory):
        if file.lower().endswith(SUPPORTED_EXTENSIONS):
            full_path = os.path.join(directory, file)
            if os.path.isfile(full_path):
                doc_files.append(file)
    return sorted(doc_files)


def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """Convert document pages to images"""
    images = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
        
        images.append(img)
    
    pdf_document.close()
    return images


def create_pdf_from_temp_files(temp_dir, output_path, num_pages, chunk_size=50):
    """Create PDF from temporary image files, loading in chunks to save memory"""
    try:
        from PyPDF2 import PdfMerger
    except ImportError:
        print(f"{Colors.RED}PyPDF2 not available for chunked processing{Colors.RESET}")
        return
    
    merger = PdfMerger()
    num_chunks = (num_pages + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_pages)
        
        print(f"{Colors.BLUE}    PDF chunk {chunk_idx + 1}/{num_chunks} (pages {start_idx + 1}-{end_idx})...{Colors.RESET}")
        
        # Load images for this chunk from temp files
        chunk_images = []
        for page_idx in range(start_idx, end_idx):
            temp_img_path = os.path.join(temp_dir, f'page_{page_idx:04d}.jpg')
            if os.path.exists(temp_img_path):
                img = Image.open(temp_img_path)
                chunk_images.append(img)
        
        if not chunk_images:
            continue
        
        # Convert chunk to PDF
        image_bytes_list = []
        for img in chunk_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=85)
            img_bytes = img_buffer.getvalue()
            image_bytes_list.append(img_bytes)
        
        # Convert to PDF and merge
        chunk_pdf_bytes = img2pdf.convert(image_bytes_list)
        merger.append(io.BytesIO(chunk_pdf_bytes))
        
        # Free memory after each chunk
        del chunk_images, image_bytes_list, chunk_pdf_bytes
        gc.collect()
    
    # Write final merged PDF
    with open(output_path, "wb") as f:
        merger.write(f)
    merger.close()


def pil_to_pdf_img2pdf(pil_images, output_path, chunk_size=50):
    """Convert PIL images to PDF, processing in chunks for memory efficiency"""
    if not pil_images:
        return
    
    try:
        # For small documents, process all at once
        if len(pil_images) <= chunk_size:
            image_bytes_list = []
            for img in pil_images:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=85)
                img_bytes = img_buffer.getvalue()
                image_bytes_list.append(img_bytes)
            
            pdf_bytes = img2pdf.convert(image_bytes_list)
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
        else:
            # For large documents, use PyPDF2 to merge chunks
            try:
                from PyPDF2 import PdfMerger
            except ImportError:
                # Fallback to processing all at once if PyPDF2 not available
                print(f"{Colors.YELLOW}  PyPDF2 not available, processing all pages at once (may be slow)...{Colors.RESET}")
                image_bytes_list = []
                for img in pil_images:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='JPEG', quality=85)
                    img_bytes = img_buffer.getvalue()
                    image_bytes_list.append(img_bytes)
                
                pdf_bytes = img2pdf.convert(image_bytes_list)
                with open(output_path, "wb") as f:
                    f.write(pdf_bytes)
                return
            
            # Process in chunks and merge
            merger = PdfMerger()
            num_chunks = (len(pil_images) + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(pil_images))
                chunk_images = pil_images[start_idx:end_idx]
                
                # Convert chunk to PDF
                image_bytes_list = []
                for img in chunk_images:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='JPEG', quality=85)
                    img_bytes = img_buffer.getvalue()
                    image_bytes_list.append(img_bytes)
                
                chunk_pdf_bytes = img2pdf.convert(image_bytes_list)
                merger.append(io.BytesIO(chunk_pdf_bytes))
                
                # Progress indicator
                print(f"{Colors.BLUE}    PDF chunk {chunk_idx + 1}/{num_chunks} ({start_idx + 1}-{end_idx})...{Colors.RESET}")
                
                # Free memory
                del image_bytes_list, chunk_pdf_bytes
            
            # Write final merged PDF
            with open(output_path, "wb") as f:
                merger.write(f)
            merger.close()
            
    except Exception as e:
        print(f"{Colors.RED}Error creating PDF: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()


def re_match(text):
    """Extract references from OCR output"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """Extract coordinates from reference text"""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx, output_path):
    """Draw bounding boxes on image and extract sub-images"""
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20, )
                
                for points in points_list:
                    x1, y1, x2, y2 = points
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{output_path}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_single_image(image, prompt, processor):
    """Process single image for OCR"""
    cache_item = {
        "prompt": prompt,
        "multi_modal_data": {"image": processor.tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)},
    }
    return cache_item


def process_single_file(input_path, output_dir, llm, sampling_params, processor):
    """Process a single document file"""
    try:
        print(f"{Colors.YELLOW}Loading document: {os.path.basename(input_path)}{Colors.RESET}")
        
        # Load images from document
        images = pdf_to_images_high_quality(input_path)
        print(f"{Colors.GREEN}Loaded {len(images)} pages{Colors.RESET}")
        
        # Prepare output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/images', exist_ok=True)
        temp_annotated_dir = f'{output_dir}/.temp_annotated'
        os.makedirs(temp_annotated_dir, exist_ok=True)
        
        # Process images in batches to avoid memory issues
        print(f"{Colors.YELLOW}Processing {len(images)} pages in batches of {BATCH_SIZE}...{Colors.RESET}")
        outputs_list = []
        
        # Split images into batches
        num_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(images))
            batch_images = images[start_idx:end_idx]
            
            print(f"{Colors.BLUE}  Batch {batch_idx + 1}/{num_batches}: pages {start_idx + 1}-{end_idx}{Colors.RESET}")
            
            # Pre-process images in this batch
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                batch_inputs = list(
                    executor.map(lambda img: process_single_image(img, PROMPT, processor), batch_images)
                )
            
            # Run OCR on this batch
            batch_outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
            outputs_list.extend(batch_outputs)
            
            # Free up memory
            del batch_inputs, batch_outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"{Colors.GREEN}✓ All batches processed{Colors.RESET}")
        
        # Process outputs
        base_filename = os.path.basename(input_path)
        filename_without_ext = os.path.splitext(base_filename)[0]
        
        mmd_det_path = os.path.join(output_dir, filename_without_ext + '_det.mmd')
        mmd_path = os.path.join(output_dir, filename_without_ext + '.mmd')
        pdf_out_path = os.path.join(output_dir, filename_without_ext + '_layouts.pdf')
        
        # Process OCR outputs with progress indicator - STREAM TO DISK to avoid OOM
        mem_start = get_memory_usage()
        print(f"{Colors.YELLOW}Post-processing {len(images)} pages (streaming everything to disk, RAM: {mem_start:.1f} GB)...{Colors.RESET}")
        skipped_pages = []
        annotated_page_count = 0
        
        # Open files for streaming write to avoid memory buildup
        with open(mmd_det_path, 'w', encoding='utf-8') as f_det, \
             open(mmd_path, 'w', encoding='utf-8') as f_out:
            
            for jdx, (output, img) in enumerate(tqdm(zip(outputs_list, images), total=len(images), desc="Post-processing")):
                try:
                    content = output.outputs[0].text

                    if '<｜end▁of▁sentence｜>' in content:
                        content = content.replace('<｜end▁of▁sentence｜>', '')
                    else:
                        if SKIP_REPEAT:
                            continue
                    
                    page_num = f'\n<--- Page Split --->'
                    
                    # Write detection output immediately to disk
                    f_det.write(content + f'\n{page_num}\n')
                    f_det.flush()  # Force write to disk

                    # Extract and draw bounding boxes with error handling
                    # SAVE TO TEMP FILE instead of keeping in memory
                    try:
                        matches_ref, matches_images, mathes_other = re_match(content)
                        result_image = draw_bounding_boxes(img.copy(), matches_ref, jdx, output_dir)
                        # Save annotated image to temp file
                        temp_img_path = os.path.join(temp_annotated_dir, f'page_{jdx:04d}.jpg')
                        result_image.save(temp_img_path, 'JPEG', quality=85)
                        annotated_page_count += 1
                    except Exception as draw_error:
                        print(f"\n{Colors.YELLOW}Warning: Could not draw boxes on page {jdx + 1}: {draw_error}{Colors.RESET}")
                        # Save original image if drawing fails
                        temp_img_path = os.path.join(temp_annotated_dir, f'page_{jdx:04d}.jpg')
                        img.save(temp_img_path, 'JPEG', quality=85)
                        annotated_page_count += 1
                        matches_images = []
                        mathes_other = matches_ref

                    for idx, a_match_image in enumerate(matches_images):
                        content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')

                    for idx, a_match_other in enumerate(mathes_other):
                        content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

                    # Write cleaned output immediately to disk
                    f_out.write(content + f'\n{page_num}\n')
                    f_out.flush()  # Force write to disk
                    
                    # Free the original image after processing to save memory
                    images[jdx] = None
                    
                    # Aggressive memory cleanup for large documents
                    if (jdx + 1) % 20 == 0:
                        gc.collect()
                        mem_gb = get_memory_usage()
                        print(f"\n{Colors.BLUE}  Memory cleanup at page {jdx + 1} (RAM: {mem_gb:.1f} GB){Colors.RESET}")
                        
                except Exception as page_error:
                    print(f"\n{Colors.RED}Error processing page {jdx + 1}: {page_error}{Colors.RESET}")
                    skipped_pages.append(jdx + 1)
                    # Write error placeholder
                    error_msg = f'\n[Error processing page {jdx + 1}]\n{page_num}\n'
                    f_out.write(error_msg)
                    f_det.write(error_msg)
                    # Save placeholder image
                    if img is not None:
                        temp_img_path = os.path.join(temp_annotated_dir, f'page_{jdx:04d}.jpg')
                        img.save(temp_img_path, 'JPEG', quality=85)
                        annotated_page_count += 1
        
        if skipped_pages:
            print(f"{Colors.YELLOW}Warning: Skipped {len(skipped_pages)} pages due to errors: {skipped_pages}{Colors.RESET}")
        
        # Free outputs_list now that we're done with it
        del outputs_list
        gc.collect()
        
        mem_after = get_memory_usage()
        print(f"{Colors.GREEN}✓ Text outputs saved (streamed to disk, RAM: {mem_after:.1f} GB){Colors.RESET}")
        
        # Create PDF with layout annotations by loading temp files in chunks
        print(f"{Colors.YELLOW}Creating layout PDF ({annotated_page_count} pages, RAM: {mem_after:.1f} GB)...{Colors.RESET}")
        create_pdf_from_temp_files(temp_annotated_dir, pdf_out_path, annotated_page_count)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_annotated_dir)
        print(f"{Colors.BLUE}✓ Cleaned up temporary files{Colors.RESET}")
        
        # Save page count before cleanup
        total_pages = len(images)
        
        # Free memory (outputs_list and draw_images no longer exist)
        del images
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        mem_final = get_memory_usage()
        print(f"{Colors.GREEN}✓ Final RAM usage: {mem_final:.1f} GB{Colors.RESET}")
        
        print(f"{Colors.GREEN}✓ Successfully processed{Colors.RESET}")
        return {
            'success': True,
            'error': None,
            'pages': total_pages
        }
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error processing file: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'pages': 0
        }


def main():
    """Main processing loop"""
    print("=" * 80)
    print("DeepSeek OCR Integrated Batch Processor")
    print("=" * 80)
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Base Directory: {OUTPUT_BASE_DIR}")
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
    
    # Initialize model (ONCE for all files)
    print(f"{Colors.RED}Initializing DeepSeek OCR model... (this may take a minute){Colors.RESET}")
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
    
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        disable_mm_preprocessor_cache=True
    )
    
    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    
    processor = DeepseekOCRProcessor()
    print(f"{Colors.GREEN}✓ Model initialized successfully{Colors.RESET}\n")
    
    # Initialize results
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
        filename_without_ext = os.path.splitext(filename)[0]
        file_output_dir = os.path.join(OUTPUT_BASE_DIR, filename_without_ext)
        
        if not file_output_dir.endswith('/'):
            file_output_dir += '/'
        
        print("=" * 80)
        print(f"[{idx}/{total_files}] Processing: {filename}")
        print(f"  Input: {input_path}")
        print(f"  Output: {file_output_dir}")
        print("=" * 80)
        
        # Process the file
        file_result = process_single_file(input_path, file_output_dir, llm, sampling_params, processor)
        
        # Update results
        results['processed'] += 1
        if file_result['success']:
            results['successful'] += 1
        else:
            results['failed'] += 1
        
        # Store file-specific results
        results['files'][filename] = {
            'index': idx,
            'input_path': input_path,
            'output_directory': file_output_dir,
            'processed_at': datetime.now().isoformat(),
            **file_result
        }
        
        # Save results after each file
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


