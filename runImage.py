import os
import sys
import gc
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---- Add Qwen2.5-VL path for submodule support ----
sys.path.append("qwen2.5-vl")

# ---- UI: Ask user for image folder ----
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window
IMAGE_DIR = filedialog.askdirectory(title="Select folder containing images")

if not IMAGE_DIR:
    messagebox.showinfo("Canceled", "No folder selected. Exiting.")
    exit()

# ---- LOAD MODEL & PROCESSOR ----
print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", use_fast=False
)

# ---- SUPPORTED IMAGE FORMATS ----
SUPPORTED_EXT = [".png", ".jpg", ".jpeg"]

# ---- PROCESS EACH IMAGE ----
for filename in os.listdir(IMAGE_DIR):
    if not any(filename.lower().endswith(ext) for ext in SUPPORTED_EXT):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    print(f"\n▶ Processing image: {filename}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": "Describe this image as a prompt. But don't say anything about the characteristics and styles, just the movement and the clothes this character is wearing."
                },
            ],
        }
    ]

    try:
        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate response (with memory-safe context)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Save or append to .txt file in the same folder
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(IMAGE_DIR, f"{base_name}.txt")
        new_prompt = output_text.strip()

        if os.path.exists(output_file):
            with open(output_file, "r+", encoding="utf-8") as f:
                existing = f.read().strip()
                if existing:
                    updated = existing + ", " + new_prompt
                else:
                    updated = new_prompt
                f.seek(0)
                f.write(updated)
                f.truncate()
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(new_prompt)

        print(f"✅ Updated prompt in: {output_file}")

        # ---- CLEANUP GPU MEMORY AFTER EACH IMAGE ----
        del inputs, image_inputs, video_inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")
