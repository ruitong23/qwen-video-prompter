import os
import sys
import gc
import torch
import tkinter as tk
from PIL import Image
from tkinter import filedialog, messagebox
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---- Add Qwen2.5-VL path for submodule support ----
sys.path.append("qwen2.5-vl")

# ---- UI: Ask user for image folder ----
root = tk.Tk()
root.withdraw()
IMAGE_DIR = filedialog.askdirectory(title="Select folder containing images")

if not IMAGE_DIR:
    messagebox.showinfo("Canceled", "No folder selected. Exiting.")
    exit()

# ---- LOAD MODEL & PROCESSOR ----
print("Loading model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
).to(device).eval()

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", use_fast=False
)

# ---- SETTINGS ----
SUPPORTED_EXT = [".png", ".jpg", ".jpeg"]
MAX_PIXELS = 1280 * 720  # Limit to avoid high token count

# ---- Resize large images ----
def resize_if_needed(image_path, max_pixels=MAX_PIXELS):
    image = Image.open(image_path)
    if image.width * image.height > max_pixels:
        ratio = (max_pixels / (image.width * image.height)) ** 0.5
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size)
        image.save(image_path)

# ---- Inference per image ----
def process_image(image_path):
    filename = os.path.basename(image_path)
    print(f"\n▶ Processing image: {filename}")

    # Resize large images
    resize_if_needed(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": "Describe this image as a prompt. But don't say anything about the characteristics and styles, prompt about the movement and background, and the clothes this character is wearing in detailed, and detailed on background"
                },
            ],
        }
    ]

    try:
        # Prepare input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

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

        # Save prompt next to image
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
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

    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")

    # ---- Cleanup memory after each image ----
    del inputs, image_inputs, video_inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()
    gc.collect()

# ---- MAIN LOOP ----
for filename in os.listdir(IMAGE_DIR):
    if any(filename.lower().endswith(ext) for ext in SUPPORTED_EXT):
        full_path = os.path.join(IMAGE_DIR, filename)
        process_image(full_path)
