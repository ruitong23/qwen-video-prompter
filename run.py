import os
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
sys.path.append("qwen2.5-vl")
# ---- UI: Ask user for video folder ----
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window
VIDEO_DIR = filedialog.askdirectory(title="Select folder containing videos")

if not VIDEO_DIR:
    messagebox.showinfo("Canceled", "No folder selected. Exiting.")
    exit()

OUTPUT_DIR = os.path.join(VIDEO_DIR, "video_prompts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

# ---- SUPPORTED VIDEO FORMATS ----
SUPPORTED_EXT = [".mp4", ".mov", ".avi", ".webm", ".mkv"]

# ---- PROCESS EACH VIDEO ----
for filename in os.listdir(VIDEO_DIR):
    if not any(filename.lower().endswith(ext) for ext in SUPPORTED_EXT):
        continue

    video_path = os.path.join(VIDEO_DIR, filename)
    print(f"\n▶ Processing video: {filename}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video as a prompt. but without the characteristics and styles, just the movement and clothes, extra detail on clothes. character name owkiriko"},
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

        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Save result to .txt file
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

        print(f"✅ Saved prompt to: {output_file}")

    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")
