from .config.NodeCategory import NodeCategory
import folder_paths
from nodes import SaveImage
import random
from PIL import Image,ImageOps
import os
import numpy as np
import json
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args


class SaveImageUnifiedOutput:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"})
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "JSON")
    FUNCTION = "save_images"
    CATEGORY = NodeCategory.CATEGORY

    def save_images(self, images, filename_prefix):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            filename_with_batch_num = filename.replace(
                "%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file),
                     pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return (images, results)


NODE_CLASS_MAPPINGS = {
    "VcSaveImage": SaveImageUnifiedOutput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VcSaveImage": "😋Save Image Unified Output"
}
