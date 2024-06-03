import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
import torch
import os
import time
import folder_paths


class OldPhotoColorizationNode:

    def __init__(self):
        # Initialize the colorizer pipeline
        self.colorizer = pipeline(Tasks.image_colorization, model='damo/cv_unet_image-colorization')
        # Define input directory and create it if it doesn't exist
        self.input_dir = folder_paths.get_input_directory()
        os.makedirs(self.input_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Changed to plural to handle multiple images
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize_images"
    CATEGORY = "😋fq393"

    def colorize_images(self, images):
        try:
            output_images = []
            for image in images:
                # Convert the input tensor to a numpy array if needed
                if isinstance(image, torch.Tensor):
                    image = image.squeeze().cpu().numpy()

                # Convert to [0, 255] range and to uint8
                image = (image * 255).astype(np.uint8)

                # Convert CHW to HWC if necessary
                if image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))

                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(image)

                # Generate a unique filename with timestamp
                timestamp = int(time.time())
                image_path = os.path.join(self.input_dir, f"input_image_{timestamp}.jpg")
                pil_image.save(image_path)

                # Colorize the image
                result = self.colorizer(image_path)

                # Delete the temporary image
                os.remove(image_path)

                # Extract the output image from the result
                if 'output_img' in result:
                    output_img = result['output_img']
                    # Convert to the proper format
                    output_img = Image.fromarray((output_img * 255).astype(np.uint8))
                    output_img = output_img.convert('RGB')
                    output_img = np.array(output_img).astype(np.float32) / 255.0
                    output_img = torch.from_numpy(output_img)[None,]
                    output_images.append(output_img)
                else:
                    output_images.append(f"Error: 'output_img' not found in the result.")
            return output_images
        except Exception as e:
            return (f"Error: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "OldPhotoColorizationNode": OldPhotoColorizationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OldPhotoColorizationNode": "😋Old Photo Colorization Node"
}
