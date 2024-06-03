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
        print('>>>', self.input_dir)
        os.makedirs(self.input_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Changed to plural to handle multiple images
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize_images"
    CATEGORY = "😋fq393"

    def colorize_images(self, image):
        try:
            output_images = []
            for ig in image:
                # Convert the input tensor to a numpy array if needed
                if isinstance(ig, torch.Tensor):
                    ig = ig.squeeze().cpu().numpy()

                # Convert to [0, 255] range and to uint8
                ig = (ig * 255).astype(np.uint8)

                # Convert CHW to HWC if necessary
                if ig.shape[0] == 3:
                    ig = np.transpose(ig, (1, 2, 0))

                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(ig)

                # Generate a unique filename with timestamp
                timestamp = int(time.time())
                image_path = os.path.join(self.input_dir, f"input_image_{timestamp}.jpg")

                pil_image.save(image_path)

                # Colorize the image
                result = self.colorizer(image_path)

                # Extract the output image from the result and save it temporarily
                if 'output_img' in result:
                    output_image_path = os.path.join(self.input_dir, f"output_image_{timestamp}.jpg")
                    cv2.imwrite(output_image_path, result['output_img'])
                    output_images.append(torch.tensor(cv2.imread(output_image_path)).permute(2, 0, 1) / 255.0)
                else:
                    output_images.append(torch.zeros_like(ig))  # Append a tensor of zeros if error

            # Stack all output images into a single tensor
            output_images_tensor = torch.stack(output_images)
            return (output_images_tensor,)
        except Exception as e:
            return (f"Error: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "OldPhotoColorizationNode": OldPhotoColorizationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OldPhotoColorizationNode": "😋Old Photo Colorization Node"
}
