import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
import torch
import os
import time

class OldPhotoColorizationNode:

    def __init__(self):
        # Initialize the colorizer pipeline
        self.colorizer = pipeline(Tasks.image_colorization, model='damo/cv_unet_image-colorization')
        # Define input directory and create it if it doesn't exist
        self.input_dir = "input/OldPhotoColorizationNode"
        os.makedirs(self.input_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize_image"
    CATEGORY = "😋fq393"

    def convert_color(self, image):
        if len(image.shape) > 2 and image.shape[2] >= 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def colorize_image(self, image):
        try:
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

            # Extract the output image from the result
            if 'output_img' in result:
                output_img = result['output_img']

                # Define output directory based on input image path
                output_dir = os.path.join(os.path.dirname(image_path), "output")
                os.makedirs(output_dir, exist_ok=True)

                # Save the output image
                output_image_path = os.path.join(output_dir, f"output_image_{timestamp}.png")
                cv2.imwrite(output_image_path, output_img)

                # Read the saved image
                output_img = cv2.imread(output_image_path)

                # Convert BGR to RGB
                output_img = self.convert_color(output_img)

                # Normalize to [0, 1] range
                output_img = output_img.astype(np.float32) / 255.0

                # Convert HWC to CHW
                output_img = np.transpose(output_img, (2, 0, 1))

                # Ensure the output is in the correct shape and type
                if len(output_img.shape) == 3:
                    output_img = output_img[np.newaxis, ...]

                output_img = torch.from_numpy(output_img)

                return (output_img,)
            else:
                return ("Error: 'output_img' not found in the result.",)
        except Exception as e:
            return (f"Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "OldPhotoColorizationNode": OldPhotoColorizationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OldPhotoColorizationNode": "😋Old Photo Colorization Node"
}

