import os
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import folder_paths
from comfy.utils import common_upscale
from .utils.utils import calculate_file_hash, get_audio, lazy_eval, hash_path, validate_path
from .config.NodeCategory import NodeCategory

video_extensions = ['webm', 'mp4', 'mkv', 'gif']

def is_gif(filename) -> bool:
    file_parts = filename.split('.')
    return len(file_parts) > 1 and file_parts[-1] == "gif"

def target_size(width, height, force_size, custom_width, custom_height) -> tuple[int, int]:
    if force_size == "Custom":
        return (custom_width, custom_height)
    elif force_size == "Custom Height":
        force_size = "?x" + str(custom_height)
    elif force_size == "Custom Width":
        force_size = str(custom_width) + "x?"

    if force_size != "Disabled":
        force_size = force_size.split("x")
        if force_size[0] == "?":
            width = (width * int(force_size[1])) // height
            width = int(width) + 4 & ~7
            height = int(force_size[1])
        elif force_size[1] == "?":
            height = (height * int(force_size[0])) // width
            height = int(height) + 4 & ~7
            width = int(force_size[0])
        else:
            width = int(force_size[0])
            height = int(force_size[1])
    return (width, height)

def load_video_cv(video: str, force_rate: int, force_size: str,
                  custom_width: int, custom_height: int, frame_load_cap: int,
                  skip_first_frames: int, select_every_nth: int):
    try:
        video_cap = cv2.VideoCapture(video)
        if not video_cap.isOpened():
            raise ValueError(f"{video} could not be loaded with cv.")
        
        # Get FPS from the video
        fps = video_cap.get(cv2.CAP_PROP_FPS)

        images = []
        total_frame_count = 0
        total_frames_evaluated = -1
        frames_added = 0
        base_frame_time = 1 / fps
        width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if force_rate == 0:
            target_frame_time = base_frame_time
        else:
            target_frame_time = 1 / force_rate
        time_offset = target_frame_time - base_frame_time
        while video_cap.isOpened():
            if time_offset < target_frame_time:
                is_returned, frame = video_cap.read()
                if not is_returned:
                    break
                time_offset += base_frame_time
            if time_offset < target_frame_time:
                continue
            time_offset -= target_frame_time
            total_frame_count += 1
            if total_frame_count <= skip_first_frames:
                continue
            else:
                total_frames_evaluated += 1

            if total_frames_evaluated % select_every_nth != 0:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = ImageOps.exif_transpose(image)
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            images.append(image)
            frames_added += 1
            if frame_load_cap > 0 and frames_added >= frame_load_cap:
                break
    finally:
        video_cap.release()
    if len(images) == 0:
        raise RuntimeError("No frames generated")
    images = torch.cat(images, dim=0)
    if force_size != "Disabled":
        new_size = target_size(width, height, force_size, custom_width, custom_height)
        if new_size[0] != width or new_size[1] != height:
            s = images.movedim(-1, 1)
            s = common_upscale(s, new_size[0], new_size[1], "lanczos", "center")
            images = s.movedim(1, -1)
    audio = lambda: get_audio(video, skip_first_frames * target_frame_time, frame_load_cap * target_frame_time)
    return (images, frames_added, lazy_eval(audio), fps)

class LoadVideoUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
                    files.append(f)
        return {
            "required": {
                "video": (sorted(files),),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1}),
                "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                "custom_width": ("INT", {"default": 512, "min": 0, "step": 8}),
                "custom_height": ("INT", {"default": 512, "min": 0, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
        }

    CATEGORY = NodeCategory

    RETURN_TYPES = ("IMAGE", "INT", "VHS_AUDIO", "FLOAT")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "fps")
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs['video'] = folder_paths.get_annotated_filepath(kwargs['video'].strip("\""))
        return load_video_cv(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        return calculate_file_hash(image_path)

    @classmethod
    def VALIDATE_INPUTS(s, video, force_size, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True

class LoadVideoPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"default": "", "vhs_path_extensions": video_extensions}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1}),
                "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                "custom_width": ("INT", {"default": 0, "min": 0, "step": 8}),
                "custom_height": ("INT", {"default": 0, "min": 0, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
        }

    CATEGORY = "😋ZMG/fq393"

    RETURN_TYPES = ("IMAGE", "INT", "VHS_AUDIO", "FLOAT")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "fps")
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        return load_video_cv(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return validate_path(video, allow_none=True)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    # "LoadVideoUpload": LoadVideoUpload,
    "LoadVideoPath": LoadVideoPath
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # "LoadVideoUpload": "😋Load Video (Upload)",
    "LoadVideoPath": "😋Load Video (Path)"
}

