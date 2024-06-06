import logging
import os
import numpy as np
from scipy.io import wavfile
import folder_paths

# from py.utils import folder_paths

output = folder_paths.get_output_directory()


class Waveform2Audio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "waveform": ("WAVEFORM",),
                "sample_rate": ("INT", {
                    "forceInput": True,
                    "default": 30000,
                    "min": 1000,
                    "max": 48000,
                    "step": 1000,
                    "display": "number"
                })
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "convert"
    CATEGORY = "😋ZMG/MatthewHan"

    @staticmethod
    def convert(waveform, sample_rate):
        # 将 waveform 数据转换成音频文件
        waveform = np.asarray(waveform)

        std_value = np.std(waveform)
        filename = f"audio_ldm2_std_{std_value:.2f}"

        if not os.path.exists(output):
            os.makedirs(output)
        output_file = os.path.join(output, f"{filename}.wav")

        count = 1
        while os.path.exists(output_file):
            output_file = os.path.join(output, f"{filename}_{count}.wav")
            count += 1

        waveform = np.int16(waveform / np.max(np.abs(waveform)) * 32767)
        wavfile.write(output_file, sample_rate, waveform)
        logging.info("sample_rate", sample_rate)
        logging.info("输出的目录", output_file)
        return (output_file,)


NODE_CLASS_MAPPINGS = {
    "Waveform2Audio": Waveform2Audio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Waveform2Audio": "Waveform to Audio Converter"
}
