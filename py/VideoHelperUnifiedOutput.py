from .videohelpersuite.nodes import LoadVideoPath, VideoCombine
from .config.NodeCategory import NodeCategory


class VideoCombineUnifiedOutput:
    @classmethod
    def INPUT_TYPES(s):
        return VideoCombine.INPUT_TYPES()

    CATEGORY = NodeCategory.CATEGORY
    RETURN_TYPES = VideoCombine.RETURN_TYPES
    RETURN_NAMES = ("result",)
    FUNCTION = 'combine_video'

    def combine_video(self, **kwargs):
        return VideoCombine().combine_video(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return VideoCombine.IS_CHANGED(video, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return VideoCombine.VALIDATE_INPUTS(video, **kwargs)


class LoadVideoPathUnifiedOutput:
    @classmethod
    def INPUT_TYPES(s):
        return LoadVideoPath.INPUT_TYPES()

    CATEGORY = NodeCategory.CATEGORY

    RETURN_TYPES = LoadVideoPath.RETURN_TYPES + ("FLOAT",)
    RETURN_NAMES = LoadVideoPath.RETURN_NAMES + ("fps",)
    FUNCTION = 'load_video'

    def load_video(self, **kwargs):
        return LoadVideoPath().load_video(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return LoadVideoPath.IS_CHANGED(video, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return LoadVideoPath.VALIDATE_INPUTS(video, **kwargs)


NODE_CLASS_MAPPINGS = {
    "VHS_Video_Combine_Unified_Output": VideoCombineUnifiedOutput,
    "VHS_Load_Video_Path_Unified_Output": LoadVideoPathUnifiedOutput
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_Video_Combine_Unified_Output": "😋Video Combine Unified Output",
    "VHS_Load_Video_Path_Unified_Output": "😋Load Video Path Unified Output"
}
