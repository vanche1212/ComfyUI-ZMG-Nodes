import json

class JsonParserNode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": (["json", "string"], {"default": "json"}),
                "input_string": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "json_path": ("STRING", {
                    "multiline": False,
                    "default": ""
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "parse_input"
    CATEGORY = "😋ZMG/fq393"

    def parse_input(self, input_type, input_string, json_path):
        if input_type == "json":
            try:
                parsed_data = json.loads(input_string)
                keys = json_path.split('.')
                for key in keys:
                    if key in parsed_data:
                        parsed_data = parsed_data[key]
                    else:
                        return (f"Key '{key}' not found",)
                return (str(parsed_data),)
            except json.JSONDecodeError:
                return (f"Invalid JSON format",)
            except Exception as e:
                return (f"Error: {str(e)}",)
        else:
            return (input_string,)

NODE_CLASS_MAPPINGS = {
    "JsonParserNode": JsonParserNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JsonParserNode": "😋JSON Parser Node"
}

