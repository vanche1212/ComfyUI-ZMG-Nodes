import requests


class FileUploader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "display": "string"
                })
            },
            "optional": {
                "upload_button": ("BUTTON", {
                    "label": "Upload File",
                    "command": "upload_file"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "upload_file"
    CATEGORY = "Utilities"

    def upload_file(self, file_path):
        url = "http://10.27.89.24:8288/upload/image"
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(url, files={'file': f})
            if response.status_code == 200:
                return (f"File uploaded successfully to {url}",)
            else:
                return (f"Failed to upload file: {response.content.decode()}",)
        except Exception as e:
            return (f"Failed to upload file: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "FileUploader": FileUploader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FileUploader": "File Uploader"
}
