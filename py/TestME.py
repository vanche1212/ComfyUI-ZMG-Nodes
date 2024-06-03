import os


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
                }),
                "destination_folder": ("STRING", {
                    "default": "uploads/",
                    "multiline": False,
                    "display": "string"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "upload_file"
    CATEGORY = "😋fq393"

    def upload_file(self, file_path, destination_folder):
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        destination_path = os.path.join(destination_folder, os.path.basename(file_path))
        try:
            with open(file_path, 'rb') as fsrc:
                with open(destination_path, 'wb') as fdst:
                    fdst.write(fsrc.read())
            return (f"File uploaded to {destination_path}",)
        except Exception as e:
            return (f"Failed to upload file: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "FileUploader": FileUploader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FileUploader": "File Uploader"
}