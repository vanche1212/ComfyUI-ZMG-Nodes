import requests
import json

class OllamaRequestNode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "phi3:3.8b-mini-instruct-4k-fp16"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "send_request"
    CATEGORY = "😋ZMG/fq393"

    def send_request(self, model, prompt, url):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            response_data = response.json()
            if 'response' in response_data:
                return (response_data['response'],)
            else:
                return ("Response field not found in the response JSON.",)
        except requests.exceptions.RequestException as e:
            return (f"Request failed: {str(e)}",)
        except json.JSONDecodeError:
            return ("Failed to decode the response as JSON.",)

NODE_CLASS_MAPPINGS = {
    "OllamaRequestNode": OllamaRequestNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaRequestNode": "😋Ollama Request Node"
}

