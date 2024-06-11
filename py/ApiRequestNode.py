import time
import random


class APIRequestNode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_url": ("STRING", {"default": "http://example.com/api"}),
                "request_method": (["GET", "POST"], {"default": "GET"}),
                "data_format": (["json", "form"], {"default": "json"}),
                "request_params": ("STRING", {"multiline": True, "default": "{}"}),
                "headers": ("STRING", {"multiline": True, "default": "{}"}),
                "any_input": ("object", {"widget": False})
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "make_request"
    CATEGORY = "😋ZMG/fq393"

    def make_request(self, api_url, request_method, data_format, request_params, headers, any_input):
        import requests
        import json

        # 解析请求参数和头信息
        try:
            params = json.loads(request_params)
        except json.JSONDecodeError:
            params = {}

        try:
            header_dict = json.loads(headers)
        except json.JSONDecodeError:
            header_dict = {}

        # 设置超时机制
        timeout = 10

        # 根据请求方法和数据格式调用接口
        try:
            if request_method == "GET":
                response = requests.get(api_url, params=params, headers=header_dict, timeout=timeout)
            elif request_method == "POST":
                if data_format == "json":
                    response = requests.post(api_url, json=params, headers=header_dict, timeout=timeout)
                else:
                    response = requests.post(api_url, data=params, headers=header_dict, timeout=timeout)

            # 返回响应内容
            if response.status_code == 200:
                return (response.text,)
            else:
                return (f"Error: {response.status_code} - {response.text}",)
        except requests.exceptions.Timeout:
            return ("Error: Request timed out",)
        except requests.exceptions.RequestException as e:
            return (f"Error: {e}",)


NODE_CLASS_MAPPINGS = {
    "APIRequestNode": APIRequestNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIRequestNode": "😋API Request Node"
}
