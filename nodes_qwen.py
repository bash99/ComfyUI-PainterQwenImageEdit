import node_helpers
import comfy.utils
import math
import comfy.model_management
import torch
import nodes
from typing import Any

class PainterQwenImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "background": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("正面条件", "负面条件")
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Qwen image editing with 8-image support, outputs positive and empty negative conditioning"

    def encode(self, clip, prompt, vae=None, background=None, image1=None, image2=None, 
               image3=None, image4=None, image5=None, image6=None, image7=None):
        ref_latents = []
        images = [background, image1, image2, image3, image4, image5, image6, image7]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                if i == 0:
                    image_prompt += "Background: <|vision_start|><|image_pad|><|vision_end|>"
                else:
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i)

        # 正向条件
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        # 负向零化条件
        negative_tokens = clip.tokenize("")
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
        if len(ref_latents) > 0:
            negative_conditioning = node_helpers.conditioning_set_values(negative_conditioning, {"reference_latents": ref_latents}, append=True)
            
        return (conditioning, negative_conditioning)

NODE_CLASS_MAPPINGS = {
    "PainterQwenImageEdit": PainterQwenImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterQwenImageEdit": "Painter Qwen Image Edit",
}
