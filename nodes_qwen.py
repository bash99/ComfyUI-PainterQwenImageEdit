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
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
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
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Qwen image editing with 8-image support, outputs positive and empty negative conditioning"

    def encode(self, clip, prompt, negative_prompt="", vae=None, background=None, image1=None, image2=None,
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

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)

        negative_tokens = clip.tokenize(negative_prompt)
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
        if len(ref_latents) > 0:
            negative_conditioning = node_helpers.conditioning_set_values(negative_conditioning, {"reference_latents": ref_latents}, append=True)

        return (conditioning, negative_conditioning)


class PainterQwenImageEditPlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image1_mask": ("MASK",),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Pixel-perfect Qwen image editing with mask support"

    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None, 
               image1_mask=None, width=1024, height=1024):
        
        images = [image1, image2, image3]
        ref_latents = []
        vl_images = []
        noise_mask = None
        
        # Determine pixel-perfect mode based on aspect ratio match
        pixel_perfect = False
        if image1 is not None:
            _, img_h, img_w, _ = image1.shape
            input_ar = img_w / img_h
            target_ar = width / height
            if abs(input_ar - target_ar) < 0.01:  # 1% tolerance
                pixel_perfect = True
        
        ref_longest_edge = max(width, height)
        
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                current_total = samples.shape[3] * samples.shape[2]
                
                # VL processing (384x384)
                vl_total = int(384 * 384)
                vl_scale_by = math.sqrt(vl_total / current_total)
                vl_width = round(samples.shape[3] * vl_scale_by)
                vl_height = round(samples.shape[2] * vl_scale_by)
                
                s_vl = comfy.utils.common_upscale(samples, vl_width, vl_height, "area", "center")
                vl_image = s_vl.movedim(1, -1)
                vl_images.append(vl_image)
                
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                
                if vae is not None:
                    ori_longest_edge = max(samples.shape[2], samples.shape[3])
                    scale_by = ori_longest_edge / ref_longest_edge
                    scaled_width = int(round(samples.shape[3] / scale_by))
                    scaled_height = int(round(samples.shape[2] / scale_by))
                    
                    # Choose crop mode based on pixel-perfect detection
                    if pixel_perfect:
                        # Use pad mode for no pixel shift
                        canvas_width = math.ceil(scaled_width / 8.0) * 8
                        canvas_height = math.ceil(scaled_height / 8.0) * 8
                        
                        canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        
                        crop = "center"
                        resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, "lanczos", crop)
                        resized_width = resized_samples.shape[3]
                        resized_height = resized_samples.shape[2]
                        
                        canvas[:, :, :resized_height, :resized_width] = resized_samples
                        s = canvas
                        
                        # Handle mask for image1
                        if i == 0 and image1_mask is not None:
                            mask = image1_mask
                            # Fix mask shape
                            if mask.dim() == 2:
                                mask_samples = mask.unsqueeze(0).unsqueeze(0)
                            elif mask.dim() == 3:
                                mask_samples = mask.unsqueeze(1)
                            else:
                                print(f"Warning: Unexpected mask shape {mask.shape}, skipping")
                                mask_samples = None
                            
                            if mask_samples is not None:
                                resized_mask = comfy.utils.common_upscale(mask_samples, resized_width, resized_height, "area", crop)
                                noise_mask = resized_mask.squeeze(1)
                    else:
                        # Original behavior: center crop
                        crop = "center"
                        width = round(scaled_width / 8.0) * 8
                        height = round(scaled_height / 8.0) * 8
                        s = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
                        
                        if i == 0 and image1_mask is not None:
                            mask = image1_mask
                            if mask.dim() == 2:
                                mask_samples = mask.unsqueeze(0).unsqueeze(0)
                            elif mask.dim() == 3:
                                mask_samples = mask.unsqueeze(1)
                            else:
                                print(f"Warning: Unexpected mask shape {mask.shape}, skipping")
                                mask_samples = None
                            
                            if mask_samples is not None:
                                m = comfy.utils.common_upscale(mask_samples, width, height, "area", crop)
                                noise_mask = m.squeeze(1)
                    
                    image = s.movedim(1, -1)
                    ref_latent = vae.encode(image[:, :, :, :3])
                    ref_latents.append(ref_latent)
        
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        negative_tokens = clip.tokenize("")
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
        if len(ref_latents) > 0:
            negative_conditioning = node_helpers.conditioning_set_values(negative_conditioning, {"reference_latents": ref_latents}, append=True)
        
        latent_out = {"samples": torch.zeros(1, 4, 128, 128)}
        if len(ref_latents) > 0:
            latent_out["samples"] = ref_latents[0]
            if noise_mask is not None:
                latent_out["noise_mask"] = noise_mask
        
        return (conditioning, negative_conditioning, latent_out)


NODE_CLASS_MAPPINGS = {
    "PainterQwenImageEdit": PainterQwenImageEdit,
    "PainterQwenImageEditPlus": PainterQwenImageEditPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterQwenImageEdit": "Painter Qwen Image Edit",
    "PainterQwenImageEditPlus": "Painter Qwen Image Edit Plus",
}
