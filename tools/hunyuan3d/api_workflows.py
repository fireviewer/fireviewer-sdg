"""ComfyUI API prompt builders for the two-phase Asset4Sim Hunyuan3D batch."""

from __future__ import annotations

from typing import Any


Prompt = dict[str, dict[str, Any]]


def geometry_prompt(image_name: str, filename_prefix: str, seed: int) -> Prompt:
    """Generate an undecimated source mesh from one reference image."""

    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {
            "class_type": "ImageResize+",
            "inputs": {
                "image": ["1", 0],
                "width": 518,
                "height": 518,
                "interpolation": "lanczos",
                "method": "pad",
                "condition": "always",
                "multiple_of": 2,
            },
        },
        "3": {
            "class_type": "TransparentBGSession+",
            "inputs": {"mode": "base", "use_jit": True},
        },
        "4": {
            "class_type": "ImageRemoveBackground+",
            "inputs": {"rembg_session": ["3", 0], "image": ["2", 0]},
        },
        "5": {
            "class_type": "Hy3DModelLoader",
            "inputs": {
                "model": "hy3dgen/hunyuan3d-dit-v2-0-fp16.safetensors",
                "attention_mode": "sdpa",
                "cublas_ops": False,
            },
        },
        "6": {
            "class_type": "Hy3DGenerateMesh",
            "inputs": {
                "pipeline": ["5", 0],
                "image": ["2", 0],
                "mask": ["4", 1],
                "guidance_scale": 5.5,
                "steps": 50,
                "seed": int(seed),
                "scheduler": "FlowMatchEulerDiscreteScheduler",
                "force_offload": True,
            },
        },
        "7": {
            "class_type": "Hy3DVAEDecode",
            "inputs": {
                "vae": ["5", 1],
                "latents": ["6", 0],
                "box_v": 1.01,
                "octree_resolution": 384,
                "num_chunks": 32_000,
                "mc_level": 0.0,
                "mc_algo": "mc",
                "enable_flash_vdm": True,
                "force_offload": True,
            },
        },
        "8": {
            "class_type": "Hy3DExportMesh",
            "inputs": {
                "trimesh": ["7", 0],
                "filename_prefix": filename_prefix,
                "file_format": "glb",
                "save_file": True,
            },
        },
    }


def texture_prompt(
    image_name: str,
    mesh_path: str,
    filename_prefix: str,
    seed: int,
) -> Prompt:
    """UV unwrap and texture a repaired low-poly mesh using the reference."""

    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {
            "class_type": "ImageResize+",
            "inputs": {
                "image": ["1", 0],
                "width": 518,
                "height": 518,
                "interpolation": "lanczos",
                "method": "pad",
                "condition": "always",
                "multiple_of": 2,
            },
        },
        "3": {
            "class_type": "TransparentBGSession+",
            "inputs": {"mode": "base", "use_jit": True},
        },
        "4": {
            "class_type": "ImageRemoveBackground+",
            "inputs": {"rembg_session": ["3", 0], "image": ["2", 0]},
        },
        "5": {
            "class_type": "SolidMask",
            "inputs": {"value": 0.8, "width": 518, "height": 518},
        },
        "6": {"class_type": "MaskToImage", "inputs": {"mask": ["5", 0]}},
        "7": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["6", 0],
                "source": ["2", 0],
                "mask": ["4", 1],
                "x": 0,
                "y": 0,
                "resize_source": False,
            },
        },
        "8": {
            "class_type": "DownloadAndLoadHy3DDelightModel",
            "inputs": {"model": "hunyuan3d-delight-v2-0"},
        },
        "9": {
            "class_type": "Hy3DDiffusersSchedulerConfig",
            "inputs": {"pipeline": ["8", 0], "scheduler": "Euler A", "sigmas": "default"},
        },
        "10": {
            "class_type": "Hy3DDelightImage",
            "inputs": {
                "delight_pipe": ["8", 0],
                "image": ["7", 0],
                "scheduler": ["9", 0],
                "steps": 50,
                "width": 512,
                "height": 512,
                "cfg_image": 1.0,
                "seed": int(seed),
            },
        },
        "11": {"class_type": "Hy3DLoadMesh", "inputs": {"glb_path": mesh_path}},
        "12": {"class_type": "Hy3DMeshUVWrap", "inputs": {"trimesh": ["11", 0]}},
        "13": {
            "class_type": "Hy3DCameraConfig",
            "inputs": {
                "camera_azimuths": "0, 90, 180, 270, 0, 180",
                "camera_elevations": "0, 0, 0, 0, 90, -90",
                "view_weights": "1, 0.1, 0.5, 0.1, 0.05, 0.05",
                "camera_distance": 1.45,
                "ortho_scale": 1.2,
            },
        },
        "14": {
            "class_type": "Hy3DRenderMultiView",
            "inputs": {
                "trimesh": ["12", 0],
                "camera_config": ["13", 0],
                "render_size": 1024,
                "texture_size": 2048,
                "normal_space": "world",
            },
        },
        "15": {
            "class_type": "DownloadAndLoadHy3DPaintModel",
            "inputs": {"model": "hunyuan3d-paint-v2-0"},
        },
        "16": {
            "class_type": "Hy3DDiffusersSchedulerConfig",
            "inputs": {"pipeline": ["15", 0], "scheduler": "Euler A", "sigmas": "default"},
        },
        "17": {
            "class_type": "Hy3DSampleMultiView",
            "inputs": {
                "pipeline": ["15", 0],
                "ref_image": ["10", 0],
                "normal_maps": ["14", 0],
                "position_maps": ["14", 1],
                "camera_config": ["13", 0],
                "scheduler": ["16", 0],
                "view_size": 512,
                "steps": 25,
                "seed": int(seed),
                "denoise_strength": 1.0,
            },
        },
        "18": {
            "class_type": "ImageResize+",
            "inputs": {
                "image": ["17", 0],
                "width": 2048,
                "height": 2048,
                "interpolation": "lanczos",
                "method": "stretch",
                "condition": "always",
                "multiple_of": 0,
            },
        },
        "19": {
            "class_type": "Hy3DBakeFromMultiview",
            "inputs": {
                "images": ["18", 0],
                "renderer": ["14", 2],
                "camera_config": ["13", 0],
            },
        },
        "20": {
            "class_type": "Hy3DMeshVerticeInpaintTexture",
            "inputs": {"texture": ["19", 0], "mask": ["19", 1], "renderer": ["19", 2]},
        },
        "21": {
            "class_type": "CV2InpaintTexture",
            "inputs": {
                "texture": ["20", 0],
                "mask": ["20", 1],
                "inpaint_radius": 3,
                "inpaint_method": "ns",
            },
        },
        "22": {
            "class_type": "Hy3DApplyTexture",
            "inputs": {"texture": ["21", 0], "renderer": ["20", 2]},
        },
        "23": {
            "class_type": "Hy3DExportMesh",
            "inputs": {
                "trimesh": ["22", 0],
                "filename_prefix": filename_prefix,
                "file_format": "glb",
                "save_file": True,
            },
        },
    }
