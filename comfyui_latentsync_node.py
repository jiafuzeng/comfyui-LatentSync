import os
import sys
import torch
import folder_paths
import uuid
from datetime import datetime
# 添加LatentSync项目路径
LATENTSYNC_PATH = os.path.dirname(os.path.abspath(__file__))
if LATENTSYNC_PATH not in sys.path:
    sys.path.append(LATENTSYNC_PATH)

from scripts.inference import main
from accelerate.utils import set_seed
from omegaconf import OmegaConf

class LatentSyncArgs:
    def __init__(self, unet_config_path, inference_ckpt_path, video_path, audio_path, video_out_path, inference_steps, guidance_scale, seed):
        self.unet_config_path = unet_config_path
        self.inference_ckpt_path = inference_ckpt_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.video_out_path = video_out_path
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

class LatentSyncNode:
    """ComfyUI节点:LatentSync唇同步生成"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False, "placeholder": "输入视频文件路径，例如: /path/to/video.mp4"}),
                "audio_path": ("STRING", {"default": "", "multiline": False, "placeholder": "输入音频文件路径，例如: /path/to/audio.wav"}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "generate_lipsync"
    CATEGORY = "LatentSync"
    
    
    def generate_lipsync(
        self,
        video_path,
        audio_path,
        num_inference_steps,
        guidance_scale,
        seed,
    ):
        """生成唇同步视频"""
        
        # 验证输入参数
        if not video_path or video_path.strip() == "":
            raise RuntimeError("视频路径不能为空，请提供有效的视频文件路径")
        if not audio_path or audio_path.strip() == "":
            raise RuntimeError("音频路径不能为空，请提供有效的音频文件路径")
            
        # 清理路径（去除可能的引号和空格）
        video_path = video_path.strip().strip('"\'')
        audio_path = audio_path.strip().strip('"\'')
        
        print(f"处理视频路径: {video_path}")
        print(f"处理音频路径: {audio_path}")

        checkpoint_path = os.path.join(LATENTSYNC_PATH,"checkpoints/latentsync_unet.pt")
        config_path = os.path.join(LATENTSYNC_PATH,"configs/unet/stage2.yaml")
        
        # 检查输入文件
        if not os.path.exists(video_path):
            raise RuntimeError(f"视频文件不存在: {video_path}")
        if not os.path.exists(audio_path):
            raise RuntimeError(f"音频文件不存在: {audio_path}")
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"检查点文件不存在: {checkpoint_path}")
        if not os.path.exists(config_path):
            raise RuntimeError(f"配置文件不存在: {config_path}")
            
        # 设置随机种子
        if seed != -1:
            set_seed(seed)
        else:
            torch.seed()
            
        print(f"初始种子: {torch.initial_seed()}")
        print(f"输入视频: {video_path}")
        print(f"输入音频: {audio_path}")
        
        # 生成输出文件路径到ComfyUI的temp目录
        temp_dir = folder_paths.get_temp_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        video_out_path = os.path.join(temp_dir, f"latentsync_{timestamp}_{unique_id}.mp4")

        os.makedirs(temp_dir, exist_ok=True)
        # 检查GPU支持
        args = LatentSyncArgs(
            unet_config_path=config_path,
            inference_ckpt_path=checkpoint_path,
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=video_out_path,
            inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        try: 
            config = OmegaConf.load(args.unet_config_path)
            main(config, args)
        except Exception as e:
            error_msg = f"生成过程中出现错误: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        return (video_out_path,)
# 注册节点
NODE_CLASS_MAPPINGS = {
    "LatentSyncNode": LatentSyncNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSyncNode": "LatentSync 唇同步"
} 