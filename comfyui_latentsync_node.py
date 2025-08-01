import os
import sys
import torch
import folder_paths
import uuid
import tempfile
import numpy as np
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
                "audio": ("AUDIO", ),
                "video_path": ("STRING", {"default": "", "multiline": False, "placeholder": "输入视频文件路径，例如: /path/to/video.mp4"}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "generate_lipsync"
    CATEGORY = "LatentSync"
    
    
    def save_audio_tensor_to_file(self, audio_tensor, sample_rate, temp_dir):
        """将音频张量保存为临时文件"""
        import soundfile as sf
        
        # 创建临时音频文件
        temp_audio_path = os.path.join(temp_dir, f"temp_audio_{uuid.uuid4().hex[:8]}.wav")
        
        # 确保音频张量是正确的格式
        if audio_tensor.dim() == 3:  # (batch, channels, samples)
            audio_tensor = audio_tensor.squeeze(0)  # 移除batch维度
        elif audio_tensor.dim() == 1:  # (samples,)
            audio_tensor = audio_tensor.unsqueeze(0)  # 添加channel维度
        
        # 确保是2D张量 (channels, samples)
        if audio_tensor.dim() != 2:
            raise RuntimeError(f"音频张量维度不正确: {audio_tensor.shape}")
        
        # 转换为numpy数组并转置为(samples, channels)格式
        audio_np = audio_tensor.cpu().numpy().T
        
        # 使用soundfile保存音频文件（更简单可靠）
        sf.write(temp_audio_path, audio_np, sample_rate, subtype='PCM_16')
        
        return temp_audio_path
    
    def generate_lipsync(
        self,
        audio,
        video_path,
        num_inference_steps,
        guidance_scale,
        seed,
    ):
        """生成唇同步视频"""
        # 清理视频路径（去除可能的引号和空格）
        video_path = video_path.strip().strip('"\'')
        # 提取音频数据
        audio_tensor = audio["waveform"]  # (batch, channels, samples)
        sample_rate = audio["sample_rate"]

        # 在/tmp下创建临时目录
        temp_dir = tempfile.mkdtemp(dir='/tmp')

        try:
            audio_path = self.save_audio_tensor_to_file(audio_tensor, sample_rate, temp_dir)
            print(f"临时音频文件: {audio_path}")
        except Exception as e:
            raise RuntimeError(f"保存音频文件失败: {str(e)}")
        
        print(f"处理视频路径: {video_path}")
        print(f"音频张量形状: {audio_tensor.shape}")
        print(f"音频采样率: {sample_rate}")

        checkpoint_path = os.path.join(LATENTSYNC_PATH,"checkpoints/latentsync_unet.pt")
        config_path = os.path.join(LATENTSYNC_PATH,"configs/unet/stage2.yaml")
        
        # 检查输入文件
        if not os.path.exists(video_path):
            raise RuntimeError(f"视频文件不存在: {video_path}")
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"检查点文件不存在: {checkpoint_path}")
        if not os.path.exists(config_path):
            raise RuntimeError(f"配置文件不存在: {config_path}")
        if not os.path.exists(audio_path):
            raise RuntimeError(f"音频文件不存在: {audio_path}")
            
        # 设置随机种子
        if seed != -1:
            set_seed(seed)
        else:
            torch.seed()
            
        print(f"初始种子: {torch.initial_seed()}")
        print(f"输入视频: {video_path}")
        
        # 生成输出文件路径到ComfyUI的temp目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        video_out_path = os.path.join(temp_dir, f"latentsync_{timestamp}_{unique_id}.mp4")
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
            
            # 清理临时音频文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"已清理临时音频文件: {audio_path}")
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"已清理临时视频文件: {video_path}")
        except Exception as e:
            # 清理临时音频文件
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"已清理临时音频文件: {audio_path}")
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"已清理临时视频文件: {video_path}")
            
            error_msg = f"生成过程中出现错误: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        return (video_out_path,)