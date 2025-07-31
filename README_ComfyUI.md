# LatentSync ComfyUI 节点

这个项目将 LatentSync 唇同步技术封装成了 ComfyUI 节点，让您可以在 ComfyUI 中轻松使用 LatentSync 进行唇同步视频生成。

## 功能特性

- 🎬 **高质量唇同步**: 基于 LatentSync 技术，生成高质量的唇同步视频
- 🎯 **ComfyUI 集成**: 完全集成到 ComfyUI 工作流中
- 🔧 **灵活配置**: 支持多种参数调整，包括推理步数、引导比例、分辨率等
- 📦 **批量处理**: 支持批量处理多个视频文件
- 🎨 **Mask 输出**: 可选择输出 mask 视频用于进一步处理
- 💾 **模型缓存**: 智能缓存模型，避免重复加载

## 安装说明

### 1. 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)
- ComfyUI
- LatentSync 项目及其依赖

### 2. 安装步骤

1. **克隆 LatentSync 项目** (如果还没有):
```bash
git clone <latentsync-repo-url>
cd LatentSync
```

2. **安装依赖**:
```bash
pip install -r requirements.txt
```

3. **下载模型文件**:
   - 下载 LatentSync 检查点文件到 `checkpoints/latentsync_unet.pt`
   - 下载 Whisper 模型到 `checkpoints/whisper/` 目录

4. **安装 ComfyUI 节点**:
   - 将 `comfyui_latentsync_node.py` 或 `comfyui_latentsync_advanced.py` 复制到 ComfyUI 的 `custom_nodes` 目录
   - 重启 ComfyUI

### 3. 目录结构

确保您的项目目录结构如下：

```
LatentSync/
├── checkpoints/
│   ├── latentsync_unet.pt
│   └── whisper/
│       ├── small.pt
│       └── tiny.pt
├── configs/
│   └── unet/
│       └── stage2.yaml
├── latentsync/
│   ├── models/
│   ├── pipelines/
│   └── whisper/
├── comfyui_latentsync_node.py
└── comfyui_latentsync_advanced.py
```

## 使用方法

### 基础节点 (LatentSyncNode)

适用于简单的文件路径输入：

1. 在 ComfyUI 中添加 "LatentSync 唇同步" 节点
2. 配置以下参数：
   - **video_path**: 输入视频文件路径
   - **audio_path**: 输入音频文件路径
   - **checkpoint_path**: 模型检查点路径
   - **config_path**: 配置文件路径
   - **output_path**: 输出视频路径
   - **num_inference_steps**: 推理步数 (默认: 20)
   - **guidance_scale**: 引导比例 (默认: 1.5)
   - **seed**: 随机种子 (默认: 1247)
   - **num_frames**: 帧数 (默认: 16)
   - **resolution**: 分辨率 (默认: 256)

### 高级节点 (LatentSyncAdvancedNode)

支持 ComfyUI 的视频和音频输入：

1. 在 ComfyUI 中添加 "LatentSync 唇同步 (高级)" 节点
2. 连接视频和音频输入
3. 配置参数（与基础节点相同）
4. 节点将输出：
   - **output_video**: 生成的唇同步视频
   - **mask_video**: mask 视频（可选）
   - **info**: 处理信息

### 批量处理节点 (LatentSyncBatchNode)

用于批量处理多个文件：

1. 在 ComfyUI 中添加 "LatentSync 批量唇同步" 节点
2. 连接视频列表和音频列表
3. 配置批量处理参数
4. 节点将输出：
   - **output_videos**: 生成的视频列表
   - **mask_videos**: mask 视频列表
   - **batch_info**: 批量处理信息

## 参数说明

### 必需参数

- **video_path/video**: 输入视频文件或视频数据
- **audio_path/audio**: 输入音频文件或音频数据
- **checkpoint_path**: LatentSync 模型检查点路径
- **config_path**: 模型配置文件路径

### 可选参数

- **num_inference_steps**: 推理步数，影响生成质量和速度
- **guidance_scale**: 引导比例，控制生成结果的创造性
- **seed**: 随机种子，用于结果复现
- **num_frames**: 处理的帧数
- **resolution**: 输出视频分辨率
- **video_fps**: 视频帧率
- **audio_sample_rate**: 音频采样率
- **mask_image_path**: mask 图像路径
- **use_fp16**: 是否使用半精度浮点数
- **save_mask**: 是否保存 mask 视频

## 工作流示例

### 基础工作流

```
视频文件 → LatentSync 唇同步 → 输出视频
音频文件 ↗
```

### 高级工作流

```
视频加载器 → LatentSync 唇同步 (高级) → 视频保存器
音频加载器 ↗                              ↓
                                    Mask 视频保存器
```

### 批量处理工作流

```
视频列表 → LatentSync 批量唇同步 → 视频列表保存器
音频列表 ↗                              ↓
                                    Mask 视频列表保存器
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查检查点文件路径是否正确
   - 确保 Whisper 模型文件存在
   - 验证配置文件路径

2. **CUDA 内存不足**
   - 减少 `num_frames` 参数
   - 降低 `resolution` 参数
   - 启用 `use_fp16` 选项

3. **输入文件格式问题**
   - 确保视频文件格式为 MP4
   - 确保音频文件格式为 WAV
   - 检查文件路径中的特殊字符

4. **依赖库缺失**
   - 安装所有必需的依赖：`pip install -r requirements.txt`
   - 确保 ComfyUI 环境正确配置

### 性能优化

- 使用 GPU 加速（推荐 RTX 3080 或更高）
- 启用 FP16 精度
- 适当调整推理步数
- 使用批量处理提高效率

## 技术细节

### 模型架构

- **UNet**: 3D 条件 UNet 用于视频生成
- **VAE**: 用于潜在空间编码/解码
- **Whisper**: 用于音频特征提取
- **调度器**: DDIM 调度器用于去噪过程

### 处理流程

1. 加载输入视频和音频
2. 提取音频特征
3. 初始化潜在空间
4. 执行去噪过程
5. 解码生成视频
6. 应用 mask 处理
7. 输出最终结果

## 许可证

本项目遵循 LatentSync 的原始许可证。请确保遵守相关使用条款。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个 ComfyUI 节点！

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基础唇同步功能
- 集成 ComfyUI 节点系统
- 添加批量处理支持 