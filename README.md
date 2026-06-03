# ClipCapApp

## ClipCap Story API

A lightweight image captioning and story generation API built on [ClipCap](https://github.com/rmokady/CLIP_prefix_caption). Given an image, it generates a short caption using CLIP + GPT-2, then optionally continues that caption into a short creative story.

### Features

- **Image Captioning** (`/caption`): Generate a concise description of the uploaded image.
- **Story Generation** (`/story`): First generate a caption, then let GPT-2 continue it into a short narrative (150-300 tokens).
- **REST API** built with FastAPI, easy to integrate into web or mobile apps.
- **Local inference** – no external API calls; runs entirely on your own hardware (GPU or CPU).

### Requirements

- Python 3.10
- CUDA 12.8+ (optional, but recommended for speed)
- 8GB+ RAM, 6GB+ GPU memory (for GPU inference)

### Quick Start

1. Clone the repository and set up a conda environment:
   ```bash
   git clone https://github.com/yourusername/clipcap-story-api.git
   cd clipcap-story-api
   conda create -n clipcap_env python=3.10
   conda activate clipcap_env
   pip install -r requirements.txt

## 主要功能

- `/caption`：生成图像描述
- `/story`：生成图像故事

## 环境要求

- Python 3.10
- PyTorch 2.9+
- CUDA 12.8（可选，用于 GPU 加速）

## 安装步骤

1. 创建 conda 环境
   ```bash
   conda create -n clipcapapp python=3.10 -y
   conda activate clipcapapp
   ```
2. 安装依赖
   ```bash
   pip install -r requirements_backup.txt
   ```
3. 下载模型权重
   - 将所需的 ClipCap 模型权重文件和 GPT-2 本地权重放置到项目目录中。
   - ClipCap模型权重文件:https://drive.google.com/file/d/1GYPToCqFREwi285wPLhuVExlz7DDUDfJ/view, 下载后放置在ClipCap文件夹下
   - GPT-2本地权重命名为gpt2_local放置在根目录下

## 启动服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API 使用示例

生成图像描述：

```bash
curl -X POST "http://127.0.0.1:8000/caption" -F "image=@path/to/image.jpg"
```

生成图像故事：

```bash
curl -X POST "http://127.0.0.1:8000/story" -F "image=@path/to/image.jpg"
```

## 已知限制

- Caption 生成通常较短
- 故事质量可能不稳定

## 许可证

MIT
