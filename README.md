# ClipCapApp

ClipCapApp is a FastAPI application for image captioning and story generation using ClipCap.

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
