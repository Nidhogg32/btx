from modelscope import snapshot_download
custom_download_path = '/home/roo/dream/zqw/qwen/qwen2-7B'
model_dir = snapshot_download('qwen/Qwen2-7B', cache_dir=custom_download_path)