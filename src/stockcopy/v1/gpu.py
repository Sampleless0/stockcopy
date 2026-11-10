import torch 

print('Version:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('Available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')