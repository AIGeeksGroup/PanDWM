import socket
import torch
x=torch.ones(1, device="cuda")
print("CUDA_OK", socket.gethostname(), torch.cuda.get_device_name(0), float(x.sum()), flush=True)
