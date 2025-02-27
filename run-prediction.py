import torch
import numpy as np

PATH = 'data/model-650M.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.jit.load(PATH)
model.eval()

logits = model(torch.tensor(np.load('data/7qoqA.npy'), dtype=torch.float32).to(device)).squeeze()
pred = torch.round(torch.sigmoid(logits))
print(pred)