import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from utils.data_preprocessing import binary_encode, fizz_buzz_encode, fizz_buzz
from utils.models import FizzBuzzModelV3

NUM_DIGITS = 12  # upper bound for the training data size
save_model_folder = 'models/fizz_buzz_train10000.pth'

valX = torch.tensor(np.array([binary_encode(i, NUM_DIGITS) for i in range(1, 101)]), dtype=torch.float32)
valY = np.array([fizz_buzz_encode(i) for i in range(1, 101)])

# hyperparameter and optimization setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Running on device: ', device)
model = FizzBuzzModelV3(num_digits=NUM_DIGITS)
model.load_state_dict(torch.load(save_model_folder))
numbers = np.arange(1, 101)

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
        test_accu = 0.0
        valX = valX.to(device)
        val_py = model(valX)

        y_label = np.argmax(valY, axis=1)
        p_label = np.argmax(val_py.detach().cpu().numpy(), axis=1)
        test_accu = np.mean([y_label[i] == p_label[i] for i in range(len(y_label))])
        np.set_printoptions(precision=4)
        print('Validation accu: ', test_accu)
        
        output = np.vectorize(fizz_buzz)(numbers, np.argmax(val_py, axis=1)) # convert to fizz-buzz format output
        print('Input: ', numbers)
        print('Ouput: ', output) # uncomment if you want to visualize the actual fizz-buzz output
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
