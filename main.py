import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from utils.data_preprocessing import binary_encode, fizz_buzz_encode, fizz_buzz
from utils.models import FizzBuzzModelV3
torch.manual_seed(42)

NUM_DIGITS = 12  # upper bound for the training data size
BATCH_SIZE = 128 # training batch size
save_model_folder = 'models'
epoch_num = 10000


# Our goal is to produce fizzbuzz for the numbers 1 to 100. So it would be
# unfair to include these in our training data. Accordingly, the training data
# corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
trX = torch.tensor(np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)]), dtype=torch.float32)
trY = torch.tensor(np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)]), dtype=torch.float32)

valX = torch.tensor(np.array([binary_encode(i, NUM_DIGITS) for i in range(1, 101)]), dtype=torch.float32)
valY = np.array([fizz_buzz_encode(i) for i in range(1, 101)])

# hyperparameter and optimization setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Running on device: ', device)
model = FizzBuzzModelV3(num_digits=NUM_DIGITS)

model.init_weights()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.05)
loss_fun = torch.nn.BCEWithLogitsLoss()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    writer = SummaryWriter()
    start_time = t = time.perf_counter()
    for epoch in range(epoch_num):
        model.train()
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p].clone().detach().to(device), trY[p].clone().detach().to(device)

        train_loss = 0.0
        train_acc = 0.0
        for start in range(0, len(trX), BATCH_SIZE):
            optimizer.zero_grad()
            end = start + BATCH_SIZE
            py = model(trX[start:end])
            loss = loss_fun(py, trY[start:end])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss

            y_label = np.argmax(trY[start:end], axis=1)
            p_label = np.argmax(py.detach().cpu().numpy(), axis=1)
            train_acc = np.mean([y_label[i] == p_label[i] for i in range(len(y_label))])

        writer.add_scalar('Loss/Train', train_loss.detach().cpu().numpy(), epoch)
        writer.add_scalar('Acc/Train', train_acc, epoch)
        # Evaluate every 1000 epochs
        if epoch%1000==0:
            model.eval()
            with torch.no_grad():
                numbers = np.arange(1, 101)
                test_accu = 0.0
                valX = valX.to(device)
                val_py = model(valX)

                y_label = np.argmax(valY, axis=1)
                p_label = np.argmax(val_py.detach().cpu().numpy(), axis=1)
                test_accu = np.mean([y_label[i] == p_label[i] for i in range(len(y_label))])
            np.set_printoptions(precision=4)
            print(epoch, ' Loss: ', train_loss.detach().cpu().numpy(), 'Training acc: ', train_acc, ' Validation accu: ', test_accu)
            writer.add_scalar('Acc/Val', test_accu, epoch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = t = time.perf_counter()
    writer.close()

    output = np.vectorize(fizz_buzz)(numbers, np.argmax(val_py, axis=1)) # convert to fizz-buzz format output
    print('Input: ', numbers)
    print('Ouput: ', output) # uncomment if you want to visualize the actual fizz-buzz output

    print('Total running time: ', end_time - start_time, ' seconds')
    
    if not os.path.exists(save_model_folder):
        os.mkdir(save_model_folder)
    print("Saving model...")
    torch.save(model.state_dict(), os.path.join(save_model_folder, 'fizz_buzz_train' + str(epoch_num) + '.pth'))
