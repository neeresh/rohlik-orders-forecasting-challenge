import os
import time

import numpy as np
import torch
from torch import optim, nn

from TSForecasting.utils.tools import EarlyStopping, adjust_learning_rate


def _select_optimizer(args, model):
    model_optim = optim.Adam(model.parameters(), lr=args['learning_rate'])
    return model_optim


def _select_criterion():
    criterion = nn.MSELoss()
    return criterion


def train(args, model, train_data, train_loader, device='cuda'):

    # path = os.path.join(args['checkpoints'], setting)
    # if not os.path.exists(path):
    #     os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args['patience'], verbose=True)

    model_optim = _select_optimizer(args, model)
    criterion = _select_criterion()

    if args['use_amp']:
        scaler = torch.cuda.amp.GradScaler()

    model.to(device)
    for epoch in range(args['train_epochs']):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # encoder - decoder
            if args['use_amp']:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x)

                    f_dim = -1 if args['features'] == 'MS' else 0
                    outputs = outputs[:, -args['pred_len']:, f_dim:]
                    batch_y = batch_y[:, -args['pred_len']:, f_dim:].to(device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                outputs = model(batch_x)

                # print(outputs.shape,batch_y.shape)
                f_dim = -1 if args['features'] == 'MS' else 0
                outputs = outputs[:, -args['pred_len']:, f_dim:]
                batch_y = batch_y[:, -args['pred_len']:, f_dim:].to(device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args['train_epochs'] - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args['use_amp']:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        # vali_loss = vali(vali_data, vali_loader, criterion)
        # test_loss = vali(test_data, test_loader, criterion)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
            epoch + 1, train_steps, train_loss))

        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        # early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch + 1, args)

    # best_model_path = path + '/' + 'checkpoint.pth'
    # model.load_state_dict(torch.load('./checkpoints'))

    return model
