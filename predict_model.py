import os

import numpy as np
import torch


def predict(args, model, pred_data, pred_loader, load=False, device='cuda'):

    preds = []
    origin_inputs = []
    ground_truth = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, _, _) in enumerate(pred_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            print(batch_x.shape, batch_y.shape)

            if args['use_amp']:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x)
            else:
                outputs = model(batch_x)

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            preds.append(pred)
            origin_input = batch_x.squeeze().detach().cpu().numpy()
            origin_inputs.append(origin_input)
            truth = batch_y.squeeze().detach().cpu().numpy()
            ground_truth.append(truth)

    origin_inputs = np.array(origin_inputs)
    print(origin_inputs.shape)
    origin_inputs = origin_inputs.reshape(-1, origin_inputs.shape[-2], origin_inputs.shape[-1])
    print(origin_inputs.shape)
    ground_truth = np.array(ground_truth)
    print(ground_truth.shape)
    preds = np.array(preds)
    ground_truth = ground_truth.reshape(-1, ground_truth.shape[-2], ground_truth.shape[-1])
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    print(preds.shape)

    # result save
    folder_path = './prediction'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + 'real_prediction.npy', preds)
    np.save(folder_path + 'origin_series.npy', origin_inputs)
    np.save(folder_path + 'ground_truth.npy', ground_truth)

    return
