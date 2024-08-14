import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from TSForecasting.data_provider.data_factory import data_provider
from train_model import train, test, predict
from TSForecasting.models.ConvTimeNet import Model


def preprocess_date(data):
    data = data.copy()

    data['date'] = pd.to_datetime(data['date'])
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    data['day_of_week'] = data['date'].dt.dayofweek
    data['week_of_year'] = data['date'].dt.isocalendar().week
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['is_holiday'] = data['holiday'].astype(int)
    data['quarter'] = data['date'].dt.quarter

    return data


def preprocess_data(train, test):
    columns_to_have = list(test.columns) + ['orders']
    train = train[columns_to_have]

    # Preprocess date columns
    train = preprocess_date(train)
    test = preprocess_date(test)

    # Remove unwanted columns
    train = train.drop(['holiday_name', 'id'], axis=1)
    test = test.drop(['holiday_name', 'id'], axis=1)

    # Label Encoding
    le = LabelEncoder()
    train['warehouse'] = le.fit_transform(train['warehouse'])
    test['warehouse'] = le.transform(test['warehouse'])

    # making target (orders) column as the last column for df
    if 'orders' in train.columns:
        cols = [col for col in train.columns if col != 'orders']
        cols.append('orders')
        train = train[cols]
    else:
        print("Target column 'target' not found in DataFrame.")

    # Making date column as the first column of df
    if 'date' in train.columns:
        cols = [col for col in train.columns if col != 'date']
        cols = ['date'] + cols
        train = train[cols]
    else:
        print("Column 'date' not found in DataFrame.")

    if 'date' in test.columns:
        cols = [col for col in test.columns if col != 'date']
        cols = ['date'] + cols
        test = test[cols]
    else:
        print("Column 'date' not found in DataFrame.")

    return train, test


def save_model(model, model_name):
    print(f"Saving the model with model_name: {model_name}")

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), model_name)

    print(f"Saving successfull!!!")


def load_model(model, model_name, device='cuda'):
    model.load_state_dict(torch.load(model_name))

    print(f"Model is set to eval() mode...")
    model.eval()

    print(f"Model is on the deivce: {device}")
    model.to(device)

    return model


def save_train_test_datasets(root_dir, data_path, preprocessing=False, save_data=False):
    train = pd.read_csv(os.path.join(root_dir, data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(root_dir, data_path, 'test.csv'))

    if preprocessing:
        train, test = preprocess_data(train, test)
        print(f"Length of train set: {train.shape}")
        print(f"Length of test set: {test.shape}")

        print(f"Train set columns: {train.columns}")
        print(f"Test set columns: {test.columns}")

    # Creating a column called "orders" for test set. This won't be used.
    test['orders'] = 5000

    if save_data:
        print(f"Saving processed train and test files...")
        train.to_csv('data/processed/train_set_processed.csv', index=False)
        test.to_csv('data/processed/test_set_processed.csv', index=False)


args = {

    # Starters
    'root_dir': '.', 'data_path': 'data/',

    # For train, val and test sets, training data is split into 3 parts
    'train_data_path': 'data/processed/train_set_processed.csv', 'train_flag': 'train',
    'val_data_path': 'data/processed/train_set_processed.csv', 'val_flag': 'val',
    'test_data_path': 'data/processed/train_set_processed.csv', 'test_flag': 'test',

    # To score unseen data, we use new_data_flag = 'pred'
    'unseen_data_path': 'data/processed/test_set_processed.csv', 'unseen_data_flag': 'pred',
    'data': 'custom', 'features': 'MS', 'target': 'orders',
    'batch_size': 16, 'freq': 'd', 'seq_len': 14, 'label_len': 14, 'pred_len': 20,
    'embed': 'timeF',

    # Training params
    'checkpoints': './checkpoints/', 'patience': 5, 'use_amp': True, 'train_epochs': 2, 'learning_rate': 0.00001,

    # Model args
    # enc_in: Number of input features to the model
    # patch_ks: Kernel size for the patch-based convolutional layers.
    # patch_sd: Stride of the convolutional kernel for patches.
    # revin: Indicates whether to use reversible networks.
    # affine:  Whether to use affine transformations in normalization layers.
    # dw_ks: List of kernel sizes for depth-wise convolutions.
    # re_param:  Indicates whether to use a specific type of parameterization.
    # re_param_kernel:  Kernel size for the reparameterization.
    # enable_res_param: Whether to enable residual parameterization.
    # head_type: Type of output head used in the model.
    'enc_in': 14, 'e_layers': 6, 'd_model': 128, 'd_ff': 256, 'dropout': 0.05, 'head_dropout': 0.0,
    'patch_ks': 16, 'patch_sd': 3, 'padding_patch': 'end', 'revin': 1, 'affine': 0,
    'subtract_last': 0, 'dw_ks': [11, 15, 21, 29, 39, 51], 're_param': 1, 're_param_kernel': 3,
    'enable_res_param': 1, 'norm': 'batch', 'act': "gelu", 'head_type': 'flatten',

    # Test
    'test_flop': False, 'do_predict': True,

}

if __name__ == '__main__':

    save_train_test_datasets(root_dir=args['root_dir'], data_path=args['data_path'],
                             preprocessing=True, save_data=True)

    train_dataset, train_loader = data_provider(root_path=args['root_dir'], data_path=args['train_data_path'],
                                                flag=args['train_flag'], features=args['features'],
                                                target=args['target'], data=args['data'],
                                                batch_size=args['batch_size'], freq=args['freq'],
                                                seq_len=args['seq_len'], label_len=args['label_len'],
                                                pred_len=args['pred_len'], embed=args['embed'])

    # val_dataset, val_loader = data_provider(root_path=args['root_dir'], data_path=args['val_data_path'],
    #                                           flag=args['test_flag'], features=args['features'],
    #                                           target=args['target'], data=args['data'],
    #                                           batch_size=args['batch_size'], freq=args['freq'],
    #                                           seq_len=args['seq_len'], label_len=args['label_len'],
    #                                           pred_len=args['pred_len'], embed=args['embed']
    #                                           )
    #
    # test_dataset, test_loader = data_provider(root_path=args['root_dir'], data_path=args['test_data_path'],
    #                                           flag=args['test_flag'], features=args['features'],
    #                                           target=args['target'], data=args['data'],
    #                                           batch_size=args['batch_size'], freq=args['freq'],
    #                                           seq_len=args['seq_len'], label_len=args['label_len'],
    #                                           pred_len=args['pred_len'], embed=args['embed']
    #                                           )

    # unseen_dataset, unseen_loader = data_provider(root_path=args['root_dir'], data_path=args['unseen_data_path'],
    #                                           flag=args['unseen_data_flag'], features=args['features'],
    #                                           target=args['target'], data=args['data'],
    #                                           batch_size=args['batch_size'], freq=args['freq'],
    #                                           seq_len=args['seq_len'], label_len=args['label_len'],
    #                                           pred_len=args['pred_len'], embed=args['embed']
    #                                           )

    model = Model(enc_in=args['enc_in'], seq_len=args['seq_len'], pred_len=args['pred_len'],
                  e_layers=args['e_layers'], d_model=args['d_model'], d_ff=args['d_ff'],
                  dropout=args['dropout'], head_dropout=args['head_dropout'],
                  patch_ks=args['patch_ks'], patch_sd=args['patch_sd'],
                  padding_patch=args['padding_patch'], revin=args['revin'],
                  affine=args['affine'], subtract_last=args['subtract_last'],
                  dw_ks=args['dw_ks'], re_param=args['re_param'],
                  re_param_kernel=args['re_param_kernel'],
                  enable_res_param=args['enable_res_param'])

    # model = train(args, model, train_dataset, train_loader, val_dataset, val_loader,
    #               test_dataset, test_loader)
    model = train(args, model, train_dataset, train_loader)

    # Saving extra timesteps for evaluation
    original_train_set = pd.read_csv('data/processed/train_set_processed.csv')
    original_test_set = pd.read_csv('data/processed/test_set_processed.csv')

    modified_test_set = pd.concat([original_train_set.iloc[-args['seq_len']:, :], original_test_set], ignore_index=True)
    modified_test_set.to_csv('data/processed/test_set_processed_modified.csv', index=False)

    unseen_data, unseen_loader = data_provider(root_path=args['root_dir'],
                                               data_path='data/processed/test_set_processed_modified.csv',
                                               flag=args['unseen_data_flag'], features=args['features'],
                                               target=args['target'], data=args['data'],
                                               batch_size=1, freq=args['freq'],
                                               seq_len=args['seq_len'], label_len=args['label_len'],
                                               pred_len=args['pred_len'], embed=args['embed']
                                               )

    predict(args, model, unseen_data, unseen_loader)

    # # Testing the trained model
    # test(args, model, test_loader, load_saved_model=True)
