import os

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from TSForecasting.data_provider.data_factory import data_provider
from predict_model import predict
from train_model import train
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

    if save_data:
        print(f"Saving processed train and test files...")
        train.to_csv('data/processed/train_set_processed.csv', index=False)
        test.to_csv('data/processed/test_set_processed.csv', index=False)


args = {

    # Starters
    'root_dir': '.', 'data_path': 'data/',
    'train_data_path': 'data/processed/train_set_processed.csv', 'train_flag': 'train',
    'test_data_path': 'data/processed/test_set_processed.csv', 'test_flag': 'test',
    'data': 'custom', 'features': 'M', 'target': 'orders',
    'batch_size': 32, 'freq': 'd', 'seq_len': 8, 'label_len': 0, 'pred_len': 1,
    'embed': 'timeF',

    # Training params
    'checkpoints': './checkpoints/', 'patience': 3, 'use_amp': False, 'train_epochs': 2, 'learning_rate': 0.0001,

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
    'enc_in': 14, 'e_layers': 6, 'd_model': 64, 'd_ff': 256, 'dropout': 0.05, 'head_dropout': 0.0,
    'patch_ks': 3, 'patch_sd': 2, 'padding_patch': 'end', 'revin': 1, 'affine': 0,
    'subtract_last': 0, 'dw_ks': [3, 4, 5, 6, 7, 8], 're_param': 1, 're_param_kernel': 3,
    'enable_res_param': 1, 'norm': 'batch', 'act': "gelu", 'head_type': 'flatten'
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

    test_dataset, test_loader = data_provider(root_path=args['root_dir'], data_path=args['test_data_path'],
                                              flag=args['test_flag'], features=args['features'],
                                              target=args['target'], data=args['data'],
                                              batch_size=args['batch_size'], freq=args['freq'],
                                              seq_len=args['seq_len'], label_len=args['label_len'],
                                              pred_len=args['pred_len'], embed=args['embed']
                                              )

    model = Model(enc_in=args['enc_in'], seq_len=args['seq_len'], pred_len=args['pred_len'],
                  e_layers=args['e_layers'], d_model=args['d_model'], d_ff=args['d_ff'],
                  dropout=args['dropout'], head_dropout=args['head_dropout'],
                  patch_ks=args['patch_ks'], patch_sd=args['patch_sd'],
                  padding_patch=args['padding_patch'], revin=args['revin'],
                  affine=args['affine'], subtract_last=args['subtract_last'],
                  dw_ks=args['dw_ks'], re_param=args['re_param'],
                  re_param_kernel=args['re_param_kernel'],
                  enable_res_param=args['enable_res_param'])

    model = train(args, model, train_dataset, train_loader)
    save_model(model, model_name=f"./checkpoints/orders_forecasting.pkl")

    predict(args, model, test_dataset, test_loader)
