from TSForecasting.data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(root_path, data_path, flag, features, target, data, batch_size, freq,
                  seq_len, label_len, pred_len, embed):

    print(f"Mode: {flag}; datapath: {data_path}, flag: {flag}; features: {features}, target: {target}, "
          f"data: {data}, batch_size: {batch_size}, freq: {freq}, seq_len: {seq_len}, label_len: {label_len}, "
          f"pred_len: {pred_len}, embed: {embed}")

    Data = data_dict[data]
    timeenc = 0 if embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = batch_size
        freq = freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = freq
        Data = Dataset_Pred
    else:
        shuffle_flag = False  # True
        drop_last = True
        batch_size = batch_size
        freq = freq

    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq
    )

    print(flag, len(data_set))
    if len(data_set) < batch_size: 
        batch_size = len(data_set)

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag,
                             num_workers=1, drop_last=drop_last)

    return data_set, data_loader
