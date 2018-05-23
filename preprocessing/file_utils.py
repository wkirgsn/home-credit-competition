import os
import sqlite3
import numpy as np
import preprocessing.config as cfg


def save_predictions(id, pred):
    if cfg.data_cfg['save_predictions']:
        with sqlite3.connect(cfg.data_cfg['db_path']) as con:
            # create table if not exists
            query = """CREATE TABLE IF NOT EXISTS 
                    predictions(id text, idx int, {} real, {} real, {} real, 
                    {} real)""".format(*pred.columns)
            con.execute(query)

            # format prediction
            df_to_db = pred.copy()
            df_to_db['id'] = id
            df_to_db['idx'] = pred.index
            entries = [tuple(x) for x in np.roll(df_to_db.values,
                                                 shift=2, axis=1)]
            con.executemany('INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)',
                            entries)
            print('Predictions of model with uuid {} saved to db.'.format(id))


def truncate_actual_target(actual, prediction):
    # trunc actual if prediction is shorter
    if prediction.shape[0] != actual.shape[0]:
        print('trunc actual from {} to {} samples'.format(actual.shape[0],
                                                          prediction.shape[0]))
    offset = actual.shape[0] - prediction.shape[0]
    return actual.iloc[offset:, :]


def get_available_gpus():

    if cfg.keras_cfg['use_cpu']:
        print("Update environment variables to use CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']

    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']