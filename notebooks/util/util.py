#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
import pandas as pd
import tensorflow_probability as tfp
from sklearn import metrics
import tensorflow as tf
import os
import numpy as np

def plot_series(data, labels=None,
                    windows=None,
                    predictions=None,
                    highlights=None,
                    val_start=None,
                    test_start=None,
                    figsize=None,
                    show_sampling_points=False,
                    show_markers=False,
                    filled_version=None,
                    std=None,
                    ci=None,
                    title=None,
                    ylim=None):
    # Open a new figure
    plt.figure(figsize=figsize)
    # Plot data
    if not show_markers:
        plt.plot(data.index, data.values, zorder=0)
    else:
        plt.plot(data.index, data.values, zorder=0,
                marker='.', markersize=3)
    if filled_version is not None:
        filled = filled_version.copy()
        filled[~data['value'].isnull()] = np.nan
        plt.scatter(filled.index, filled,
                marker='.', c='tab:orange', s=5);
    if show_sampling_points:
        vmin = data.min()
        lvl = np.full(len(data.index), vmin)
        plt.scatter(data.index, lvl, marker='.',
                c='tab:red', s=5)
    # Plot standard deviations
    if std is not None:
        lb = data.values.ravel() - std.values.ravel()
        ub = data.values.ravel() + std.values.ravel()
        plt.fill_between(data.index, lb, ub, alpha=0.3, label='+/- std')
    # Plot confidence intervals
    if ci is not None:
        lb = ci[0].values.ravel()
        ub = ci[1].values.ravel()
        plt.fill_between(data.index, lb, ub, alpha=0.3, label='C.I.')
    # Rotated x ticks
    plt.xticks(rotation=45)
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels],
                    color=anomaly_color, zorder=2, s=5)
    # Plot windows
    if windows is not None:
        for _, wdw in windows.iterrows():
            plt.axvspan(wdw['begin'], wdw['end'],
                        color=anomaly_color, alpha=0.3, zorder=1)
    # Plot training data
    if val_start is not None:
        plt.axvspan(data.index[0], val_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is None and test_start is not None:
        plt.axvspan(data.index[0], test_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is not None:
        plt.axvspan(val_start, test_start,
                    color=validation_color, alpha=0.1, zorder=-1)
    if test_start is not None:
        plt.axvspan(test_start, data.index[-1],
                    color=test_color, alpha=0.3, zorder=0)
    # Predictions
    if predictions is not None:
        plt.scatter(predictions.values, data.loc[predictions],
                    color=prediction_color, alpha=.4, zorder=3,
                    s=5)
    # Force y limits
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(linestyle=':')
    plt.title(title)
    plt.tight_layout()


def plot_dataframe(data, labels=None, vmin=-1.96, vmax=1.96,
        figsize=None, s=4, xlabel=None, ylabel=None):
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(np.mod(labels, 10)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def build_nn_model(input_shape, output_shape, hidden, output_activation='linear'):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
    model_out = layers.Dense(output_shape, activation=output_activation)(x)
    model = keras.Model(model_in, model_out)
    return model


def train_nn_model(model, X, y, loss,
        verbose=0, patience=10,
        validation_split=0.0, **fit_params):
    # Compile the model
    model.compile(optimizer='Adam', loss=loss)
    # Build the early stop callback
    cb = []
    if validation_split > 0:
        cb += [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    # Train the model
    history = model.fit(X, y, callbacks=cb,
            validation_split=validation_split,
            verbose=verbose, **fit_params)
    return history


def plot_nn_model(model):
    return keras.utils.plot_model(model, show_shapes=True,
            show_layer_names=True, rankdir='LR', show_layer_activations=True)


def plot_training_history(history=None,
        figsize=None, print_final_scores=True):
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'], label='val. loss')
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.xlabel('epochs')
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.show()
    if print_final_scores:
        trl = history.history["loss"][-1]
        s = f'Final loss: {trl:.4f} (training)'
        if 'val_loss' in history.history:
            vll = history.history["val_loss"][-1]
            s += f', {vll:.4f} (validation)'
        print(s)



def load_ed_data(data_file):
    # Read the CSV file
    data = pd.read_csv(data_file, sep=';', parse_dates=[2, 3])
    # Remove the "Flow" column
    f_cols = [c for c in data.columns if c != 'Flow']
    data = data[f_cols]
    # Convert a few fields to categorical format
    data['Code'] = data['Code'].astype('category')
    data['Outcome'] = data['Outcome'].astype('category')
    # Sort by triage time
    data.sort_values(by='Triage', inplace=True)
    # Discard the firl
    return data


def plot_bars(data, figsize=None, tick_gap=1, series=None):
    plt.figure(figsize=figsize)
    # x = np.arange(len(data))
    # x = 0.5 + np.arange(len(data))
    # plt.bar(x, data, width=0.7)
    # x = data.index-0.5
    x = data.index
    plt.bar(x, data, width=0.7)
    # plt.bar(x, data, width=0.7)
    if series is not None:
        # plt.plot(series.index-0.5, series, color='tab:orange')
        plt.plot(series.index, series, color='tab:orange')
    if tick_gap > 0:
        plt.xticks(x[::tick_gap], data.index[::tick_gap], rotation=45)
    plt.grid(linestyle=':')
    plt.tight_layout()



def build_nn_poisson_model(input_shape, hidden, rate_guess=1):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
    log_rate = layers.Dense(1, activation='linear')(x)
    lf = lambda t: tfp.distributions.Poisson(rate=rate_guess * tf.math.exp(t))
    model_out = tfp.layers.DistributionLambda(lf)(log_rate)
    model = keras.Model(model_in, model_out)
    return model


def build_nn_normal_model(input_shape, hidden, stddev_guess=1):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
    mu_logsigma = layers.Dense(2, activation='linear')(x)
    lf = lambda t: tfp.distributions.Normal(loc=t[:, :1], scale=tf.math.exp(t[:, 1:]))
    model_out = tfp.layers.DistributionLambda(lf)(mu_logsigma)
    model = keras.Model(model_in, model_out)
    return model


def plot_pred_scatter(y_true, y_pred, figsize=None, print_metrics=True):
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, y_true, marker='.', alpha=0.1)
    xl, xu = plt.xlim()
    yl, yu = plt.ylim()
    l, u = min(xl, yl), max(xu, yu)
    plt.plot([l, u], [l, u], ':', c='0.3')
    plt.grid(linestyle=':')
    plt.xlim(l, u)
    plt.ylim(l, u)
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.tight_layout()

    if print_metrics:
        print(f'R2: {metrics.r2_score(y_true, y_pred):.2f}')
        print(f'MAE: {metrics.mean_absolute_error(y_true, y_pred):.2f}')


def load_cmapss_data(data_folder):
    # Read the CSV files
    fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
    cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
    datalist = []
    nmcn = 0
    for fstem in fnames:
        # Read data
        dfname = os.path.join(f'{data_folder}', f'{fstem}.txt')
        data = pd.read_csv(dfname, sep=' ', header=None)
        # Drop the last two columns (parsing errors)
        data.drop(columns=[26, 27], inplace=True)
        # Replace column names
        data.columns = cols
        # Add the data source
        data['src'] = fstem
        # Shift the machine numbers
        data['machine'] += nmcn
        nmcn += len(data['machine'].unique())
        # Generate RUL data
        cnts = data.groupby('machine')[['cycle']].count()
        cnts.columns = ['ftime']
        data = data.join(cnts, on='machine')
        data['rul'] = data['ftime'] - data['cycle']
        data.drop(columns=['ftime'], inplace=True)
        # Store in the list
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    # Put the 'src' field at the beginning and keep 'rul' at the end
    data = data[['src'] + cols + ['rul']]
    # data.columns = cols
    return data


def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res


def partition_by_machine(data, tr_machines):
    # Separate
    tr_machines = set(tr_machines)
    tr_list, ts_list = [], []
    for mcn, gdata in data.groupby('machine'):
        if mcn in tr_machines:
            tr_list.append(gdata)
        else:
            ts_list.append(gdata)
    # Collate again
    tr_data = pd.concat(tr_list)
    ts_data = pd.concat(ts_list)
    return tr_data, ts_data


def random_censoring(data, rel_censoring_lb, seed=None):
    # seed the RNG
    np.random.seed(seed)
    # Process all experiments
    tr_list = []
    for _, gdata in data.groupby('machine'):
        # sample a censoring point
        sep = int(len(gdata) * np.random.rand())
        tr_list.append(gdata.iloc[:sep])
    # Collate again
    res = pd.concat(tr_list)
    return res


class RULCostModel:
    def __init__(self, maintenance_cost, safe_interval=0):
        self.maintenance_cost = maintenance_cost
        self.safe_interval = safe_interval

    def cost(self, machine, pred, thr, return_margin=False):
        # Merge machine and prediction data
        tmp = np.array([machine, pred]).T
        tmp = pd.DataFrame(data=tmp,
                           columns=['machine', 'pred'])
        # Cost computation
        cost = 0
        nfails = 0
        slack = 0
        for mcn, gtmp in tmp.groupby('machine'):
            idx = np.nonzero(gtmp['pred'].values < thr)[0]
            if len(idx) == 0:
                cost += self.maintenance_cost
                nfails += 1
            else:
                cost -= max(0, idx[0] - self.safe_interval)
                slack += len(gtmp) - idx[0]
        if not return_margin:
            return cost
        else:
            return cost, nfails, slack


def plot_rul(pred=None, target=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=None,
        title=None,
        xlabel=None,
        ylabel=None):
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target',
                color='tab:orange')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
        if stddev is not None:
            ax.fill_between(range(len(pred)),
                    pred-stddev, pred+stddev,
                    alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(linestyle=':')
    plt.tight_layout()


def predict_cf(model, ref_sample, columns, values):
    # Replicate the reference sample
    df = np.repeat(ref_sample.values.reshape(1, -1), len(values), axis=0)
    # Replace the columns
    df = pd.DataFrame(columns=ref_sample.index, data=df)
    df[columns] = values
    # Compute predictions
    pred = model.predict(df.astype('float64'), verbose=0)
    # Add index and return
    return pd.Series(data=pred.ravel())


def rolling_survival_cmapss(hazard_model, data, look_up_window):
    # Repeat the rows in the dataframe
    rdata = data.loc[data.index.repeat(len(look_up_window))]
    # Add cycle offsets
    rdata['cycle'] += np.tile(look_up_window, len(data))
    # Compute hazard values
    hazard = hazard_model.predict(rdata, verbose=0)
    # Reshape hazard values (one row per original sample)
    hazard = hazard.reshape(len(data), len(look_up_window))
    # Compute survival
    survival = np.cumprod(1-hazard, axis=1)
    # Prepare a dataframe and return
    return pd.DataFrame(index=data.index, data=survival, columns=look_up_window)


