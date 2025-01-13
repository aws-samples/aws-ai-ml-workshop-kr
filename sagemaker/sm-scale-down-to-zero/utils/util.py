import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def resample_clickstream(clickstream_cnt, freq='10T'):
    resampling_clickstream = clickstream_cnt.set_index('timestamp').resample(freq)
    clickstream_resample = resampling_clickstream.nunique()[['page','user']]
    clickstream_resample['clicks'] = resampling_clickstream.count()['session_id']
    clickstream_resample.columns = ['page','user','click']
    return clickstream_resample

def plot_clicks(df, start_dt=None, end_dt=None):
    start_dt = df.index[0] if start_dt is None else start_dt
    end_dt   = df.index[-1] if end_dt is None else end_dt
    plt.figure(figsize=(20,3))
    plt.plot(df[['click','user']][start_dt:end_dt])
    plt.legend(['click','user'])
    plt.title('clicks and users by time')
    plt.show()
    
def plot_click_w_fault(df, start_dt=None, end_dt=None):
    start_dt = df.index[0] if start_dt is None else start_dt
    end_dt   = df.index[-1] if end_dt is None else end_dt
    fig = plt.figure(constrained_layout=True, figsize=(20,5))
    spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[4,1])
    ax1 = fig.add_subplot(spec[0,0])
    ax1.plot(df[['click','user']][start_dt:end_dt])
    ax1.legend(['click','user'])
    ax1.set_title('clicks and users by time')
    ax2 = fig.add_subplot(spec[1,0])
    ax2.plot(df['fault'][start_dt:end_dt])
    ax2.legend(['fault'])
    plt.show()

def plot_click_w_fault_and_res(df, start_dt=None, end_dt=None):
    start_dt = df.index[0] if start_dt is None else start_dt
    end_dt   = df.index[-1] if end_dt is None else end_dt
    fig = plt.figure(constrained_layout=True, figsize=(20,6))
    spec = fig.add_gridspec(ncols=1, nrows=3, height_ratios=[4,2,1])
    ax1 = fig.add_subplot(spec[0,0])
    ax1.plot(df[['click','user']][start_dt:end_dt])
    ax1.legend(['click','user'])
    ax1.set_title('clicks and users by time')
    ax2 = fig.add_subplot(spec[1,0])
    ax2.plot(df['residual'][start_dt:end_dt])
    ax2.legend(['residual'])
    ax3 = fig.add_subplot(spec[2,0])
    ax3.plot(df['fault'][start_dt:end_dt])
    ax3.legend(['fault'])
    plt.show()

def plot_click_w_fault_res_ad(df, anomalous, threshold, start_dt=None, end_dt=None):
    start_dt = df.index[0] if start_dt is None else start_dt
    end_dt   = df.index[-1] if end_dt is None else end_dt
    anomalous = anomalous[start_dt:end_dt]
    
    fig = plt.figure(constrained_layout=True, figsize=(20,6))
    spec = fig.add_gridspec(ncols=1, nrows=3, height_ratios=[4,2,1])
    
    ax1 = fig.add_subplot(spec[0,0])
    ax1.plot(df[['click','user']][start_dt:end_dt])
    ax1.legend(['click','user'])
    ax1.set_title('clicks and users by time')
    ax1.scatter(x= anomalous.index, y=anomalous['user'].values, c='blue')
    ax1.scatter(x= anomalous.index, y=anomalous['click'].values, c='red')

    ax2 = fig.add_subplot(spec[1,0])
    ax2.plot(df['residual'][start_dt:end_dt])
    ax2.legend(['residual'])
    ax2.scatter(x= anomalous.index, y=anomalous['residual'].values, c='red')
    
    ax3 = fig.add_subplot(spec[2,0])
    ax3.plot(df['fault'][start_dt:end_dt])
    ax3.legend(['fault'])
    plt.show()

    
def plot_click_w_ad_exp(df, anomalous, threshold, start_dt=None, end_dt=None, score="ANOMALY_SCORE"):
    anomalous = anomalous[start_dt:end_dt]
    fig = plt.figure(constrained_layout=True, figsize=(20,8))
    spec = fig.add_gridspec(ncols=1, nrows=3, height_ratios=[4,3,4])
    
    ax1 = fig.add_subplot(spec[0,0])
    ax1.plot(df[['click','user']][start_dt:end_dt])
    ax1.legend(['click','user'])
    ax1.set_title('clicks and users by time')
    ax1.scatter(x= anomalous.index, y=anomalous['user'].values, c='blue')
    ax1.scatter(x= anomalous.index, y=anomalous['click'].values, c='red')

    # Anormaly score graph
    ax2 = fig.add_subplot(spec[1,0])
    ax2.plot(df[score][start_dt:end_dt])
    ax2.scatter(x= anomalous.index, y=anomalous[score].values, c='red')
    ax2.axhline(y=threshold, linestyle='--', c='b')
    ax2.legend([score,'threshold'])
    
    # Attribution score graph
    ax3 = fig.add_subplot(spec[2,0])
    ax3.plot(df[['URLS_ATTRIBUTION_SCORE','USERS_ATTRIBUTION_SCORE','CLICKS_ATTRIBUTION_SCORE', 'RESIDUALS_ATTRIBUTION_SCORE']][start_dt:end_dt])
    ax3.legend(['URLS_ATTRIBUTION_SCORE','USERS_ATTRIBUTION_SCORE','CLICKS_ATTRIBUTION_SCORE','RESIDUALS_ATTRIBUTION_SCORE'] )
#     ax3.scatter(x= anomalous.index, y=anomalous['URLS_ATTRIBUTION_SCORE'].values, c='blue')
#     ax3.scatter(x= anomalous.index, y=anomalous['USERS_ATTRIBUTION_SCORE'].values, c='orange')
#     ax3.scatter(x= anomalous.index, y=anomalous['CLICKS_ATTRIBUTION_SCORE'].values, c='green')
#     ax3.scatter(x= anomalous.index, y=anomalous['RESIDUALS_ATTRIBUTION_SCORE'].values, c='red')

    plt.show()

def resample_click_by_page(clickstream_cnt, page, freq='10T'):
    resampling_clickstream = clickstream_cnt.set_index('timestamp').resample(freq)
    clickstream_resample = {'user':resampling_clickstream.nunique()['user']}
    clickstream_resample.update({'click':resampling_clickstream.count()['session_id']})
    clickstream_resample.update({'page':page})
    
    return clickstream_resample

def plot_click_by_page(df):
    plt.figure(figsize=(20,12))
    freq='1H'

    pages = df['page'].unique()
    clickstream_cnt = df.groupby(['timestamp', 'page']).count().reset_index()
    click_by_page = []
    for page in pages:
        filtered_click = clickstream_cnt[clickstream_cnt['page']==page]
        click_by_page.append(resample_click_by_page(filtered_click, page, freq=freq))

    xticks = np.arange(min(click_by_page[0]['click'].index), max(click_by_page[0]['click'].index), int(5e11))

    for index, cbp in enumerate(click_by_page):
        plt.subplot(4,5,index+1)
        plt.plot(cbp['click'])
        plt.title(cbp['page'])
        plt.xticks(xticks)