import streamlit as st
import boto3
import datetime
import numpy as np        
from scipy import stats
import matplotlib.pyplot as plt
import time

dynamodb = boto3.client('dynamodb', region_name='ap-northeast-2')
ap_invocation = ap_reward = sm_invocation = sm_reward = 0

def update_beta_distribution(batch_count):
    table_name = 'Model'

    response = dynamodb.scan(TableName=table_name)
    items = response.get('Items', [])
    
    for item in items:
        if item.get('model_name').get('S') == 'model_AP':
            ap_invocation = int(item.get('invocation_count').get('N'))
            ap_reward = int(item.get('reward_click').get('N'))
        else:
            sm_invocation = int(item.get('invocation_count').get('N'))
            sm_reward = int(item.get('reward_click').get('N'))

    alpha_ap = ap_reward
    beta_ap = ap_invocation - ap_reward
    
    alpha_sm = sm_reward
    beta_sm = sm_invocation - sm_reward
      
    fig, axs = plt.subplots()
    x = np.arange (0, 0.1, 0.001)
    axs.set(title='batch #' + str(batch_count), xlabel='click_rate', ylabel='relative probability')

    axs.plot(x, stats.beta.pdf(x, alpha_ap, beta_ap), label='Personalize Model')
    axs.plot(x, stats.beta.pdf(x, alpha_sm, beta_sm), label='SageMaker Model')
    axs.legend()
    st.pyplot(fig)


batch_count = 0
def update_data(batch_count):
    while batch_count < 11:
        update_beta_distribution(batch_count)
        time.sleep(60)
        batch_count += 1

update_beta_distribution(batch_count)
# update_data(batch_count)
