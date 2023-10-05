import os
from copy import deepcopy
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import glob
import pandas as pd
import torch
from tqdm import tqdm

from Dino_sampler import DinoSampler

sampler=DinoSampler()

def make_sample_list(path_to_dataset:str,oen):
    out=[]
    for img_set in os.listdir(os.path.join(path_to_dataset,oen)):
        skip_set=False
        s={'path':{},'img_set':img_set}
        if not os.path.isdir(os.path.join(path_to_dataset,oen,img_set)):
            continue
        for i, v in enumerate(['Top', 'Left', 'Right']):
            founded=glob.glob(os.path.join(path_to_dataset, oen, img_set, f'Part_{img_set}_{v}_Color*.jpg'))
            if len(founded) <1:
                skip_set=True
                print(f'DEBUG. did not found view {v} image for {oen} in {img_set}. skip {img_set}')
                break
            else:
                s['path'].update({v:{'RGB':glob.glob(os.path.join(path_to_dataset, oen, img_set, f'Part_{img_set}_{v}_Color*.jpg'))[0]}})
        if not skip_set:
            out.append(s)
    return out


def get_MV_samples_by_oen(full_samples, k_cluster=None,min_samples=20,max_samples=100,plotting=True):
    if isinstance(full_samples,list):
        sample_list=deepcopy(full_samples)
    else:
        sample_list=full_samples['all']
    if k_cluster is None:
        k_cluster=min_samples
    feature_list=[]
    break_cnt=30
    for s in tqdm(sample_list):
        mv_feature_list=None
        for v in list(s['path'].keys()):
            tmp=sampler.get_vectors(Image.open(s['path'][v]['RGB']))[0]
            tmp = tmp.detach().cpu().numpy()
            if mv_feature_list is None:
                mv_feature_list=tmp
            else:
                mv_feature_list=np.concatenate((mv_feature_list,tmp))
        feature_list.append((feature_fusion(mv_feature_list,method='mean'),s['img_set']))
        break_cnt-=1
        if break_cnt == 0:
            break
    pca_array,pca_df=sampler.do_pca(np.array([x[0] for x in feature_list],dtype=object),dim=3)
    #sampler.show_kmeans_WSS(data=pca_array)
    center_points,clusters,dists=sampler.do_clustering(k_cluster=k_cluster,feature_list=pca_array)
    pca_df['cluster'] = pd.Series(clusters, index=pca_df.index)
    pca_df['img_set'] = pd.Series([x[1] for x in feature_list], index=pca_df.index)
    pca_df['dists']= pd.Series([x for x in dists], index=pca_df.index)
    pca_df['center dist'] = pd.Series([dists[i][c] for i, c in zip(pca_df.index.to_numpy(), pca_df['cluster'].to_numpy())], index=pca_df.index)
    out_samples=sample_from_clusters(pca_df,max_samples=max_samples)
    if plotting:
        #df = pd.DataFrame(pca_res, columns=['x', 'y', 'z'])
        #fig = px.scatter_3d(df, x="x", y="y", z='z')
        # Getting unique labels

        u_labels = np.unique(clusters)
        default_colors = px.colors.qualitative.Plotly
        # plotting the results:
        Scene = dict(xaxis=dict(title='X'), yaxis=dict(title='Y'),
                     zaxis=dict(title='Z'))
        data=[]
        for i in u_labels:
            c=default_colors[i]
            data.append(go.Scatter3d(x=pca_df[pca_df.cluster==i]['x'], y=pca_df[pca_df.cluster==i]['y'], z=pca_df[pca_df.cluster==i]['z'], mode='markers',
                             marker=dict(color=c, size=6), name = 'Cluster ' + str(i)))
            data.append(go.Scatter3d(x=[center_points[i, 0]], y=[center_points[i, 1]], z=[center_points[i, 2]], mode='markers',
                                 marker=dict(symbol='x',color=c, size=4, line=dict(color='black', width=20),showlegend=False)))
        layout = go.Layout(margin=dict(l=0, r=0), scene=Scene)
        fig = go.Figure(data=data, layout=layout)
        fig.show()
    return feature_list,pca_array

def sample_from_clusters(pca_df,max_samples,fill_method='next_nearest',oulier_method='half_median'):
    out_samples=[]
    cluster_cnt=len(pca_df['cluster'].unique())
    data=[go.Bar(x=list(pca_df['cluster'].value_counts().index),y=list(pca_df['cluster'].value_counts().values),name='cluster counts')]

    occur_median=np.median(pca_df['cluster'].value_counts())
    data.append(go.Line(x=pca_df['cluster'].unique(),y=[occur_median]*cluster_cnt,name='median'))
    fig = go.Figure(data=data)
    fig.show()
    pca_df_noOutlier=remove_outlier(pca_df,method=oulier_method)
    after=len(pca_df_noOutlier['cluster'].unique())
    print(f'Cluster before removing Outlier Cluster {cluster_cnt} and after {after} | (rest amout of samples {len(pca_df_noOutlier)/len(pca_df)*100:.2f} %)')
    finished=False
    while not finished:
        if len(out_samples)>=max_samples:
            break
        if pca_df_noOutlier.empty:
            break
        for sorted_cluster_idx in list(pca_df_noOutlier['cluster'].value_counts().index):
            next_cluster=pca_df_noOutlier.loc[pca_df_noOutlier['cluster']==sorted_cluster_idx].sort_values(by=['center dist'])
            out_samples.append(next_cluster.iloc[0,:])
            pca_df_noOutlier=pca_df_noOutlier.drop(index=0, axis=0)
    return out_samples

def remove_outlier(pca_df,method):
    out_df=deepcopy(pca_df)
    if method == 'half_median':
        lower_cluster_list=list(out_df['cluster'].value_counts()[out_df['cluster'].value_counts()<=int(np.median(out_df['cluster'].value_counts())/2)].index)
        return out_df[~out_df['cluster'].isin(lower_cluster_list)]
    elif method == 'median':
        lower_cluster_list=list(out_df['cluster'].value_counts()[out_df['cluster'].value_counts()<=int(np.median(out_df['cluster'].value_counts()))].index)
        return out_df[~out_df['cluster'].isin(lower_cluster_list)]
    else:
        print(f' Outlier method {method} not known. Use default cut by median')
        lower_cluster_list=list(out_df['cluster'].value_counts()[out_df['cluster'].value_counts()<=int(np.median(out_df['cluster'].value_counts()))].index)
        return out_df[~out_df['cluster'].isin(lower_cluster_list)]

def rebuild_path(image_path, new_dataset_path):
    filePart = image_path.split('03_DataSet')[1]
    out = new_dataset_path + filePart
    return out


def feature_fusion(input_array,method='mean'):
    if method == 'mean':
        return np.mean(input_array, axis=0)
    else:
        print(f'Method {method} not implemented. Use default Method: mean')
        return np.mean(input_array, axis=0)


if __name__== '__main__':
    path_to_ds=r'S:\DataSets\EIBA\01_Verlesedaten\03_DataSet\Part Number'
    oen='0001123012'
    img_list=make_sample_list(path_to_dataset=path_to_ds,oen=oen)
    f_list,pca_res=get_MV_samples_by_oen(full_samples=img_list,min_samples=20,k_cluster=10)
