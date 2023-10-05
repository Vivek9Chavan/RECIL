from torchvision import transforms as pth_transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

from utlilties import utils, vision_transformer as vits

import torch.nn as nn
import numpy as np
import pandas as pd
import math

f = lambda i: i >= 0 and f(math.floor(i / 26 - 1)) + chr(int(round(65 + i % 26))) or ''

class DinoSampler():

    def __init__(self,device = 'cpu'):
        self.model = vits.vit_small()
        self.device=device
        if self.device == 'cuda':
            self.model.cuda()
        if self.device == 'cpu':
            self.model = self.model.cpu()
        self.model=utils.load_pretrained_weights(model=self.model, pretrained_weights=True, checkpoint_key='teacher',
                                      model_name='vit_small',
                                      patch_size=16)
        self.model.eval()
        self.transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def resize_image(self,img):
        aspect_ratio = img.size[0] / img.size[1]
        img = img.resize((int(224 * aspect_ratio), 224))
        return img

    def get_vectors(self,pil_image):
        """
        Get the feature vectors for all images in the path.
        """
        img = pil_image.convert('RGB')
        img = self.resize_image(img)
        img = self.transform(img)
        # add batch dimension
        img = img.unsqueeze(0)
        # Get the feature vector for the image
        if self.device == 'cuda':
            vector = self.model(img.cuda(non_blocking=True))
        if self.device == 'cpu':
            vector = self.model(img)
        # making rank 0
        vector = nn.functional.normalize(vector, dim=1, p=2)
        # get image name
        target = pil_image
        # print(target)
        return vector, target

    def get_all_vectors(self,image_list):
        """
        from image path, extract vectors, downsample and plot a plotly scatter plot
        get all images in the path
        """
        all_vectors = []
        for img in image_list:
            # print(image)
            vector, target = self.get_vectors(img)
            vector = vector.detach().cpu().numpy()
            # cut the target to only have the name of the image
            # target = target.split('/')[-1]
            # remove the extension
            # target = target.split('.')[0]
            # store all in all_vectors
            # target = Image.open(image).convert('RGB')
            # resize image
            # target = target.resize((512, 512))
            # target = np.array(target)
            vectors = [target, vector]
            all_vectors.append(vectors)
            all_vectors = np.concatenate(all_vectors[:, 1], axis=0)
            all_vectors = np.concatenate(all_vectors, axis=0)
        return np.array(all_vectors, dtype=object)

    def do_pca(self,original_feature_list,dim=3,save_file=None):
        #original_array = np.concatenate(original_feature_list, axis=0)
        pca = PCA(n_components=dim, svd_solver='full', random_state=404543)
        pca_array = pca.fit_transform(original_feature_list)
        if dim==2:
            col_names=['x','y']
        elif dim == 3:
            col_names=['x','y','z']
        else:
            col_names=[f(i) for i in range(0,dim)]
        pca_df = pd.DataFrame(pca_array, columns=col_names)
        if save_file is not None:
            pca_df.to_csv(save_file, index=False)
            print('Data has been exported to csv file: pca_results.csv')
        return pca_array,pca_df

    def do_clustering(self,k_cluster,feature_list,random_state=0,n_init=10, init='k-means++'):
        kmeans = KMeans(n_clusters=k_cluster, random_state=random_state,init=init, n_init=n_init).fit(feature_list)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(feature_list)
        X_dist = kmeans.transform(feature_list) ** 2
        sns.countplot(pred_clusters)
        return centroids,pred_clusters,X_dist

    def show_kmeans_WSS(self,data,max_iter=20):
        WCSS = []
        for i in range(1, max_iter):
            model = KMeans(n_clusters=i, init='k-means++')
            model.fit(data)
            WCSS.append(model.inertia_)
        fig = plt.figure(figsize=(7, 7))
        plt.plot(range(1, max_iter), WCSS, linewidth=4, markersize=6, marker='o', color='red')
        plt.xticks(np.arange(max_iter+1))
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        plt.show()
