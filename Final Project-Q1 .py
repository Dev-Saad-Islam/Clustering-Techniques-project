import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
import plotly.express as px
import seaborn as sns
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

#==================================================================================================================================

data = pd.read_csv("ClimateDataBasel.csv")

data.columns = ['Temp(Min)(C)','Temp(Max)(C)','Temp(Mean)(C)','Rel.Humidity.(Min).%','Rel.Humidity.(Max).%',
                'Rel.Humidity.(Mean).%','Sea.Level.Pressure.(Min)(hPa)','Sea.Level.Pressure.(Max)(hPa)',
                'Sea.Level.Pressure.(Mean)(hPa)','Precipitation.Total(mm)','Snowfall.Amount(cm)','Sunshine.Duration(Min)',
                'Wind.Gust.(Min)(Km/h)','Wind.Gust.(Max)(Km/h)','Wind.Gust.(Mean)(Km/h)','Wind.Speed.(Min)(Km/h)',
                'Wind.Speed.(Max)(Km/h)','Wind.Speed.(Mean)(Km/h)']

feature_names = ['Temp(Min)(C)','Temp(Max)(C)','Temp(Mean)(C)','Rel.Humidity.(Min).%','Rel.Humidity.(Max).%',
                'Rel.Humidity.(Mean).%','Sea.Level.Pressure.(Min)(hPa)','Sea.Level.Pressure.(Max)(hPa)',
                'Sea.Level.Pressure.(Mean)(hPa)','Precipitation.Total(mm)','Snowfall.Amount(cm)','Sunshine.Duration(Min)',
                'Wind.Gust.(Min)(Km/h)','Wind.Gust.(Max)(Km/h)','Wind.Gust.(Mean)(Km/h)','Wind.Speed.(Min)(Km/h)',
                'Wind.Speed.(Max)(Km/h)','Wind.Speed.(Mean)(Km/h)']


print(data.shape)

#checking for null / missing values
print(data.isnull().sum)

standardized_data = pd.DataFrame(StandardScaler().fit_transform(data))

#plotting data after standardization.
sns.boxplot(standardized_data)
plt.show()
#==================================================================================================================================
#zscore for outlier removal:
z_scores = stats.zscore(standardized_data)
abs_z_scores = np. abs(z_scores)
filtered_entries = (abs_z_scores< 3). all(axis=1)
standardized_filtered_data = standardized_data[filtered_entries]

#plotting data after outlier removal.
sns.boxplot(standardized_filtered_data)
plt.show()
 
print(standardized_filtered_data.shape)
#170 entries removed as outliers
#==================================================================================================================================
#applying PCA for 92% variance
pca = PCA(n_components=0.92)
pca_fit = pca.fit_transform(standardized_filtered_data)

# Plot the explained variances
 
ex_var_pca = pca.explained_variance_ratio_
cum_sm_eigenvalues = np.cumsum(ex_var_pca)

#plotting PCA variance graph:
plt.bar(range(0,len(ex_var_pca)), ex_var_pca, alpha=0.5, align='center', 
        label='Individual explained variance',color="black")
plt.step(range(0,len(cum_sm_eigenvalues)), cum_sm_eigenvalues, where='mid',
         label='Cumulative explained variance',color="red")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#making a PCA Dataframe
pcas_data = pd.DataFrame(data = pca_fit, 
                         columns = 
                         ['PC_1','PC_2'
                          ,'PC_3','PC_4'                          
                          ,'PC_5','PC_6'])

#==================================================================================================================================

# Principal components correlation coefficients
loadings = pca.components_

# Number of features before PCA
n_features = pca.n_features_

# Feature names before PCA

# PC names
pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]

# Match PC names to loadings
pc_loadings = dict(zip(pc_list, loadings))

# Matrix of corr coefs between feature names and PCs
loadings_df = pd.DataFrame.from_dict(pc_loadings)
loadings_df['feature_names'] = feature_names
loadings_df = loadings_df.set_index('feature_names')

ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()

#==================================================================================================================================

model = KMeans()
# k is range of number of clusters.
visualizer_kmean = KElbowVisualizer(model, k=(2,10), timings= False)
visualizer_kmean.fit(pcas_data)        # Fit data to visualizer
visualizer_kmean.show()

# Create a KMeans instance with k clusters: model
model = KMeans(n_clusters=visualizer_kmean.elbow_value_)

# Fit model to samples
kmean_label = model.fit_predict(pca_fit)

sns.scatterplot(x=pcas_data["PC_1"], y=pcas_data["PC_2"],hue=kmean_label)
plt.show()

total_var = ex_var_pca.sum() * 100

#3d plot for the Kmeans model
fig_kmean = px.scatter_3d(
    pcas_data, x='PC_1', y='PC_2', z='PC_3', color=kmean_label,
    title=f'Total Explained Variance: {total_var:.2f}% \n KMeans Clustering',
)
fig_kmean.show()


#==================================================================================================================================
#silhouette method to determine the number of clusters
silhouette_avg = []
for num_clusters in range(2,10):
    # initialise kmeans
    birch = Birch(n_clusters=num_clusters)
    birch.fit(pcas_data)
    cluster_labels = birch.labels_
    # silhouette score
    silhouette_avg.append(silhouette_score(pcas_data, cluster_labels))

plt.plot(range(2,10),silhouette_avg,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')
plt.show()

#making the birch model for 4 clusters
Birch_clustering = Birch(n_clusters=4)
Birch_label= Birch_clustering.fit_predict(pca_fit)

#3d plot for the birch model
fig_BIRCH = px.scatter_3d(
    pcas_data, x='PC_1', y='PC_2', z='PC_3', color=Birch_label,
    title=f'Total Explained Variance: {total_var:.2f}% \n Birch Clustering',
)
fig_BIRCH.show()



#==================================================================================================================================

optic_clustering = OPTICS(min_samples=4)
optic_labels= optic_clustering.fit_predict(pca_fit)

#3d plot for the OPTICS model
fig_optics = px.scatter_3d(
    pcas_data, x='PC_1', y='PC_2', z='PC_3', color=optic_labels,
    title=f'Total Explained Variance for Optic Clustering: {total_var:.2f}% \n Optic Clustering',
)
fig_optics.show()


#==================================================================================================================================
