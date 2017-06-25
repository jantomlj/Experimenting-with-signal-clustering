from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from VariationalAutoencoder import VariationalAutoencoder
from GaussianVariationalAutoencoder import GaussianVariationalAutoencoder
from DenoisingAutoencoder import DenoisingAutoencoder
import plotting as P
import datahandling as D
import accuracy as A
import numpy as np
from sklearn.manifold import TSNE
from sklearn.externals import joblib

load_folder = "F:/Fer-novo/Diplomski projekt/current_data_for_use" # folder where data is stored
save_folder = "F:/Fer-novo/Diplomski projekt/results_dn_kmeans_v6" # folder where clusters are stored
temp_save_folder = "F:/Fer-novo/Diplomski projekt/temp_saves" # folder where data is stored in a compact way, faster to load from here

num = 250

batch_size = 10
test_set_size = 60

n_clusters = 4 # <= 8

data, chimeric, repeat, normal, original_data = D.loadTemp(temp_save_folder)

test_set_list = D.load_test_set(load_folder, num=num)
n_samples = len(data)

var_auto = DenoisingAutoencoder(250, 230, 200, 150, 150, 200, 230, 250, 12, num, batch_size)
var_auto.train(n_samples, data, n_epochs=30)

# var_auto.save_model("F:/Fer-novo/Diplomski rad/modeli/saves/autoenkoder.tensorflow") # optional storing the model for future use

# visualisation of the reconstruction on training data
x_sample = data[:batch_size]
x_reconstruct = var_auto.reconstruct(x_sample)
P.plot_reconstructions(x_sample, x_reconstruct)

# ploting unseen test data
x_sample_test_1 = test_set_list[0][:1]
x_sample_test_2 = test_set_list[1][:1]
x_sample_test_3 = test_set_list[2][:1]
x_sample_test_4 = test_set_list[3][:1]
x_sample_test = np.concatenate((x_sample_test_1, x_sample_test_2, x_sample_test_3, x_sample_test_4, test_set_list[0][:batch_size]), axis=0)
x_reconstruct_test = var_auto.reconstruct(x_sample_test)
P.plot_reconstructions(x_sample_test, x_reconstruct_test)

# k means on z
zs = var_auto.encode(data)


model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
#model = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors")
#model = DBSCAN(eps=.2)
results = model.fit_predict(zs)
#joblib.dump(model, "F:/Fer-novo/Diplomski rad/modeli/saves/kmeans.pkl") # optional storing the model for future use

# tsne on clusters on train data
tsne = TSNE(n_components = 2, random_state=42)
tsne_results = tsne.fit_transform(zs[-1300:])
P.scatterTsne(tsne_results, results[-1300:])

#tsne for test data
test_data_all = np.array([]).reshape(0, num)
for test_data in test_set_list:
    test_data_all = np.append(test_data_all, test_data, axis=0)
padding = np.zeros(1000*num).reshape(1000, num)
test_data_all = np.append(test_data_all, padding.reshape(-1, num), axis=0)

zs_test = var_auto.encode(test_data_all)[:(test_set_size * len(test_set_list))]
tsne_results_test = tsne.fit_transform(zs_test)
P.scatterTsne(tsne_results_test, model.predict(zs_test))

#save results
#D.saveResults(save_folder, results, original_data)

print(A.confusion_matrix(test_set_list, var_auto, model, n_clusters, num=num, test_set_size=test_set_size))

