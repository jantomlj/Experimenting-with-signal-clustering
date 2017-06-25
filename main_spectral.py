from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from VariationalAutoencoder import VariationalAutoencoder
from GaussianVariationalAutoencoder import GaussianVariationalAutoencoder
from DenoisingAutoencoder import DenoisingAutoencoder
import plotting as P
import datahandling as D
import accuracy as A
import numpy as np
from sklearn.manifold import TSNE

load_folder = "F:/Fer-novo/Diplomski projekt/current_data_for_use" # folder where data is stored
save_folder = "F:/Fer-novo/Diplomski projekt/results_dn_spectral_v2" # folder where clusters are stored
temp_save_folder = "F:/Fer-novo/Diplomski projekt/temp_saves" # folder where data is stored in a compact way, faster to load from here

num = 250

batch_size = 20
test_set_size = 60

n_clusters = 3 # <= 8
n_z = 12

data, chimeric, repeat, normal, original_data = D.loadTemp(temp_save_folder)

test_set_list = D.load_test_set(load_folder, num=num)
n_samples = len(data)

var_auto = DenoisingAutoencoder(250, 220, 200, 150, 150, 200, 220, 250, n_z, num, batch_size)
var_auto.train(n_samples, data, n_epochs=20)

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

# adding test set to training set
for test_data in reversed(test_set_list):
    data = np.append(test_data, data, axis=0)

zs = var_auto.encode(data)


#model = KMeans(n_clusters=n_clusters, random_state=42)
model = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors")
results = model.fit_predict(zs).astype(int)

test_results = []
for i in range(0, 4):
    ind = test_set_size * i
    test_results.append(results[ind:(ind+test_set_size)])

# tsne on clusters on train data
tsne = TSNE(n_components = 2, random_state=42)
tsne_results = tsne.fit_transform(zs[240:1540])
P.scatterTsne(tsne_results, results[240:1540])

#tsne for test data
test_data_all = np.array([]).reshape(0, num)
result_test_all = np.array([])
for test_data, result_data in zip(test_set_list, test_results):
    test_data_all = np.append(test_data_all, test_data, axis=0)
    result_test_all = np.append(result_test_all, result_data)

result_test_all = result_test_all.astype(int)
padding = np.zeros(1000*num).reshape(1000, num)
test_data_all = np.append(test_data_all, padding.reshape(-1, num), axis=0)
        
zs_test = var_auto.encode(test_data_all)[:(test_set_size * len(test_set_list))]
tsne_results_test = tsne.fit_transform(zs_test)
P.scatterTsne(tsne_results_test, result_test_all)

#save results
D.saveResults(save_folder, results[:len(original_data)], original_data)

print(A.confusion_matrix_spectral(test_results, n_clusters))
