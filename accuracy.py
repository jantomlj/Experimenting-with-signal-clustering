import numpy as np
import pywt

def confusion_matrix(test_set_list, var_auto, model, n_clusters, num, test_set_size):
    conf_matrix = np.zeros(n_clusters * len(test_set_list)).reshape(n_clusters, len(test_set_list))
    i = 0
    for test_data in test_set_list:
        padding = np.zeros(1000*num).reshape(1000, num)
        data = np.append(test_data, padding.reshape(-1, num), axis=0)
        
        zs = var_auto.encode(data)[:test_set_size]
        results = model.predict(zs)
        for j in range(len(results)):
            conf_matrix[results[j]][i] += 1
        
        i+=1
    return conf_matrix

def confusion_matrix_spectral(results, n_clusters):
    conf_matrix = np.zeros(n_clusters * len(results)).reshape(n_clusters, len(results))
    i = 0
    for result_data in results:
        for j in range(len(result_data)):
            conf_matrix[result_data[j]][i] += 1
        
        i+=1
    return conf_matrix
