import numpy as np
import os
from sklearn.preprocessing import normalize

def load(folder, num=400):
    chimeric, original_chimeric = loadSingleFolder(folder + "/chimeric", num)
    repeat, original_repeat = loadSingleFolder(folder + "/repeat", num)
    normal, original_normal = loadSingleFolder(folder + "/normal", num)
    
    combined = np.array([]).reshape(0, num)
    combined = np.append(combined, chimeric, axis=0)
    combined = np.append(combined, repeat, axis=0)
    combined = np.append(combined, normal, axis=0)
    
    original_combined = original_chimeric + original_repeat + original_normal
    
    original_combined_np = np.array(original_combined)
    
    p = np.random.permutation(len(combined))
    combined = combined[p]
    original_combined_np = original_combined_np[p]
    
    return combined, chimeric, repeat, normal, original_combined_np

def loadFast(folder, num=250):
    chimeric, original_chimeric = loadSingleFolderFast(folder + "/chimeric", num)
    repeat, original_repeat = loadSingleFolderFast(folder + "/repeat", num)
    normal, original_normal = loadSingleFolderFast(folder + "/normal", num)
    
    combined = np.array([]).reshape(0, num)
    combined = np.append(combined, chimeric, axis=0)
    combined = np.append(combined, repeat, axis=0)
    combined = np.append(combined, normal, axis=0)
    
    original_combined = original_chimeric + original_repeat + original_normal
    original_combined_np = np.array(original_combined)
    
    p = np.random.permutation(len(combined))
    combined = combined[p]
    original_combined_np = original_combined_np[p]
    
    return combined, chimeric, repeat, normal, original_combined_np
    

def load_test_set(folder, num=400):
    chimeric, _ = loadSingleFolder(folder + "/chimeric_test", num)
    left_repeat, _ = loadSingleFolder(folder + "/left_repeat_test", num)
    right_repeat, _ = loadSingleFolder(folder + "/right_repeat_test", num)
    normal, _ = loadSingleFolder(folder + "/normal_test", num)
    
    test_set_list = []
    test_set_list.append(chimeric)
    test_set_list.append(left_repeat)
    test_set_list.append(right_repeat)
    test_set_list.append(normal)
    
    return test_set_list
    
def loadSingleFolderFast(folder, num=250):
    result = np.array([]).reshape(0, num)
    original_result = []
    for file in os.listdir(folder):
        original = np.load(folder + "/" + file)
        result = np.append(result, normalize(interpolate(original, num)).reshape(1, num), axis=0)
        original_result.append(original)
    return result, original_result


def loadSingleFolder(folder, num=400):
    result = np.array([]).reshape(0, num)
    original_result = []
    for file in os.listdir(folder):
        original = list(np.load(folder + "/" + file))
        result = np.append(result, normalize(interpolate(original, num)).reshape(1, num), axis=0)
        original_result.append(original)
    return result, original_result

def saveResults(save_folder, results, original_data):
    i = 0
    for res in results:
        np.save(save_folder + "/group_" + str(res) + "/number_" + str(i) + ".npy", original_data[i])
        i+=1

def saveTemp(temp_save_folder, data, chimeric, repeat, normal, original_data):
    np.save(temp_save_folder + "/data.npy", data)
    np.save(temp_save_folder + "/chimeric.npy", chimeric)
    np.save(temp_save_folder + "/repeat.npy", repeat)
    np.save(temp_save_folder + "/normal.npy", normal)
    np.save(temp_save_folder + "/original_data.npy", original_data)
    
def loadTemp(temp_save_folder):
    data = np.load(temp_save_folder + "/data.npy")
    chimeric = np.load(temp_save_folder + "/chimeric.npy")
    repeat = np.load(temp_save_folder + "/repeat.npy")
    normal = np.load(temp_save_folder + "/normal.npy")
    original_data = np.load(temp_save_folder + "/original_data.npy")
    return data, chimeric, repeat, normal, original_data

#interpolation
def interpolate(coverage, num=400):
    x = np.linspace(0, 1, num=num)
    xp = np.linspace(0, 1, num=len(coverage))
    return np.interp(x, xp, coverage)

def f(x, ma, mi):
    return (x-mi) / (ma-mi)

def normalize(read):
    ma, mi = read.max(), read.min()
    fun = np.vectorize(f)
    return fun(read, ma, mi)