import datahandling as D

load_folder = "F:/Fer-novo/Diplomski projekt/current_data_for_use" # data folder
temp_save_folder = "F:/Fer-novo/Diplomski projekt/temp_saves" # folder to save compact data format

# used to load data and save it in a more compact way, faster to reuse

num = 250
data, chimeric, repeat, normal, original_data = D.loadFast(load_folder, num=num)
D.saveTemp(temp_save_folder, data, chimeric, repeat, normal, original_data)

