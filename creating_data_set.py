import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#%matplotlib inline

# Coding for Scissors
scissor_str = r"images\scissor"						# 1083 image
image = img.imread(r'images\scissor\28.jpg')
reshaped_scissor_image_prev = np.reshape(image, (784,))

count = 0

for file in os.listdir(scissor_str):
    if file == '28.jpg':
    	count+=1
    	continue
    filename = os.path.join(scissor_str, file)
    if filename.endswith(".jpg"):
        image = img.imread(filename)
        reshaped_image = np.reshape(image, (784,))
        combined = np.vstack((reshaped_scissor_image_prev, reshaped_image))
        reshaped_scissor_image_prev = combined
        count+=1									

#plt.imshow(reshaped_scissor_image_prev[23].reshape(28, 28))

value_scissor = np.zeros((count,))

#print(count)

for i in range(0, count):
	value_scissor[i] = 2

# Concatenating in the dataframe
dataset_scissor = np.vstack((reshaped_scissor_image_prev.T, value_scissor))	

# Coding for Rock
rock_str = r"images\rock"							# 1072 images
image = img.imread(r'images\rock\26.jpg')
reshaped_rock_image_prev = np.reshape(image, (784,))

count = 0

for file in os.listdir(rock_str):
    if file == '26.jpg':
    	count+=1
    	continue
    filename = os.path.join(rock_str, file)
    if filename.endswith(".jpg"):
        image = img.imread(filename)
        reshaped_image = np.reshape(image, (784,))
        combined = np.vstack((reshaped_rock_image_prev, reshaped_image))
        reshaped_rock_image_prev = combined
        count+=1									

#plt.imshow(reshaped_rock_image_prev[23].reshape(28, 28))

value_rock = np.zeros((count,))

#print(count)

for i in range(0, count):
	value_rock[i] = 1

# Concatenating in the dataframe
dataset_rock = np.vstack((reshaped_rock_image_prev.T, value_rock))

# Coding for Paper
paper_str = r"images\paper"							# 982 images
image = img.imread(r'images\paper\38.jpg')
reshaped_paper_image_prev = np.reshape(image, (784,))

count = 0

for file in os.listdir(paper_str):
    if file == '38.jpg':
    	count+=1
    	continue
    filename = os.path.join(paper_str, file)
    if filename.endswith(".jpg"):
        image = img.imread(filename)
        reshaped_image = np.reshape(image, (784,))
        combined = np.vstack((reshaped_paper_image_prev, reshaped_image))
        reshaped_paper_image_prev = combined
        count+=1

#plt.imshow(reshaped_paper_image_prev[23].reshape(28, 28))

value_paper = np.zeros((count,))

#print(count)

for i in range(0, count):
	value_paper[i] = 0

# Concatenating in the dataframe
dataset_paper = np.vstack((reshaped_paper_image_prev.T, value_paper))

combined_dataset = np.vstack((dataset_scissor.T, dataset_rock.T))
combined_dataset = np.vstack((combined_dataset, dataset_paper.T))

columns = []

for i in range(1, 785):
	columns.append('pixel' + str(i))

columns.append('label')

dataset_df = pd.DataFrame(data = combined_dataset[0:, 0:], index = range(0, 3208), columns = columns)
dataset_df = dataset_df.sample(frac = 1).reset_index(drop = True)
dataset_df.to_csv(r'created_dataset\dataset_df.csv', index = False)