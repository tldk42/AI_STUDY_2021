import pickle
"""with open('mnist.pkl', 'rb') as f:
    dataset = pickle.load(f)
x_train, y_train = dataset['train_img'], dataset['train_label']
x_test, y_test = dataset['test_img'], dataset['test_label']

with open('train_dataset.pkl', 'wb') as f:
    pickle.dump(x_train, f)
    pickle.dump(y_train, f)


with open('test_dataset.pkl', 'wb') as f:
    pickle.dump(x_test, f)
    pickle.dump(y_test, f)"""

with open('train_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
    dataset2 = pickle.load(f)
x, y = dataset, dataset2
print(dataset.shape)