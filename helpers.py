import numpy as np
import matplotlib.pyplot as plt

def create_random_2d_dataset_2classes(mu_1=1.0, sigma_1=2.0, mu_2=4.0, sigma_2=1.0, save=True, file_name='random_data_matrix.npy'):
    x_1 = np.random.normal(mu_1, sigma_1, 20)
    y_1 = np.random.normal(mu_1, sigma_1, 20)

    x_2 = np.random.normal(mu_2, sigma_2, 20)
    y_2 = np.random.normal(mu_2, sigma_2, 20)

    data_1 = np.hstack((x_1.reshape(-1, 1), y_1.reshape(-1, 1)))
    data_2 = np.hstack((x_2.reshape(-1, 1), y_2.reshape(-1, 1)))
    data_1 = np.hstack((data_1, np.zeros((data_1.shape[0], 1))))
    data_2 = np.hstack((data_2, np.ones((data_2.shape[0], 1))))
    random_data_matrix = np.vstack((data_1, data_2))
    if save:
        np.save(file_name, random_data_matrix, allow_pickle=False, fix_imports=False)
    return random_data_matrix


def create_random_2d_dataset_3classes(mu_1=1.0, sigma_1=2.0, mu_2=4.0, sigma_2=1.0, mu_3=-1.0, sigma_3=2.0, save=True, file_name='random_data_matrix_3_classes.npy'):
    x_1 = np.random.normal(mu_1, sigma_1, 20)
    y_1 = np.random.normal(mu_1, sigma_1, 20)

    x_2 = np.random.normal(mu_2, sigma_2, 20)
    y_2 = np.random.normal(mu_2, sigma_2, 20)
    
    x_3 = np.random.normal(mu_3, sigma_3, 20)
    y_3 = np.random.normal(mu_3, sigma_3, 20)
    
    data_1 = np.hstack((x_1.reshape(-1, 1), y_1.reshape(-1, 1)))
    data_2 = np.hstack((x_2.reshape(-1, 1), y_2.reshape(-1, 1)))
    data_3 = np.hstack((x_3.reshape(-1, 1), y_3.reshape(-1, 1)))
    
    data_1 = np.hstack((data_1, np.zeros((data_1.shape[0], 1))))
    data_2 = np.hstack((data_2, np.ones((data_2.shape[0], 1))))
    data_3 = np.hstack((data_3, np.ones((data_3.shape[0], 1)) * 2))
    
    random_data_matrix = np.vstack((data_1, data_2, data_3))
    if save:
        np.save(file_name, random_data_matrix, allow_pickle=False, fix_imports=False)
    return random_data_matrix


def create_plot(data, labels, clf, title='Your Title', legend_loc='upper left', colormap='coolwarm', alpha=0.6, figur_size=(20, 15), steps=0.1):
    x = data[:,0]
    y = data[:,1]
    classes = np.unique(labels)
    n_classes = len(np.unique(labels))

    labels_axis = ['class ' + str(c) for c in classes.astype(np.int)]

    cm = plt.cm.get_cmap(colormap, n_classes)
    x_min = x.min() - 2
    x_max = x.max() + 2
    y_min = y.min() - 2
    y_max = y.max() + 2
    XX, YY = np.meshgrid(np.arange(x_min, x_max, steps), np.arange(y_min, y_max, steps))
    # prediction for each point in the meshgrid
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    # reshape for the grid
    Z = Z.reshape(XX.shape)

    plt.figure(figsize=figur_size)
    # contour plot
    plt.contourf(XX, YY, Z, cmap=plt.cm.get_cmap(colormap), alpha=alpha)
    
    for i in range(n_classes):
        curr_class_data = data[np.where(labels==classes[i])[0].tolist()]
        plt.scatter(curr_class_data[:,0], curr_class_data[:,1], c=np.array([cm(i)]), label=labels_axis[i])
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.legend()
    plt.title(title)
    plt.show()


def create_sub_plot(data, labels, clf, ax, title='Your Title', legend_loc='upper left', colormap='coolwarm', alpha=0.6, steps=0.1):
    x = data[:,0]
    y = data[:,1]
    classes = np.unique(labels)
    n_classes = len(np.unique(labels))

    labels_axis = ['class ' + str(c) for c in classes.astype(np.int)]

    cm = plt.cm.get_cmap(colormap, n_classes)
    x_min = x.min() - 2
    x_max = x.max() + 2
    y_min = y.min() - 2
    y_max = y.max() + 2
    XX, YY = np.meshgrid(np.arange(x_min, x_max, steps), np.arange(y_min, y_max, steps))
    # prediction for each point in the meshgrid
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    # reshape for the grid
    Z = Z.reshape(XX.shape)

    # contour plot
    ax.contourf(XX, YY, Z, cmap=plt.cm.get_cmap(colormap), alpha=alpha)
    
    for i in range(n_classes):
        curr_class_data = data[np.where(labels==classes[i])[0].tolist()]
        ax.scatter(curr_class_data[:,0], curr_class_data[:,1], c=np.array([cm(i)]), label=labels_axis[i])
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')
    ax.legend(loc=legend_loc)
    ax.set_title(title)