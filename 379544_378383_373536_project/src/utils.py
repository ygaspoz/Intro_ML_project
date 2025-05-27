import numpy as np 


# Generaly utilies
##################
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    # Split the data
    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)

def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (array): of shape (N,D)
    Returns:
        (array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]),data], axis=1)
    return data

def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.
    
    Arguments:
        data (array): of shape (N,D)
        means (array): of shape (1,D)
        stds (array): of shape (1,D)
    Returns:
        (array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds

def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.
    
    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)


# Metrics
#########

def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.

def macrof1_fn(pred_labels, gt_labels):
    """Return the macro F1-score."""
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels==val)
        
        tp = sum(predpos*gtpos)
        fp = sum(predpos*~gtpos)
        fn = sum(~predpos*gtpos)
        if tp == 0:
            continue
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

        macrof1 += 2*(precision*recall)/(precision+recall)

    return macrof1/len(class_ids)

def mse_fn(pred,gt):
    """
        Mean Squared Error
        Arguments:
            pred: NxD prediction matrix
            gt: NxD groundtruth values for each predictions
        Returns:
            returns the computed loss
    """
    loss = (pred-gt)**2
    loss = np.mean(loss)
    return loss


def augment_data(images, labels, allow_rotations=True, flip_horizontal=True, flip_vertical=False):
    """
    Augments image data with random 90-degree rotations and flips using only NumPy.

    Arguments:
        images (np.ndarray): Original training images. Expected shape (N, H, W, C) or (N, C, H, W).
        labels (np.ndarray): Corresponding labels (N,).
        allow_rotations (bool): Whether to apply random 90, 180, or 270 degree rotations.
        flip_horizontal (bool): Whether to apply random horizontal flips.
        flip_vertical (bool): Whether to apply random vertical flips.

    Returns:
        augmented_images (np.ndarray): Augmented images.
        augmented_labels (np.ndarray): Corresponding labels for augmented images.
    """
    augmented_images_list = []
    augmented_labels_list = []

    if images.ndim == 4:
        if images.shape[1] < images.shape[2] and images.shape[1] < images.shape[3]:
            is_channels_first = True
            axes_for_rotation = (2, 3)
            h_axis = 2
            w_axis = 3
        else:
            is_channels_first = False
            axes_for_rotation = (1, 2)
            h_axis = 1
            w_axis = 2
    elif images.ndim == 3:
        is_channels_first = False
        axes_for_rotation = (1, 2)
        h_axis = 1
        w_axis = 2
    else:
        print("Warning: Unexpected image dimensions for augmentation. Skipping augmentation.")
        return images, labels

    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i]

        augmented_images_list.append(image)
        augmented_labels_list.append(label)

        if allow_rotations:
            k = np.random.randint(1, 4)

            if image.ndim == 3 and not is_channels_first:
                rotated_image = np.rot90(image, k, axes=(0, 1))
            elif image.ndim == 3 and is_channels_first:
                rotated_image = np.rot90(image, k, axes=(1, 2))
            elif image.ndim == 2:
                rotated_image = np.rot90(image, k, axes=(0, 1))
            else:
                rotated_image = image

            augmented_images_list.append(rotated_image)
            augmented_labels_list.append(label)

        if flip_horizontal:
            if np.random.rand() > 0.5:
                if image.ndim == 3 and not is_channels_first:
                    flipped_image = image[:, ::-1, :]
                elif image.ndim == 3 and is_channels_first:
                    flipped_image = image[:, :, ::-1]
                elif image.ndim == 2:
                    flipped_image = image[:, ::-1]
                else:
                    flipped_image = image
                augmented_images_list.append(flipped_image)
                augmented_labels_list.append(label)

        if flip_vertical:
            if np.random.rand() > 0.5:
                if image.ndim == 3 and not is_channels_first:
                    flipped_image = image[::-1, :, :]
                elif image.ndim == 3 and is_channels_first:
                    flipped_image = image[:, ::-1, :]
                elif image.ndim == 2:
                    flipped_image = image[::-1, :, :]
                else:
                    flipped_image = image
                augmented_images_list.append(flipped_image)
                augmented_labels_list.append(label)

        """
        noise = np.random.normal(0, 0.05, image.shape)
        noisy_image = np.clip(image + noise, 0, 1)
        augmented_images_list.append(noisy_image)
        augmented_labels_list.append(label)
        """
        factor = 0.8 + 0.4 * np.random.rand()
        bright_image = np.clip(image * factor, 0, 1)
        augmented_images_list.append(bright_image)
        augmented_labels_list.append(label)

    return np.array(augmented_images_list), np.array(augmented_labels_list)