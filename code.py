'''
Question 2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
from q1 import logsumexp_stable

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for k in range(10):
        # get subset of digits that belong to k class
        data_k = data.get_digits_by_label(train_data, train_labels, k)
        # take avg of each feature in this subset of digits
        means[k] = np.mean(data_k, axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))

    means = compute_mean_mles(train_data, train_labels)
    identity = 0.01 * np.identity(64)

    # Compute covariances
    for k in range(10):
        # digits in k class - its mean_mles
        x_sub_mu = data.get_digits_by_label(train_data, train_labels, k) - means[k]
        # take avg of x_sub_mu.T @ x_sub_mu
        covariances[k] = x_sub_mu.T @ x_sub_mu / x_sub_mu.shape[0]    
        # add a small multiple of the identity to ensure numerical stability
        covariances[k] += identity
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    n, d = digits.shape
    generative_likelihood = np.zeros((n, 10))
    for k in range(10):
        # calculate log p(x|y=k,mu,Sigma)
        part1 =  -d/2 * np.log(2*np.pi) - 1/2 * np.log((np.linalg.det(covariances[k])))
        part2 = -1/2 * (digits - means[k]) @ np.linalg.inv(covariances[k]) * (digits - means[k])
        generative_likelihood[:,k] = part1 + np.sum(part2, axis=1)
    return generative_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    d = digits.shape[1]
    gen_likelihood = generative_likelihood(digits, means, covariances)

    # caculate log(p(x)) = log(sum_k(exp(logp(x,y=k)))), where 
    # logp(x,y=k) = logp(x|y=k) + logp(y=k) = gen_likelihood + log(1/10)
    log_p_x = logsumexp_stable(gen_likelihood + np.log(1/10), axis=1)

    conditional_likelihood = gen_likelihood + np.log(1/10) - np.tile(log_p_x, (10,1)).T
    return conditional_likelihood

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    n = digits.shape[0]
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # repsent lables in 2D one-hot array. label_matrix[i] = [0,0,0,0,1,0,0,0,0,0] repsents labels[i] = 4
    label_matrix = np.zeros((n, 10))
    label_matrix[np.arange(n), labels.astype(int)] = 1

    # only the cond_likelihood of true class labels left
    label_likelihood = cond_likelihood * label_matrix

    avg_cond_likelihood = np.sum(label_likelihood) / n
    return avg_cond_likelihood

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    prediction = np.argmax(cond_likelihood, axis=1)
    return prediction

def plot_images(images, ax, ims_per_row=5, padding=2, digit_dimensions=(8, 8),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)

def main():
    data.load_all_data_from_zip("hw4digits.zip", "./data/")
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    
    # 2a. calculate average conditional log-likelihood on the training and test set
    train_avg_like = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_like = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print(f'average conditional log-likelihood on the training set is {train_avg_like}')
    print(f'average conditional log-likelihood on the test set is {test_avg_like}')

    # 2b. make prediction and report accuracy on training and test set
    train_pred = classify_data(train_data, means, covariances)
    train_accu = np.sum(train_pred == train_labels) / train_data.shape[0]
    print(f'accuracy on training set is {train_accu}')
    
    test_pred = classify_data(test_data, means, covariances)
    test_accu = np.sum(test_pred == test_labels) / test_data.shape[0]
    print(f'accuracy on test set is {test_accu}')

    # 2c. compute the leading eigenvectors for each class covariance matrix
    lead_egvec = np.zeros((10, 8, 8))
    for k in range(10):
        values, vectors = np.linalg.eig(covariances[k])
        lead_egvec[k] = vectors[:, np.argmax(values)].reshape((8, 8))
        
    save_images(lead_egvec, '2c.leading eigenvectors for each class.png')

if __name__ == '__main__':
    main()
