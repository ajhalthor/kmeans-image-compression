import numpy as np
from kmeans import KMeans
import matplotlib
import matplotlib.pyplot as plt

def transform_image(image, code_vectors):
    '''Quantize image using the code_vectors

    Return new image from the image by replacing each RGB value in image 
    with nearest code vectors (nearest in euclidean distance sense).

    Args
    ----
        image (numpy.ndarray): Input image to quantize 
        code_vectors (numpy.array): centroids

    Returns
    -------
        new_image (numpy.ndarray): Quantized image
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    
    D1 , D2 , _ = image.shape
    K , _ = code_vectors.shape 

    new_image = np.zeros(image.shape)

    for d1 in range(D1):
        for d2 in range(D2):

            dist = np.zeros(K)
            for k in range(K):
                dist[k] = np.inner(image[d1, d2] - code_vectors[k], image[d1, d2] - code_vectors[k])

            k = np.argmin(dist)

            new_image[d1, d2] = code_vectors[k]

    return new_image


def kmeans_image_compression():
    """Use KMeans for image compression.

    Notes
    -----
        - Load an image
        - Scale it to [0, 1] and compress it
        - Save the results in png and npz formats

    """
    im = plt.imread('colorful_img.jpg')
    N, M = im.shape[:2]
    im = im / 255 # Easier, more powerful to process images with a continuous scale.

    # convert to RGB array
    data = im.reshape(N * M, 3)

    k_means = KMeans(n_cluster=16, max_iter=100, e=1e-6)
    centroids, _, i = k_means.fit(data)

    print('RGB centroids computed in {} iteration'.format(i))
    new_im = transform_image(im, centroids)

    assert new_im.shape == im.shape, \
        'Shape of transformed image should be same as image'

    mse = np.sum((im - new_im)**2) / (N * M)
    print('Mean square error per pixel is {}'.format(mse))
    plt.imsave('plots/compressed_colorful_img.png', new_im)

    np.savez('results/k_means_compression.npz', im=im, centroids=centroids,
             step=i, new_image=new_im, pixel_error=mse)


if __name__ == '__main__':
    kmeans_image_compression()
