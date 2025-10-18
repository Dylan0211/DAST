import numpy as np
from sklearn.manifold import TSNE
from matplotlib.pyplot import MultipleLocator

import numpy as np
import matplotlib.pyplot as plt


def draw_tsne_for_real_and_syn_data(real_data, syn_data, title):
    """
    :param real_data: normalized real data
    :param syn_data: normalized syn data
    """
    # t-SNE
    input_data = np.concatenate((real_data, syn_data), axis=0)
    length = len(real_data)

    # for p in range(2, 51):
    #     tsne = TSNE(n_components=2, perplexity=p, init='pca', n_iter=250)
    #     tsne_data = tsne.fit_transform(input_data)
    #
    #     area = np.pi * 5 ** 2
    #
    #     plt.figure(figsize=(8, 8))
    #
    #     plt.subplot(111)
    #     for i in range(tsne_data[:length].shape[0]):
    #         plt.scatter(tsne_data[:length][i, 0], tsne_data[:length][i, 1], s=area, color='r', label='original data',
    #                     alpha=0.4)
    #     for i in range(tsne_data[length:].shape[0]):
    #         plt.scatter(tsne_data[length:][i, 0], tsne_data[length:][i, 1], s=area, color='b', label='synthetic data',
    #                     alpha=0.4)
    #
    #     # ax = plt.gca()
    #     # ax.yaxis.tick_right()
    #     # plt.xticks(np.arange(-10, 10, 5), size=25)
    #     # plt.yticks(np.arange(-10, 10, 5), size=25)
    #     # plt.xlim((-10.0, 10.0))
    #     # plt.ylim((-10.0, 10.0))
    #     plt.xlabel('x_tsne', fontsize=25)
    #     plt.ylabel('y_tsne', fontsize=25)
    #     plt.title('p = {}'.format(p))
    #
    #     plt.show()
    #     # plt.savefig('tsne.svg')

    # t-SNE
    input_data = np.concatenate((real_data, syn_data), axis=0)
    length = len(real_data)

    tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=250)
    tsne_data = tsne.fit_transform(input_data)

    area = np.pi * 4 ** 2

    plt.figure(figsize=(6, 6))

    original_x_val, original_y_val = [], []
    for i in range(tsne_data[:length].shape[0]):
        original_x_val.append(tsne_data[:length][i, 0])
        original_y_val.append(tsne_data[:length][i, 1])
    synthetic_x_val, synthetic_y_val = [], []
    for i in range(tsne_data[length:].shape[0]):
        synthetic_x_val.append(tsne_data[length:][i, 0])
        synthetic_y_val.append(tsne_data[length:][i, 1])

    plt.subplot(111)
    plt.scatter(original_x_val, original_y_val, s=area, color='b', label='original data', alpha=0.4)
    plt.scatter(synthetic_x_val, synthetic_y_val, s=area, color='r', label='synthetic data', alpha=0.4)

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(2))
    # ax.yaxis.set_major_locator(MultipleLocator(2))

    # ax = plt.gca()
    # ax.yaxis.tick_right()
    plt.xticks(size=20)
    plt.yticks(size=20)
    # plt.xticks(np.arange(-10, 10, 4), size=20)
    plt.yticks([-2, -1, 0, 1, 2], size=20)
    # plt.xlim((-5.0, 5.0))
    plt.ylim((-2.2, 2.2))
    plt.xlabel('x_tsne', fontsize=20)
    plt.ylabel('y_tsne', fontsize=20)
    plt.tight_layout()
    # plt.legend(loc='upper right', prop={'size': 10}, ncol=1)

    # plt.show()
    plt.savefig('./tmp_data/output_files/{}.pdf'.format(title))

