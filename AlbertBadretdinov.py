import glob
import os

from matplotlib import numpy as np
import math
import matplotlib.image as mpimg


def create_folder(foldername):
    try:
        os.mkdir(foldername)
    except OSError:
        print("Creation of the directory %s failed (Already exist)" % foldername)
        pass


def fft(row):
    N = row.shape[0]
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 2:  # this cutoff should be optimized
        return dft(row)
    else:
        X_even = fft(row[::2])
        X_odd = fft(row[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        concatenated = np.concatenate([X_even + factor[:N // 2] * X_odd,
                                       X_even + factor[N // 2:] * X_odd])
        return concatenated


def ifft(row):
    N = row.shape[0]
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 2:  # this cutoff should be optimized
        return dft(row)
    else:
        X_even = ifft(row[::2])
        X_odd = ifft(row[1::2])
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        concatenated = np.concatenate([X_even + factor[:N // 2] * X_odd,
                                       X_even + factor[N // 2:] * X_odd])
        return concatenated


def dft(row):
    row = np.asarray(row, dtype=float)
    N = row.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, row)


def idft(row):
    row = np.asarray(row, dtype=float)
    N = row.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, row) / N


def shuffle(row):
    N = math.ceil(math.log2(len(row)))
    new_row = np.array([0] * 2 ** N)
    for i in range(len(row)):
        formated = format(i, '0' + str(N) + 'b')
        reversed_string = [formated[len(formated) - 1 - k] for k in range(len(formated))]
        formated = ''
        for j in reversed_string:
            formated += j
        formated = int(formated, 2)
        new_row[formated] = int(row[i])
    return new_row


def fftc(file):
    file_for_compression = file
    # file_for_compression = np.array([shuffle(row) for row in file])
    # file_for_compression = np.transpose(file_for_compression)
    # file_for_compression = np.array([shuffle(row) for row in file_for_compression])
    # file_for_compression = np.transpose(file_for_compression)

    col_idft = np.transpose(file_for_compression)
    col_idft = [np.fft.fft(row) for row in col_idft]
    row_idft = np.transpose(col_idft)
    row_idft = [np.fft.fft(row) for row in row_idft]
    return row_idft


def fftdc(compressed):
    file_for_decompression = compressed

    col_idft = np.transpose(file_for_decompression)
    col_idft = [np.fft.ifft(row) for row in col_idft]
    row_idft = np.transpose(col_idft)
    row_idft = [np.fft.ifft(row) for row in row_idft]

    # file_for_decompression = np.array([shuffle(row) for row in row_idft])
    # file_for_decompression = np.transpose(file_for_decompression)
    # file_for_decompression = np.array([shuffle(row) for row in file_for_decompression])
    # file_for_decompression = np.transpose(file_for_decompression)
    return np.array(row_idft)


if __name__ == '__main__':
    types = ['.jpg']  # Making the list of possible file types
    output_path = 'AlbertBadretdinovOutputs'
    create_folder(output_path)
    for file_path in glob.glob("input" + "/*", recursive=True):  # And for every file of the type do
        file = mpimg.imread(file_path)
        compressed = fftc(file)
        compressed_output_path = output_path + '/' + file_path[6:-4] + 'Compressed' + types[0]
        comp_save = np.log(np.abs(compressed))
        mpimg.imsave(compressed_output_path, comp_save, cmap='gray', vmin=0, vmax=np.amax(comp_save))

        decompressed = fftdc(compressed)
        decompressed_output_path = output_path + '/' + file_path[6:-4] + 'Decompressed' + types[0]
        decomp_save = np.log(np.abs(decompressed))
        mpimg.imsave(decompressed_output_path, decomp_save, cmap='gray', vmin=0, vmax=decomp_save.max())

