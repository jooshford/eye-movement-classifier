import numpy as np


def process_data(data):
    data_in = np.array(data)
    data_processed = np.zeros(0)
    i = 1
    while i < len(data_in)-1:
        if data_in[i] > 127:
            # Found beginning of frame
            # Extract one sample from 2 bytes
            intout = (np.bitwise_and(data_in[i], 127))*128
            i = i + 1
            intout = intout + data_in[i]
            data_processed = np.append(data_processed, intout)
        i = i+1

    return data_processed


def process_gaussian_fft(t, data_t, sigma_gauss):
    dt = t[1]-t[0]  # time interval
    maxf = 1/dt     # maximum frequency
    df = 1/(np.max(t) - np.min(t))   # frequency interval
    f_fft = np.linspace(-maxf/2,
                        maxf/2+df,
                        len(data_t))  # define frequency domain

    # DO FFT
    data_f = np.fft.fftshift(np.fft.fft(data_t))  # FFT of data

    # GAUSSIAN FILTER
    gauss_filter = np.exp(-(f_fft)**2/sigma_gauss**2)   # gaussian filter used
    # gaussian filter spectrum in frquency domain
    data_f_filtered = data_f*gauss_filter
    # bring filtered signal in time domain
    data_t_filtered = np.fft.ifft(np.fft.ifftshift(data_f_filtered))
    return [x.real for x in data_t_filtered]
