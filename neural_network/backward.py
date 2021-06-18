from neural_network.utils import *


def backward_convolution(dconv_prev, conv_in, filt, s):
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape

    # initialize derivatives
    dout = np.zeros(conv_in.shape)
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f, 1))
    for curr_f in range(n_f):
        # loop by filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # filter loss gradient
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y + f, curr_x:curr_x + f]
                # input loss gradient
                dout[:, curr_y:curr_y + f, curr_x:curr_x + f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

        # bias loss gradient
        dbias[curr_f] = np.sum(dconv_prev[curr_f])

    return dout, dfilt, dbias


def backward_maxpool(dpool, orig, f, s):
    (n_c, orig_dim, _) = orig.shape

    dout = np.zeros(orig.shape)

    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
                dout[curr_c, curr_y + a, curr_x + b] = dpool[curr_c, out_y, out_x]

                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return dout
