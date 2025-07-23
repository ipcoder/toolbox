""" Bit streams operations"""


def save_bit_streams(full_file_name, streams, order=None):
    """ Save bit stream

    :param full_file_name: the full path of the .tiff file to be saved
    :param streams: dict of dicts of streams. See my_streams in '__main__' for example
    :param order: list of streams' names to record in specific order,
                or list of fio {name:offset} to record in pre-defined offsets
    :return: ---
    """
    import tifffile, json
    from toolbox.utils.binary import join_bits
    streams = streams.copy()

    def extract_bits_data(names):
        return zip(*[(streams[name]['bits'], streams[name]['data']) for name in names])

    bits, data_streams = extract_bits_data(streams)
    check_clip = False
    ord_streams = None

    if order is not None:
        if type(order) == dict:
            offsets = [order[name] for name in streams]
            for name in streams:     # ADD offsets TO streams to be used in unpacking
                streams[name]['offset'] = order[name]
            bits = list(zip(bits, offsets))
        else:
            assert hasattr(order, '__iter__')
            bits, data_streams = extract_bits_data(order)
            ord_streams = {name: streams[name] for name in order}

    tiff_im = join_bits(bits, data_streams, check_clip)

    for name in streams.keys():
        del streams[name]['data']
    metadata = json.dumps(streams if ord_streams is None else ord_streams)  # REMOVE 'data' FROM streams AND PACK INTO JSON

    tifffile.imsave(full_file_name, tiff_im, description=metadata)


def load_bit_streams(full_file_name):
    """ Load bit streams from tiff file with YML metadata

    :param full_file_name:  the full path of the .tiff file to be loaded.
                            the file is assumed to contain YML metadata recorded by save_bit_streams()
    :return: split data streams
    """
    from toolbox.utils.binary import split_bits
    from imread import imread
    tiff_im, streams = imread(full_file_name, get_meta=True)

    bits = [streams[name]['bits'] for name in streams.keys()]

    if 'offset' in list(streams.values())[0].keys():  # list(streams.values())[0]['offset'] != None:
        offsets = [streams[name]['offset'] for name in streams.keys()]
        bits = list(zip(bits, offsets))

    data = split_bits(tiff_im, bits)

    for i, name in enumerate(streams):
        streams[name]['data'] = data[i]

    return streams


if __name__ == '__main__':

    sz_x = 10   #640
    sz_y = 10   #480
    import numpy as np
    conf_inf_bits = 4
    conf_inf = np.random.randint(2 ** conf_inf_bits, size=(sz_x, sz_y)).astype('uint8')

    disp_inf_bits = 1
    disp_inf = np.ones((sz_x, sz_y), dtype='uint8')

    my_streams = {
        'conf_inf': {
            'data': conf_inf,
            'bits': conf_inf_bits,
            'info': 'information score estimated from confidence'
        },
        'disp_inf': {
            'data': disp_inf,
            'bits': disp_inf_bits,
            'info': 'important estimated from disparity'
        }
    }

    # Have to support order = None, order as list, and order as dictionary
    order_my = ['disp_inf', 'conf_inf']
    order_dict_my = {'disp_inf': 1, 'conf_inf': 3}  # name: offset.

    save_bit_streams('some_file.tif', my_streams, None) #order_my)
    res_streams = load_bit_streams('some_file.tif')
