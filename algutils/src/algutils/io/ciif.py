""" Operate with ciif and tif file formats."""

import math
import re
import warnings
from ast import literal_eval
from functools import reduce
from os import path, makedirs

import numpy as np
import skimage.io as io

from ..short import unless
from .. import logger
from ..events import TimePoints

SCHEME_DEF_LOC = r'\\bkp01\tmp\Algo\test_point\test_points_scheme.xlsx'

_TM = TimePoints(True)
_log = logger('io')


def load_scheme(info_file=SCHEME_DEF_LOC, sheet_name=None) -> dict:
    """ Load test points info excel file to the dictionary DB

    :return dict:   keys - test points file names (no extension)
            values: dict with keys: ('signals', 'io', 'groups')
    """
    import pandas as pd

    def parse_struct(structure):
        struc = structure.strip().split('.')
        comp = struc[0] if struc else None  # dpe, ppr, …
        block = struc[1] if struc and (len(struc) > 1) else None  # agg, unite, opt
        view = struc[2] if struc and (len(struc) > 2) and (struc[2] in ['left', 'right']) else None  # # left|right
        tp_name = '.'.join([w for w in struc if w not in [comp, block, view]])
        return comp, block, view, tp_name

    def parse_bits(str_format, total_length):
        bits = [int(s) for s in re.split('[^\d]+', str_format.strip())]
        bits_in_group = sum(bits)

        unless(total_length % bits_in_group == 0,
               '%d signals in ciif file %s can not be grouped by %d' % (total_length, i, bits_in_group))
        bit_groups_num = total_length//bits_in_group
        return bits * bit_groups_num

    def index_groups(names, num, sep='_'):
        """ Given names in a group generate for multiple groups """
        if num == 1:
            return names
        return reduce(list.__add__, ([f + sep + str(gid) for f in names] for gid in range(num)))

    _log.debug('reading from: %s' % path.abspath(info_file))
    dfs = pd.read_excel(info_file, sheet_name=sheet_name, converters={'format': str})
    columns = ['structure', 'comments', 'name', 'alias', 'signals', 'length', 'format']  # the columns we will need

    df = pd.concat(dv[columns].dropna(axis='index', how='any', subset=['structure', 'name', 'signals', 'length', 'format'])
                   for dv in dfs.values()
                   if all(c in dv.columns for c in columns))  # type: pd.DataFrame

    df.set_index('name', inplace=True)
    # file_name: name
    df.rename_axis('file_name', inplace=True)

    df = df.reindex(columns=columns + ['groups', 'component', 'block', 'view', 'tp_name'])
    df['groups'] = np.zeros((len(df), 1))  # number of groups may be useful

    for i, r in df.iterrows():
        try:
            bits = parse_bits(r.format, int(r.length))

            signals = re.split('[^\w]+', r.signals.strip())
            sig_in_group = len(signals)
            unless(len(bits) % sig_in_group == 0,
                   '%s signals in ciif file %s can not be grouped by %s' % (r.signals.strip(), i, r.format.strip()))
            sig_groups_num = len(bits)//sig_in_group
            unless(all(bits[k*sig_in_group:(k+1)*sig_in_group] == bits[0:sig_in_group] for k in range(sig_groups_num)),
                   'format in ciif file %s is not a repeating pattern of size %d' % (i, sig_in_group))
            signals = index_groups(signals, sig_groups_num)

            comp, block, view, tp_name = parse_struct(r.structure) if r.structure else None, None, None, None
            df.loc[i, ['signals', 'format', 'groups', 'component', 'block', 'view', 'tp_name']] = \
                   signals, bits, sig_groups_num, comp, block, view, tp_name
        except Exception as e:
            raise Exception('TP %s descr problem in file %s: %s' % (i, info_file, e))

    return df.to_dict('index')


def load_signals(ciif_file, tp_scheme=SCHEME_DEF_LOC, out_df=False, show=False):
    """Load content of ciif files to dictionary according to the tp scheme.
    High level wrapper on load_ciif.

    :param ciif_file:  test points - the source of the signals
    :param tp_scheme:  as excel file or as dictionary
    :param out_df:     if True - return data as pandas DataFrame
    :param show:       show additional processing information
    :return:  dict of {field_name: image_ndarray} or additional DataFrame if requested
    """
    tps_db = load_scheme(info_file=tp_scheme) if isinstance(tp_scheme, str) else tp_scheme

    m = re.match('(?P<name>\w+)_(0|1)_00.ciif', path.basename(ciif_file)).groupdict()

    # name = '_'.join(file_name(ciif_file).split('_')[:-2])
    return load_ciif(ciif_file, data_fields=tps_db[m['name']]['signals'], out='df' if out_df else dict,
                     show=(['summary'] if show else []))


def images_to_ciif(file_path: str, images,
                   bits_num=None,
                   start_bits=None,
                   stream_names=None,
                   lane_shape=None,
                   stream_sign=None,
                   bits_write=None,
                   new_hdr=True):
    """Save array of same-sized images to ciif file io.

    :Hdr io:

        //Frame_width=(200, 0)
        //Frame_height=(26, 0)
        //Stream_name=(y10)
        //Stream_bit_num=(10)
        //Stream_sign=(unsigned)

    :Example:

        images_to_ciif('resulting.ciif', img)   # simplest form - use first max_bit(img) bits
        images_to_ciif('resulting.ciif', [img1, img2], [2, 13])   # usual form - 2 and 13 bits data
        images_to_ciif('resulting.ciif', [img, img], [6, 8], [0, 6]) # extract fields from same img

    :param file_path: Distention file path
    :param images: iterable of images of numpy.ndarray type (or a single image)
    :param bits_num: numbers of meaningful bits in the data elements (default: estimate from data)
    :param start_bits: list of (0-based) starting bits used in each image data element (default = 0)
    :param stream_names: names for the data elements (by default will be set: 'f0', 'f1' ...)
    :param lane_shape: shape of lane (default: 0,0)
    :param stream_sign: types of the data elements (by default will be set: 'unsigned')
    :param bits_write: numbers of stream bits in output ciif (default: from bits_num)
    :param new_hdr: flag for header io; True by default)
    :param pix_format: used if new_hdr==False; USER_DEFINED by default)
    """

    def data_bits(d):
        return np.ceil(np.log(1 + min(1, d.max() + 1)) / np.log(2))

    # normalize the inputs io
    if isinstance(images, np.ndarray):
        images = [images]

    if isinstance(stream_names, str):
        stream_names = [stream_names]

    if not len(images):
        warnings.warn('No images provided - exiting!')
        return

    if start_bits is None:
        start_bits = [0] * len(images)
    if not type(start_bits) in (tuple, list):
        start_bits = [start_bits] * len(images)
    if isinstance(start_bits[0], str):
        start_bits = [int(bits) for bits in start_bits]

    if stream_names is None:
        if isinstance(images[0], dict):
            stream_names = images.keys()
        else:
            stream_names = ['f%d' % i for i in range(len(images))]

    if bits_num is None:
        bits_num = [data_bits(im) for im in images]
    if not type(bits_num) in (tuple, list):
        bits_num = [bits_num] * len(images)
    if isinstance(bits_num[0], str):
        bits_num = [int(bits) for bits in bits_num]

    if bits_write is None:
        bits_write = bits_num
    if not type(bits_write) in (tuple, list):
        bits_write = [bits_write] * len(images)
    if isinstance(bits_write[0], str):
        bits_write = [int(bits) for bits in bits_write]

    if stream_sign is None:
        stream_sign = ['unsigned' for im in images]
    elif not type(stream_sign) in (tuple, list):
        stream_sign = [stream_sign] * len(images)

    if lane_shape is None:
        lane_shape = [0, 0]

    # validate inputs
    unless(len(images) == len(start_bits) == len(bits_num) == len(stream_sign) == len(bits_write),
           'mismatched arrays of images and bits')  # It shouldt work
    for im, start, bits, bits_w, signed in zip(images, start_bits, bits_num, bits_write, stream_sign):
        unless(im.dtype.kind in 'bBui', 'Only integer data types are supported')
        if signed != 'unsigned':  # Consider: add support
            unless(bits <= 32, 'Negative data supported only up to 32 bits')
        unless(start >= 0 and bits != 0 and bits_w != 0 and bits <= bits_w,
               'invalid start_bits, bits_num, bits_write: %s, %s, %s' % (start, bits, bits_w))

    images = [im.astype('uint32') if sgn == 'signed' else im for sgn, im in zip(stream_sign, images)]

    unless(all(len(im.shape) == 2 for im in images), 'images must be greyscaled')
    height, width = next(iter(images)).shape
    unless(all(im.shape == (height, width) for im in images), 'images must be of same size')

    # define formatting
    if new_hdr:
        ciif_header = ("Frame_width=({width}, {lane_shape_w})\n"
                       "Frame_height=({height}, {lane_shape_h})\n"
                       "Stream_name=({stream_names})\n"
                       "Stream_bit_num=({bits_o})\n"
                       "Stream_sign=({signed})"
                       ).format(width=width, lane_shape_w=lane_shape[0],
                                height=height, lane_shape_h=lane_shape[1],
                                stream_names=', '.join([name for name in stream_names]),
                                bits_o=', '.join(['%u' % b_num for b_num in bits_write]),
                                signed=', '.join([sign for sign in stream_sign]))
    else:
        ciif_header = ("Frame_width={width}\n"
                       "Frame_height={height}\n"
                       "Number_disparities=1\n"
                       "Number_frames=1\n"
                       "Pixel_format=({bits_o})\n"  # USER_SPECIFIC
                       "Line_direction=LeftToRight\n"
                       "Data_format=Hex"
                       ).format(width=width, height=height,
                                bits_o=', '.join(['%u' % b_num for b_num in bits_write]),)

    # line_format = "(   0,%4u,%4u) " + "_".join("%%0%ux" % np.ceil(bits / 4) for bits in bits_num)
    if len(bits_write) > 1:
        line_format = "(   0,%4u,%4u) " + "_".join(
            "%%0%lux" % np.ceil(abs(bits) / 4) for bits in bits_write)  # abs(bits) for signed numbers support
    else:
        line_format = "(   0,%4u,%4u) " + "%%0%lux" % np.ceil(abs(bits_write[0]) / 4)

        # prepare data for printing out
    masked_images = (((im.flatten() >> start) & int(2 ** abs(bits) - 1)) for  # abs(bits) for signed numbers support
                     im, start, bits in zip(images, start_bits, bits_num))

    data = np.stack((*np.where(np.ones((height, width))), *masked_images))
    data = data.astype('uint64') if np.sum(bits_write) > 32 else data.astype('uint32')
    np.savetxt(file_path, data.T, fmt=line_format, header=ciif_header, comments='//')


def _ciif_hdr_info(ciif_path) -> dict:
    """ Extract header info as dict; count number of header lines

    :param ciif_path: file_path to ciif file
    :return: hdr_attr_dict
    """
    import pandas as pd

    def extract_stream_name(header_dict):
        """ Extract stream names from from ciif hdr dict

        :param header_dict: dict read from ciif hdr dict

        :return: stream_names list
        """

        names_str = header_dict.get('Stream_name', None)
        if names_str and isinstance(names_str, str):  # new hdr io
            names = re.split(',', names_str.strip('()'))
            return [field.strip() for field in names]
        return None

    def extract_shape(header_dict):
        """ Extract Frame_height and Frame_width from ciif hdr dict

        :param header_dict: dict read from ciif hdr
        :return: (Frame_height, Frame_width) values tuple
        """

        dim_fields = ['Frame_height', 'Frame_width']
        the_shape = None
        if set(dim_fields).issubset(header_dict):
            h = literal_eval(header_dict['Frame_height'])
            w = literal_eval(header_dict['Frame_width'])

            if isinstance(h, tuple):  # new hdr io
                the_shape = (h[0], w[0])
            else:  # old hdr io
                the_shape = (h, w)
        return the_shape

    def extract_stream_bit_num(header_dict):
        """ Extract Stream_bit_num from ciif hdr dict

        :param header_dict: dict read from ciif hdr
        :return: Stream_bit_num values list
        """
        if 'Stream_bit_num' not in header_dict:
            return None

        str_bits = re.split(',', header_dict.get('Stream_bit_num').strip('()'))
        return [int(str_bits_num) for str_bits_num in str_bits]

    def extract_stream_sign(header_dict):
        """ Extract Stream_sign from ciif hdr dict

        :param header_dict: dict read from ciif hdr
        :return: Stream_sign str list
        """
        if 'Stream_sign' not in header_dict:
            return None
        signed = re.split(',', header_dict.get('Stream_sign').strip('()'))
        return [s.strip(' ') for s in signed]

    def extract_dtypes(header_dict):
        """ Extract Stream_sign from ciif hdr dict

        :param header_dict: dict read from ciif hdr
        :return: Stream_sign str list
        """

        def hdr_str_to_dtype(str_signed, str_bits_num):
            num_of_bits = int(str_bits_num)
            num_of_bits = 2 ** math.ceil(math.log2(num_of_bits))
            num_of_bits = max(num_of_bits, 8)
            if str_signed.strip() == 'unsigned':
                signed_int = 'uint'
            else:
                signed_int = 'int'
            return eval('np.{}{}'.format(signed_int, num_of_bits))

        stream_signed = extract_stream_sign(header_dict)
        bits_num = extract_stream_bit_num(header_dict)
        dtypes = None
        if stream_signed and bits_num:
            dtypes = []
            for sign, bits in zip(stream_signed, bits_num):
                dtype = hdr_str_to_dtype(sign, bits)
                dtypes.append(dtype)
        return dtypes

    try:
        header = pd.read_table(ciif_path, sep='=|//', engine='python', nrows=50,
                               usecols=range(1, 3), names=['', ' ', 'val'], index_col=0).dropna()
    except BaseException as e:
        # print('missing header in "%s"' % ciif_path)
        return {}
    else:
        hdr_dict = header['val'].to_dict()
        shape_info = extract_shape(hdr_dict)

        return {'Frame_height': shape_info[0], 'Frame_width': shape_info[1],
                'Stream_name': extract_stream_name(hdr_dict),
                'Stream_bit_num': extract_stream_bit_num(hdr_dict),
                'Stream_sign': extract_stream_sign(hdr_dict),
                'dtypes': extract_dtypes(hdr_dict),
                'hdr_rows': header.shape[0]}


def _ciif_data_info(ciif_path) -> dict:
    """ Extract header info as dict; count number of header lines

    :param ciif_path: file_path to ciif file
    :return: attr_dict extracted from data
    """

    import pandas as pd

    def default_field_names(df_readdata):
        num_found_fields = df_readdata.shape[1] - len(('frame', 'row', 'col'))
        return ['f%d' % i for i in range(num_found_fields)]

    hdr_info = _ciif_hdr_info(ciif_path)
    hdr_rows = hdr_info.get('hdr_rows', 0) if hdr_info else 0

    df = pd.read_table(ciif_path, sep=r',\s*|\(\s*|\)\s*|_', engine='python', header=None,
                       skiprows=hdr_rows, usecols=range(1, 10000), index_col=False)
    data_fields = default_field_names(df)

    position_fields = ('frame', 'row', 'col')
    df.columns = position_fields + tuple(data_fields)
    data_shape = tuple(np.array(df[['row', 'col']].apply(np.max)) + 1)

    return {'Stream_name': default_field_names(df),
            'Frame_height': data_shape[0],
            'Frame_width': data_shape[1],
            'hdr_rows': hdr_rows}


def extract_metadata(file_name) -> dict:
    """ Extract CIIF file info from ciif_hdr or from ciif_data.

    :param file_name: name of the ciif file.
    :return: dict with ciif info.
    """

    all_keys = ['Frame_width', 'Frame_height', 'Stream_name', 'Stream_bit_num', 'Stream_sign', 'dtypes', 'hdr_rows']

    res_dict = _ciif_hdr_info(file_name)
    default_keys = [k for k in all_keys if res_dict.get(k, None) is None]

    if len(default_keys) > 0:
        res_dict.update({'default': True})
        data_dict = _ciif_data_info(file_name)
        for key in default_keys:
            res_dict.update({key: data_dict.get(key, None)})
    return res_dict


def load_ciif(file_name, data_fields=None, out=dict, partial=False, show=None):
    """ Load CIIF file with multiple fields and extract them into dictionary of corresponding images

    :param file_name: name of the ciif file
    :param data_fields: list of names of the fields - defaults=['f0', 'f1', ...]
    :param out: output io [dict] | tuple | list | df (DataFrame) | 'content';
                IF out=’content’, ignore other flags and return instead of the data:
                - if new header io: list of stream names
                - if old header io: number of streams.
    :param partial: if True, falls on not complete data
    :param show: list of strings to control verbosity: ['summary', ]
    :return: dict of {field_name: image_ndarray}
    """

    import pandas as pd

    def device_data_fields(param_field_names, default_field_names):
        """
        Device data field names, using stream names read from new hdr, field names from call parameters, default field names.

        :param param_field_names: field names from call parameter;
        :param default_field_names:

        :return: Data field names to be used in result df or dict.
        """
        field_names = param_field_names or []
        if not all(re.fullmatch('\w+', field) for field in field_names):
            raise ValueError('An invalid data field in: %s' % field_names)

        if param_field_names and (len(param_field_names) != 0) and (len(default_field_names) != len(param_field_names)):
            print('Found %d fields instead of %d. Using defaults.' % (len(default_field_names), len(param_field_names)))
            return default_field_names

        return field_names or default_field_names

    if not out:
        out = dict

    _TM.point(file_name)
    meta_dict = extract_metadata(file_name)

    # -------------- Format output ---------------
    field_names = meta_dict.get('Stream_name', None)
    if out == 'content':
        return len(field_names) if meta_dict.get('default', False) else field_names

    if out in {pd.DataFrame, 'df'}:
        out = pd.DataFrame
    elif out not in {dict, tuple, list}:
        raise ValueError('Unsupported output io: ', out)

    # --------------  data_fields ---------------
    if meta_dict.get('default', False):
        data_fields = device_data_fields(param_field_names=data_fields, default_field_names=field_names)
    else:
        data_fields = field_names

    # --------------  read data ---------------
    df = pd.read_table(file_name, sep=r',\s*|\(\s*|\)\s*|_', engine='python', header=None,
                       skiprows=meta_dict.get('hdr_rows', 0), usecols=range(1, 10000), index_col=False)
    position_fields = ('frame', 'row', 'col')
    df.columns = position_fields + tuple(data_fields)

    # --------------  data_shape: check consistency with header ---------------
    data_shape = tuple(np.array(df[['row', 'col']].apply(np.max)) + 1)
    hdr_shape = (meta_dict.get('Frame_height', None), meta_dict.get('Frame_width', None))
    if not data_shape == hdr_shape:
        msg = 'Mismatch ({})!=({}) in {}'.format(data_shape, hdr_shape, file_name)
        if partial:
            warnings.warn(msg)
        else:
            raise ValueError(msg)
        # update actual number of rows:
        data_shape = int(df.shape[0] / data_shape[1]), data_shape[1]
        df = df[: (data_shape[0] * data_shape[1])]

    # --------------  dtypes ---------------
    dtypes = meta_dict.get('dtypes', [])
    stream_signed = meta_dict.get('Stream_sign', None)
    bits_num = meta_dict.get('Stream_bit_num', None)

    def hdr_str_to_dtype(str_signed, num_of_bits):
        num_of_bits = 2 ** math.ceil(math.log2(num_of_bits))
        num_of_bits = max(num_of_bits, 8)
        if str_signed.strip() == 'unsigned':
            signed_int = 'uint'
        else:
            signed_int = 'int'
        return eval('np.{}{}'.format(signed_int, num_of_bits))

    if stream_signed and bits_num:
        for sign, bits in zip(stream_signed, bits_num):
            dtype = hdr_str_to_dtype(sign, bits)
            dtypes.append(dtype)

    # _TM.point('convert hex')
    # convert data to numeric
    def str_to_dtype(str):
        num_of_bits = len(str) * 4
        num_of_bits = 2 ** math.ceil(math.log2(num_of_bits))
        num_of_bits = max(num_of_bits, 8)
        return eval('np.uint{}'.format(num_of_bits))

    for field in data_fields:
        if df[field].dtype is np.dtype('int64'):
            df.loc[:, field] = df[field].apply(str)

        if meta_dict.get('default', False):  # is_old_format
            dtype = str_to_dtype(df[field].iloc[0])
            dtypes.append(dtype)

        df.loc[:, field] = df[field].apply(int, args=(16,))

    # --------------  show summary ---------------
    _TM.point(measure_from=file_name)
    if show and 'summary' in show:
        print(df[[*data_fields]].describe())

    # --------------  out ---------------
    # special request to return only data frame
    if out in {'df', pd.DataFrame}:
        return df[[*data_fields]]

    images = {field: df[field].values.reshape(data_shape)[df.row, df.col].reshape(data_shape).astype(image_type)
              for field, image_type in zip(data_fields, dtypes)}
    return images if out is dict else out(images[field] for field in data_fields)


def save_tiffs(images, path_prefix, compress=0):
    """ Save dictionary of images into multiple tiff files as type 'uint16'
    using given path_prefix as prefix and keys as suffix

    :param images: dictionary of 1:3 images, of view {field:image}, where image is ndarray
    :param path_prefix: output_folder [and output files basename without suffix and extension]
    :param compress: not used for now
    :usage: process_ciif_to_tiffs(images,args.prefix)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for field in images:
            io.imsave('%s_%s.tif' % (path_prefix, field), images[field].astype('uint16'),
                      compress=compress)


def parse_command_line():
    import argparse

    parser = argparse.ArgumentParser(prog='ciif')

    x_group = parser.add_mutually_exclusive_group(required=True)
    x_group.add_argument('-ciif', action='store_true',
                         help='Input CIIF mode. Mutually exclusive with --img')
    x_group.add_argument('-img', '--image', action='store_true',
                         help='Input images mode - provide image files accordingly')

    # parser = parser.add_argument_group('calling conventions')
    parser.add_argument('input_files', nargs='+', type=str,
                        help='single *.ciif file in -ciif mode; '
                             '1+ *.ciif files in -ciif -c mode;'
                             '1+ image files in -img mode')
    parser.add_argument('-o', '--output_folder', type=str, nargs='?', default=[],
                        help='Default: the same as input files folder')
    parser.add_argument('-p', '--prefix', type=str,
                        help='Prefix for output file names. '
                             '-ciif mode: --signals are used as suffix.'
                             '    Default: use prefix extracted from the input name.'
                             '-ciif -c mode: full name of the output file.'
                             '-img mode: name of the output file.'
                             '    May contain folder name if -o not defined.'
                             '    default: _ joined names of the input files.')

    # input param in -ciif -c mode: concat option
    parser.add_argument('-c', '--concat', type=int, nargs='+',
                        help='(-ciif -c mode only) Number of streams to use from each input ciif file')
    parser.add_argument('-w', '--width', type=int,
                        help='(-ciif -c mode only) Width of concatenated result stream')

    # input params in -img mode and -ciif -c mode
    parser.add_argument('-f', '--first_bit', type=int, nargs='+',
                        help='(-img mode; -ciif -c mode) Starting bit (zero based) for each input signal')
    parser.add_argument('-b', '--bits_read', type=int, nargs='+',
                        help='(-img mode; -ciif -c mode) Number of bits to read from each tiff file')

    # input params in -img mode
    parser.add_argument('-g', '--grey', action='store_true',
                        help='(-img mode only) If set color inputs will be excepted and converted to grey')

    # output params in -img mode
    parser.add_argument('-ho', '--header_old', action='store_true',
                        help='(-img mode only) Ciif header format to be written: new header by default')
    parser.add_argument('-bw', '--bits_write', type=int, nargs='+',
                        help='(-img mode only) Number of bits in each signal')
    parser.add_argument('-ft', '--formats', type=str, nargs='+',
                        help='(-img mode only) Format "signed"/"unsigned" for each signal')

    f_group = parser.add_mutually_exclusive_group()
    f_group.add_argument('-s', '--signals', nargs='+',
                         help='(-img mode; -ciif -c mode) list names of all the signals in the ciif or in repeating group'
                              'Default values: "s1", "s2", ...')
    f_group.add_argument('-db', '--formats_db',
                         help='Excel file describing io of test points')

    specific_verb_terms = ['time', 'summary', 'db']
    general_verb_terms = ['0', 'none', 'all']
    parser.add_argument('-v', '--verbose', type=str, nargs='+', default=[],
                        choices=specific_verb_terms + general_verb_terms,
                        help='Add information type to hear about')

    class Args:
        def __init__(self):  # just to please PyCharm ;-)
            self.ciif = None
            self.image = None
            self.input_files = None
            self.input_folder = None
            self.output_folder = None
            self.prefix = None
            self.ciif_file = None
            self.grey = None
            self.first_bit = None
            self.bits_read = None
            self.header_old = None
            self.bits_write = None
            self.formats_db = None
            self.concat = None
            self.width = None
            self.signals = []
            self.formats = []
            self.verbose = []

        def devise_out_folder(self):
            if not self.output_folder:
                input_folder = path.dirname(self.input_files[0])
                if path.commonpath(self.input_files) != input_folder:
                    warnings.warn('Using first input file folder for output: %s' % input_folder)
                self.output_folder = input_folder
            # Prepare output dir
            if not path.exists(self.output_folder):
                makedirs(self.output_folder)
            return self.output_folder

        def devise_out_ciif_path(self):
            if self.prefix:  # output folder may be part of the prefix argument:
                if path.split(self.prefix)[0]:
                    unless(not self.output_folder, 'output folder provided also with prefix')
                    self.output_folder = path.dirname(self.prefix)
                    self.ciif_file = path.basename(self.prefix)
                else:
                    self.ciif_file = self.prefix
            else:  # if output ciif image name not provided combine it from inputs names
                # eliminate common prefix from file names to leave only meaningful signal tag
                file_names = [path.split(inp)[1] for inp in self.input_files]  # throw paths away
                if len(file_names) > 1:
                    prefix = path.commonprefix(file_names)
                    signals = [path.splitext(name[len(prefix):])[0] for name in file_names]
                    self.ciif_file = prefix + '_' + '_'.join(signals) + '.ciif'
                else:
                    self.ciif_file = path.splitext(file_names[0])[0] + '.ciif'

            self.devise_out_folder()
            self.ciif_file = path.join(self.output_folder, self.ciif_file)

    arguments = Args()
    parser.parse_args(namespace=arguments)

    if 'all' in arguments.verbose:
        arguments.verbose += specific_verb_terms

    return arguments


def _concat_im_fields(images, signals, first_bits, bits_reads):
    """Concat fields of images

    :param images: dict of {signal : im} pairs
    :param signals: list of signals from dict.keys() to be used for result
    :param first_bits: list of first_bits per signal from signals to be used for result
    :param bits_reads list of bits_reads per signal from signals to be used for result
    :return: uint64 3ith MSB from signals[0] and LSB from signals[-1]
    """
    if type(images) is dict:
        init_base = [np.zeros_like(im) for _, im in images.items()][0]
    elif type(images) is list:
        init_base = np.zeros_like(images[0])
    else:
        init_base = np.zeros_like(images)

    res = init_base.flatten().astype('uint64')
    shift = 0
    for signal, first_bit, bits in zip(signals, first_bits, bits_reads):
        im = ((images[signal].flatten() >> first_bit) & int(2 ** abs(bits) - 1))
        # old: signals_list[0] will be LSB in output: res += im << shift; shift += bits

        # new: signals_list[0] will be MSB in output
        shift += bits
        res = (res << bits) + im
    return res.reshape(init_base.shape)


def concat_ciifs_to_ciif(args):
    """Treat ciif(s) -> ciif concat mode of the command line processing.

    :param args: command line arguments
    :return:
    """
    ciif_num = len(args.concat)
    if len(args.input_files) != ciif_num:
        warnings.warn('-c arg needs the same num of vals as input files num')
        return

    im_dicts_list = [load_ciif(f) for f in args.input_files]

    start_pos = 0
    res = []
    bits_reads = []
    for im_dict, f_num in zip(im_dicts_list, args.concat):
        num = int(f_num)
        res.append(_concat_im_fields(images=im_dict,
                                     signals=args.signals[start_pos:start_pos + num],
                                     first_bits=args.first_bit[start_pos:start_pos + num],
                                     bits_reads=args.bits_read[start_pos:start_pos + num]))
        read = sum(args.bits_read[start_pos:start_pos + num])
        bits_reads.append(read)
        start_pos += num

    res = _concat_im_fields(images=res, signals=range(ciif_num),
                            first_bits=[0] * ciif_num, bits_reads=bits_reads)
    images_to_ciif(args.prefix, res, bits_num=sum(bits_reads), bits_write=args.width, new_hdr=False)
    pass


def process_ciif_to_tiff(args):
    """Treat ciif -> tiff mode of the command line processing

    :param args: command line arguments
    :return:
    """
    unless(len(args.input_files) == 1, 'ciif source requires single input file name')
    args.ciif_file = args.input_files[0]
    args.devise_out_folder()

    def file_name_part(filename):
        return path.splitext(path.basename(filename))[0]

    if not args.prefix:  # extract prefix from the ciif file: /ddd/aa_bb_ok.ext -> aa
        args.prefix = file_name_part(args.ciif_file).split('_')[0]
    args.prefix = path.join(args.output_folder, args.prefix)

    # _TM.point('load DB')
    if args.formats_db:  # if excel Test Points Data Base is provided get signals from there
        tp_db = load_scheme(args.formats_db)
        if 'db' in args.verbose:
            print('Test Points formats found in the DB:')
            [print('\t' + tp) for tp in tp_db]

        try:
            args.signals = tp_db[file_name_part(args.ciif_file)]['signals']
        except KeyError:
            warnings.warn('No io info from file "%s" - using defaults' % args.format[0])
        else:
            print('Successfully extracted Test Points data io.')

    # _TM.point('load ciif complete', measure_from='load DB')
    images = load_ciif(args.ciif_file, args.signals, show=args.verbose)

    # _TM.point('save tiffs', measure_from='load ciif complete')
    save_tiffs(images, args.prefix)
    # _TM.point(measure_from='save tiffs')


def rgb2gray(img):
    from skimage.color import rgb2gray
    return (rgb2gray(img) * 0xFF).astype('uint8')


def process_tiff_to_ciif(args):
    """Treat tiff -> ciif mode of the command line processing

    :param args: command line arguments
    :return:
    """
    args.devise_out_ciif_path()
    images = [io.imread(file) for file in args.input_files]

    if args.grey:
        images = [im[:, :, 1] if im.ndim == 3 and im.shape[2] in [3, 4] else im for im in images]
        # images = [rgb2gray(im) for im in images]

    if not args.bits_write:
        args.bits_write = args.bits_read

    images_to_ciif(args.ciif_file,
                   images=images,
                   bits_num=args.bits_read,
                   start_bits=args.first_bit,
                   stream_names=args.signals,
                   new_hdr = (not args.header_old),
                   bits_write=args.bits_write)

# OLDER VERSION?
# def cif_to_tif(file):
#     """
#     Convert images in a single ciif file into multiple separate tif files
#     named `{original}_{internal_item_name}.tif`
#
#     :param file: The source
#     :return: number of files created
#     """
#     import warnings
#     from skimage.io import imsave
#     mport ciff as cif
#     ext = '.ciif'
#
#     try:
#         file = Path(file)
#         if not file.is_file():
#             raise FileExistsError(file)
#         if not file.suffix.lower() is ext:
#             raise TypeError(f'{file} not {ext} file')
#
#         ciif_data = cif.load_ciif(file)
#         for key, image in ciif_data.items():
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')
#                 img_name = f'{file[:-len(ext)]}_{key}.tif'
#                 imsave(img_name, image)
#                 print(f'Created: {img_name}')
#     except Exception as e:
#         print('Failed to parse:', file)
#         print(e)
#         return 0
#     return len(ciif_data)


if __name__ == '__main__':
    cmd_args = parse_command_line()
    _TM.enable = 'time' in cmd_args.verbose

    if cmd_args.ciif:
        if cmd_args.width:
            concat_ciifs_to_ciif(cmd_args)
        else:
            process_ciif_to_tiff(cmd_args)
    elif cmd_args.image:
        process_tiff_to_ciif(cmd_args)
