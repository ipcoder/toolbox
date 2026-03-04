from PIL import Image
from PIL.TiffTags import TAGS_V2


def compression_types():
    # from https://github.com/libsdl-org/libtiff/blob/master/libtiff/tiff.h
    ss = """
    #define COMPRESSION_NONE 1            /* dump mode */
    #define COMPRESSION_CCITTRLE 2        /* CCITT modified Huffman RLE */
    #define COMPRESSION_CCITTFAX3 3       /* CCITT Group 3 fax encoding */
    #define COMPRESSION_CCITT_T4 3        /* CCITT T.4 (TIFF 6 name) */
    #define COMPRESSION_CCITTFAX4 4       /* CCITT Group 4 fax encoding */
    #define COMPRESSION_CCITT_T6 4        /* CCITT T.6 (TIFF 6 name) */
    #define COMPRESSION_LZW 5             /* Lempel-Ziv  & Welch */
    #define COMPRESSION_OJPEG 6           /* !6.0 JPEG */
    #define COMPRESSION_JPEG 7            /* %JPEG DCT compression */
    #define COMPRESSION_T85 9             /* !TIFF/FX T.85 JBIG compression */
    #define COMPRESSION_T43 10            /* !TIFF/FX T.43 colour by layered JBIG compression */
    #define COMPRESSION_NEXT 32766        /* NeXT 2-bit RLE */
    #define COMPRESSION_CCITTRLEW 32771   /* #1 w/ word alignment */
    #define COMPRESSION_PACKBITS 32773    /* Macintosh RLE */
    #define COMPRESSION_THUNDERSCAN 32809 /* ThunderScan RLE */
    /* codes 32895-32898 are reserved for ANSI IT8 TIFF/IT <dkelly@apago.com) */
    #define COMPRESSION_IT8CTPAD 32895 /* IT8 CT w/padding */
    #define COMPRESSION_IT8LW 32896    /* IT8 Linework RLE */
    #define COMPRESSION_IT8MP 32897    /* IT8 Monochrome picture */
    #define COMPRESSION_IT8BL 32898    /* IT8 Binary line art */
    /* compression codes 32908-32911 are reserved for Pixar */
    #define COMPRESSION_PIXARFILM 32908 /* Pixar companded 10bit LZW */
    #define COMPRESSION_PIXARLOG 32909  /* Pixar companded 11bit ZIP */
    #define COMPRESSION_DEFLATE 32946   /* Deflate compression, legacy tag */
    #define COMPRESSION_ADOBE_DEFLATE 8 /* Deflate compression, as recognized by Adobe */
    /* compression code 32947 is reserved for Oceana Matrix <dev@oceana.com> */
    #define COMPRESSION_DCS 32947      /* Kodak DCS encoding */
    #define COMPRESSION_JBIG 34661     /* ISO JBIG */
    #define COMPRESSION_SGILOG 34676   /* SGI Log Luminance RLE */
    #define COMPRESSION_SGILOG24 34677 /* SGI Log 24-bit packed */
    #define COMPRESSION_JP2000 34712   /* Leadtools JPEG2000 */
    #define COMPRESSION_LERC 34887     /* ESRI Lerc codec: https://github.com/Esri/lerc */
    /* compression codes 34887-34889 are reserved for ESRI */
    #define COMPRESSION_LZMA 34925             /* LZMA2 */
    #define COMPRESSION_ZSTD 50000             /* ZSTD: WARNING not registered in Adobe-maintained registry */
    #define COMPRESSION_WEBP 50001             /* WEBP: WARNING not registered in Adobe-maintained registry */
    #define COMPRESSION_JXL 50002              /* JPEGXL: WARNING not registered in Adobe-maintained registry */
    """.split('\n')
    import re
    rex = re.compile(r"\s*#define\sCOMPRESSION_(\w+)\s(\d+)\s+.*")
    return {m.group(2): m.group(1) for m in map(rex.match, ss) if m}


def compression_info():
    ss = """
    #define TIFFTAG_JPEGQUALITY 65537    /* Compression quality level */
    /* Note: quality level is on the IJG 0-100 scale.  Default value is 75 */
    #define TIFFTAG_JPEGCOLORMODE 65538  /* Auto RGB<=>YCbCr convert? */
    #define JPEGCOLORMODE_RAW 0x0000     /* no conversion (default) */
    #define JPEGCOLORMODE_RGB 0x0001     /* do auto conversion */
    #define TIFFTAG_JPEGTABLESMODE 65539 /* What to put in JPEGTables */
    #define JPEGTABLESMODE_QUANT 0x0001  /* include quantization tbls */
    #define JPEGTABLESMODE_HUFF 0x0002   /* include Huffman tbls */
    /* Note: default is JPEGTABLESMODE_QUANT | JPEGTABLESMODE_HUFF */
    #define TIFFTAG_FAXFILLFUNC 65540     /* G3/G4 fill function */
    #define TIFFTAG_PIXARLOGDATAFMT 65549 /* PixarLogCodec I/O data sz */
    #define PIXARLOGDATAFMT_8BIT 0        /* regular u_char samples */
    #define PIXARLOGDATAFMT_8BITABGR 1    /* ABGR-order u_chars */
    #define PIXARLOGDATAFMT_11BITLOG 2    /* 11-bit log-encoded (raw) */
    #define PIXARLOGDATAFMT_12BITPICIO 3  /* as per PICIO (1.0==2048) */
    #define PIXARLOGDATAFMT_16BIT 4       /* signed short samples */
    #define PIXARLOGDATAFMT_FLOAT 5       /* IEEE float samples */
    /* 65550-65556 are allocated to Oceana Matrix <dev@oceana.com> */
    #define TIFFTAG_DCSIMAGERTYPE 65550     /* imager model & filter */
    #define DCSIMAGERMODEL_M3 0             /* M3 chip (1280 x 1024) */
    #define DCSIMAGERMODEL_M5 1             /* M5 chip (1536 x 1024) */
    #define DCSIMAGERMODEL_M6 2             /* M6 chip (3072 x 2048) */
    #define DCSIMAGERFILTER_IR 0            /* infrared filter */
    #define DCSIMAGERFILTER_MONO 1          /* monochrome filter */
    #define DCSIMAGERFILTER_CFA 2           /* color filter array */
    #define DCSIMAGERFILTER_OTHER 3         /* other filter */
    #define TIFFTAG_DCSINTERPMODE 65551     /* interpolation mode */
    #define DCSINTERPMODE_NORMAL 0x0        /* whole image, default */
    #define DCSINTERPMODE_PREVIEW 0x1       /* preview of image (384x256) */
    #define TIFFTAG_DCSBALANCEARRAY 65552   /* color balance values */
    #define TIFFTAG_DCSCORRECTMATRIX 65553  /* color correction values */
    #define TIFFTAG_DCSGAMMA 65554          /* gamma value */
    #define TIFFTAG_DCSTOESHOULDERPTS 65555 /* toe & shoulder points */
    #define TIFFTAG_DCSCALIBRATIONFD 65556  /* calibration file desc */
    /* Note: quality level is on the ZLIB 1-9 scale. Default value is -1 */
    #define TIFFTAG_ZIPQUALITY 65557      /* compression quality level */
    #define TIFFTAG_PIXARLOGQUALITY 65558 /* PixarLog uses same scale */
    /* 65559 is allocated to Oceana Matrix <dev@oceana.com> */
    #define TIFFTAG_DCSCLIPRECTANGLE 65559 /* area of image to acquire */
    #define TIFFTAG_SGILOGDATAFMT 65560    /* SGILog user data format */
    #define SGILOGDATAFMT_FLOAT 0          /* IEEE float samples */
    #define SGILOGDATAFMT_16BIT 1          /* 16-bit samples */
    #define SGILOGDATAFMT_RAW 2            /* uninterpreted data */
    #define SGILOGDATAFMT_8BIT 3           /* 8-bit RGB monitor values */
    #define TIFFTAG_SGILOGENCODE 65561     /* SGILog data encoding control*/
    #define SGILOGENCODE_NODITHER 0        /* do not dither encoded values*/
    #define SGILOGENCODE_RANDITHER 1       /* randomly dither encd values */
    #define TIFFTAG_LZMAPRESET 65562       /* LZMA2 preset (compression level) */
    #define TIFFTAG_PERSAMPLE 65563        /* interface for per sample tags */
    #define PERSAMPLE_MERGED 0             /* present as a single value */
    #define PERSAMPLE_MULTI 1              /* present as multiple values */
    #define TIFFTAG_ZSTD_LEVEL 65564       /* ZSTD compression level */
    #define TIFFTAG_LERC_VERSION 65565     /* LERC version */
    #define LERC_VERSION_2_4 4
    #define TIFFTAG_LERC_ADD_COMPRESSION 65566 /* LERC additional compression */
    #define LERC_ADD_COMPRESSION_NONE 0
    #define LERC_ADD_COMPRESSION_DEFLATE 1
    #define LERC_ADD_COMPRESSION_ZSTD 2
    #define TIFFTAG_LERC_MAXZERROR 65567   /* LERC maximum error */
    #define TIFFTAG_WEBP_LEVEL 65568       /* WebP compression level */
    #define TIFFTAG_WEBP_LOSSLESS 65569    /* WebP lossless/lossy */
    #define TIFFTAG_WEBP_LOSSLESS_EXACT 65571  /* WebP lossless exact mode. Set-only mode. Default is 1. Can be set to 0 to increase compression rate, but R,G,B in areas where alpha = 0 will not be preserved */
    """.split('\n')
    import re
    rex = re.compile(r"#define\sTIFFTAG_(\w+)\s(\d+)\s+.*")
    return {m.group(2): m.group(1) for m in map(rex.match, ss) if m}


def standard_tags():
    return {tag.name: tag.value for tag in TAGS_V2.values()}


NAME_TO_TAG = {
    'NewSubfileType': 254,
    'SubfileType': 255,
    'ImageWidth': 256,
    'ImageLength': 257,
    'BitsPerSample': 258,
    'Compression': 259,
    'PhotometricInterpretation': 262,
    'Threshholding': 263,
    'CellWidth': 264,
    'CellLength': 265,
    'FillOrder': 266,
    'DocumentName': 269,
    'ImageDescription': 270,
    'Make': 271,
    'Model': 272,
    'StripOffsets': 273,
    'Orientation': 274,
    'SamplesPerPixel': 277,
    'RowsPerStrip': 278,
    'StripByteCounts': 279,
    'MinSampleValue': 280,
    'MaxSampleValue': 281,
    'XResolution': 282,
    'YResolution': 283,
    'PlanarConfiguration': 284,
    'PageName': 285,
    'XPosition': 286,
    'YPosition': 287,
    'FreeOffsets': 288,
    'FreeByteCounts': 289,
    'GrayResponseUnit': 290,
    'GrayResponseCurve': 291,
    'T4Options': 292,
    'T6Options': 293,
    'ResolutionUnit': 296,
    'PageNumber': 297,
    'TransferFunction': 301,
    'Software': 305,
    'DateTime': 306,
    'Artist': 315,
    'HostComputer': 316,
    'Predictor': 317,
    'WhitePoint': 318,
    'PrimaryChromaticities': 319,
    'ColorMap': 320,
    'HalftoneHints': 321,
    'TileWidth': 322,
    'TileLength': 323,
    'TileOffsets': 324,
    'TileByteCounts': 325,
    'SubIFDs': 330,
    'InkSet': 332,
    'InkNames': 333,
    'NumberOfInks': 334,
    'DotRange': 336,
    'TargetPrinter': 337,
    'ExtraSamples': 338,
    'SampleFormat': 339,
    'SMinSampleValue': 340,
    'SMaxSampleValue': 341,
    'TransferRange': 342,
    'JPEGTables': 347,
    'JPEGProc': 512,
    'JPEGInterchangeFormat': 513,
    'JPEGInterchangeFormatLength': 514,
    'JPEGRestartInterval': 515,
    'JPEGLosslessPredictors': 517,
    'JPEGPointTransforms': 518,
    'JPEGQTables': 519,
    'JPEGDCTables': 520,
    'JPEGACTables': 521,
    'YCbCrCoefficients': 529,
    'YCbCrSubSampling': 530,
    'YCbCrPositioning': 531,
    'ReferenceBlackWhite': 532,
    'XMP': 700,
    'Copyright': 33432,
    'IptcNaaInfo': 33723,
    'PhotoshopInfo': 34377,
    'ExifIFD': 34665,
    'ICCProfile': 34675,
    'GPSInfoIFD': 34853,
    'ExifVersion': 36864,
    'InteroperabilityIFD': 40965,
    'CFAPattern': 41730,
    'MPFVersion': 45056,
    'NumberOfImages': 45057,
    'MPEntry': 45058,
    'ImageUIDList': 45059,
    'TotalFrames': 45060,
    'MPIndividualNum': 45313,
    'PanOrientation': 45569,
    'PanOverlap_H': 45570,
    'PanOverlap_V': 45571,
    'BaseViewpointNum': 45572,
    'ConvergenceAngle': 45573,
    'BaselineLength': 45574,
    'VerticalDivergence': 45575,
    'AxisDistance_X': 45576,
    'AxisDistance_Y': 45577,
    'AxisDistance_Z': 45578,
    'YawAngle': 45579,
    'PitchAngle': 45580,
    'RollAngle': 45581,
    'FlashPixVersion': 40960,
    'MakerNoteSafety': 50741,
    'BestQualityScale': 50780,
    'ImageJMetaDataByteCounts': 50838,
    'ImageJMetaData': 50839}

COMPRESSION_TAGS = {
    1: 'NONE',
    2: 'CCITTRLE',
    3: 'CCITT_T4',
    4: 'CCITT_T6',
    5: 'LZW',
    6: 'OJPEG',
    7: 'JPEG',
    8: 'ADOBE_DEFLATE',
    9: 'T85',
    10: 'T43',
    32766: 'NEXT',
    32771: 'CCITTRLEW',
    32773: 'PACKBITS',
    32809: 'THUNDERSCAN',
    32895: 'IT8CTPAD',
    32896: 'IT8LW',
    32897: 'IT8MP',
    32898: 'IT8BL',
    32908: 'PIXARFILM',
    32909: 'PIXARLOG',
    32946: 'DEFLATE',
    32947: 'DCS',
    34661: 'JBIG',
    34676: 'SGILOG',
    34677: 'SGILOG24',
    34712: 'JP2000',
    34887: 'LERC',
    34925: 'LZMA',
    50000: 'ZSTD',
    50001: 'WEBP',
    50002: 'JXL'
}

COMPRESSION_INFO = {
    '65537': 'JPEGQUALITY',
    '65538': 'JPEGCOLORMODE',
    '65539': 'JPEGTABLESMODE',
    '65540': 'FAXFILLFUNC',
    '65549': 'PIXARLOGDATAFMT',
    '65550': 'DCSIMAGERTYPE',
    '65551': 'DCSINTERPMODE',
    '65552': 'DCSBALANCEARRAY',
    '65553': 'DCSCORRECTMATRIX',
    '65554': 'DCSGAMMA',
    '65555': 'DCSTOESHOULDERPTS',
    '65556': 'DCSCALIBRATIONFD',
    '65557': 'ZIPQUALITY',
    '65558': 'PIXARLOGQUALITY',
    '65559': 'DCSCLIPRECTANGLE',
    '65560': 'SGILOGDATAFMT',
    '65561': 'SGILOGENCODE',
    '65562': 'LZMAPRESET',
    '65563': 'PERSAMPLE',
    '65564': 'ZSTD_LEVEL',
    '65565': 'LERC_VERSION',
    '65566': 'LERC_ADD_COMPRESSION',
    '65567': 'LERC_MAXZERROR',
    '65568': 'WEBP_LEVEL',
    '65569': 'WEBP_LOSSLESS',
    '65571': 'WEBP_LOSSLESS_EXACT'
}


def read_image_tags(file, tags=(
        'ImageWidth',
        'ImageLength',
        'BitsPerSample',
        'SamplesPerPixel',
        'Compression',
        'TileWidth',
        'TileLength'), *more_tags):

    tags = [tag if isinstance(tag, int) else NAME_TO_TAG.get(tag)
            for tag in [*(tags or []), *more_tags]]

    with Image.open(file) as img:
        img_tags = img.tag_v2

        if tags:
            d = {TAGS_V2[tag].name: img_tags.get(tag) for tag in tags}
        else:
            d = {TAGS_V2[tag].name: val for tag, val in img_tags.items()}

        if compr := d.get('Compression', None):
            d['Compression'] = COMPRESSION_TAGS[compr]
        return d
