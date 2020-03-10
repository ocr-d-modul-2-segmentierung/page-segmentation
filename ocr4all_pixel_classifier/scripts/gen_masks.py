import argparse
import glob
import multiprocessing
import os

from ocr4all_pixel_classifier.lib.pagexml import \
    MaskGenerator, \
    MaskSetting, \
    MaskType, \
    PCGTSVersion, \
    PageXMLTypes


def main():
    parser = argparse.ArgumentParser(add_help=False)

    paths_args = parser.add_argument_group("Paths")
    paths_args.add_argument("-I", "--input-dir", type=str, required=True,
                            help="Image directory to process")
    paths_args.add_argument("-O", "--output-dir", type=str, required=True,
                            help="The output dir for the mask files")

    conf_args = parser.add_argument_group("optional arguments")
    conf_args.add_argument("-h", "--help", action="help", help="show this help message and exit")
    conf_args.add_argument("-M", "--image-map-dir", type=str, default=None,
                           help="location for writing the image map")
    conf_args.add_argument("-s", '--setting',
                           default='all_types',
                           choices=[t.value for t in MaskType],
                           help='select types of region to be included (default: %(default)s)')
    conf_args.add_argument("-e", '--mask-extension',
                           default='png',
                           choices=['png', 'dib', 'eps', 'gif', 'icns', 'ico', 'im', 'jpeg', 'msp',
                                    'pcx', 'ppm', 'sgi', 'tga', 'tiff', 'webp', 'xbm'],
                           metavar='FILE_EXT',
                           help='Filetype to use for masks (any of: %(choices)s)')
    conf_args.add_argument("-p", '--pcgts-version',
                           default='2017',
                           choices=[v.value for v in PCGTSVersion],
                           help='PCGTS Version')
    conf_args.add_argument("-w", '--line-width', type=int, default=5, help='Width of the line to be drawn')
    conf_args.add_argument("-j", "--jobs", "--threads", metavar='THREADS', dest='threads',
                           type=int, default=multiprocessing.cpu_count(),
                           help="Number of threads to use")

    args = parser.parse_args()

    pool = multiprocessing.Pool(int(args.threads))
    mask_gen = MaskGenerator(MaskSetting(MASK_TYPE=MaskType(args.setting), MASK_EXTENSION=args.mask_extension,
                                         PCGTS_VERSION=PCGTSVersion(args.pcgts_version), LINEWIDTH=args.line_width))

    files = glob.glob(args.input_dir + '/*.xml')
    from itertools import product
    pool.starmap(mask_gen.save, product(files, [args.output_dir]))

    if args.image_map_dir:
        with open(os.path.join(args.image_map_dir, 'image_map.json'), 'w') as fp:
            import json
            json.dump(PageXMLTypes.image_map(MaskType(args.setting)), fp)


if __name__ == '__main__':
    main()
