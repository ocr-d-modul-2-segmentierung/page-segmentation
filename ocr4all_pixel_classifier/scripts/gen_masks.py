import argparse
import enum
import glob
import multiprocessing
import os
import xml.etree.ElementTree as ElementTree
from typing import NamedTuple, List, Tuple, Optional

from PIL import Image, ImageDraw


class MaskType(enum.Enum):
    ALLTYPES = 'all_types'
    TEXT_NONTEXT = 'text_nontext'
    BASE_LINE = 'baseline'
    TEXT_LINE = 'textline'


class PCGTSVersion(enum.Enum):
    PCGTS2017 = '2017'
    PCGTS2013 = '2013'

    def get_namespace(self):
        return {
            PCGTSVersion.PCGTS2017: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15',
            PCGTSVersion.PCGTS2013: 'https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
        }[self]


class MaskSetting(NamedTuple):
    MASK_EXTENSION: str = 'png'
    MASK_TYPE: MaskType = MaskType.ALLTYPES
    PCGTS_VERSION: PCGTSVersion = PCGTSVersion.PCGTS2017
    LINEWIDTH: int = 5


class PageXMLTypes(enum.Enum):
    PARAGRAPH = ('paragraph', (255, 0, 0))
    IMAGE = ('ImageRegion', (0, 255, 0))
    GRAPHIC = ('GraphicRegion', (0, 255, 0))
    HEADING = ('heading', (0, 0, 255))
    HEADER = ('header', (0, 255, 255))
    CATCH_WORD = ('catch-word', (255, 255, 0))
    PAGE_NUMBER = ('page-number', (255, 0, 255))
    SIGNATURE_MARK = ('signature-mark', (128, 0, 128))
    MARGINALIA = ('marginalia', (128, 128, 0))
    OTHER = ('other', (0, 128, 128))
    DROP_CAPITAL = ('drop-capital', (255, 128, 0))
    FLOATING = ('floating', (255, 0, 128))
    CAPTION = ('caption', (128, 255, 0))
    ENDNOTE = ('endnote', (0, 255, 128))

    def __new__(cls, value, color):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.color = color
        obj.label = value
        return obj

    def color_text_nontext(self):
        return (0, 255, 0) if self is PageXMLTypes.IMAGE or self is PageXMLTypes.GraphicRegion else (255, 0, 0)

    @classmethod
    def image_map(cls, mask_type: MaskType):
        types = {
            MaskType.ALLTYPES: PageXMLTypes,
            MaskType.TEXT_NONTEXT: [PageXMLTypes.PARAGRAPH, PageXMLTypes.IMAGE],
            MaskType.TEXT_LINE: [PageXMLTypes.PARAGRAPH],
            MaskType.BASE_LINE: [PageXMLTypes.PARAGRAPH],
        }[mask_type]

        map = {
            str(xmltype.color): (i + 1, xmltype.label)
            for (i, xmltype) in enumerate(types)
        }
        map['(0, 0, 0)'] = (0, 'background')
        return map


class RegionType(NamedTuple):
    polygon: List[Tuple[int, int]]
    type: PageXMLTypes


class PageRegions(NamedTuple):
    image_size: Tuple[int, int]
    xml_regions: List[RegionType]
    filename: str


class MaskGenerator:

    def __init__(self, settings: MaskSetting):
        self.settings = settings

    def save(self, file, output_dir):
        a = get_xml_regions(file, self.settings)
        mask_pil = page_region_to_mask(a, self.settings)
        filename_wo_ext = os.path.splitext(os.path.basename(a.filename))[0]
        os.makedirs(output_dir, exist_ok=True)
        mask_pil.save(os.path.join(output_dir, filename_wo_ext + '.mask.' + self.settings.MASK_EXTENSION))


def string_to_lp(points: str):
    lp_points: List[Tuple[int, int]] = []
    if points is not None:
        for point in points.split(' '):
            x, y = point.split(',')
            lp_points.append((int(x), int(y)))
    return lp_points


def coords_for_element(element, namespaces, tag: str = 'pcgts:Coords') -> Optional[RegionType]:
    coords = element.find(tag, namespaces)
    if coords is not None:
        polyline = string_to_lp(coords.get('points'))
        type = element.get('type') if 'type' in element.attrib else 'paragraph'
        return RegionType(polygon=polyline, type=PageXMLTypes(type))
    else:
        return None


def nested_child_regions(child, namespaces, tag: str = 'pcgts:Coords') -> List[RegionType]:
    return [
        coords_for_element(textline, namespaces, tag)
        for textline in child.findall('pcgts:TextLine', namespaces)
        if textline is not None
    ]


def get_xml_regions(xml_file, setting: MaskSetting) -> PageRegions:
    namespaces = {'pcgts': setting.PCGTS_VERSION.get_namespace()}
    root = ElementTree.parse(xml_file).getroot()
    region_by_types = []

    for name, value in namespaces.items():
        ElementTree.register_namespace(name, value)

    for child in root.findall('.//pcgts:TextRegion', namespaces):
        if setting.MASK_TYPE == setting.MASK_TYPE.TEXT_NONTEXT or setting.MASK_TYPE == setting.MASK_TYPE.ALLTYPES:
            region_by_types.append(coords_for_element(child, namespaces))
        elif setting.MASK_TYPE == setting.MASK_TYPE.TEXT_LINE:
            region_by_types += nested_child_regions(child, namespaces, 'pcgts:Coords')
        elif setting.MASK_TYPE == setting.MASK_TYPE.BASE_LINE:
            region_by_types += nested_child_regions(child, namespaces, 'pcgts:Baseline')

    for child in root.findall('.//pcgts:ImageRegion', namespaces):
        if setting.MASK_TYPE == setting.MASK_TYPE.TEXT_NONTEXT or setting.MASK_TYPE == setting.MASK_TYPE.ALLTYPES:
            coords = child.find('pcgts:Coords', namespaces)
            if coords:
                polyline = string_to_lp(coords.get('points'))
                region_by_types.append(RegionType(polygon=polyline, type=PageXMLTypes('ImageRegion')))

    page = root.find('.//pcgts:Page', namespaces)
    page_height = page.get('imageHeight')
    page_width = page.get('imageWidth')

    f_name = resolve_relative_path(xml_file, page.get('imageFilename'))
    return PageRegions(image_size=(int(page_height), int(page_width)), xml_regions=region_by_types, filename=f_name)


def resolve_relative_path(base, path):
    """
    Resolve a path relative to a given base
    :param base: the base path. if it is an existing file, only the directory part will be used.
    :param path: the path to resolve.
    :return: the resolved path, or the input path if it was absolute.
    """
    from os.path import normpath, join, dirname, isabs, isfile
    if isabs(path):
        return path
    else:
        if isfile(base):
            base = dirname(base)
        return normpath(join(base, path))


def page_region_to_mask(page_region: PageRegions, setting: MaskSetting) -> Image:
    height, width = page_region.image_size
    pil_image = Image.new('RGB', (width, height), (255, 255, 255))
    for x in page_region.xml_regions:
        if setting.MASK_TYPE is MaskType.ALLTYPES:
            if len(x.polygon) > 2:
                ImageDraw.Draw(pil_image).polygon(x.polygon, outline=x.type.color, fill=x.type.color)
        elif setting.MASK_TYPE is MaskType.TEXT_NONTEXT:
            if len(x.polygon) > 2:
                ImageDraw.Draw(pil_image).polygon(x.polygon, outline=x.type.color_text_nontext(),
                                                  fill=x.type.color_text_nontext())
        elif setting.MASK_TYPE is MaskType.BASE_LINE:
            ImageDraw.Draw(pil_image).line(x.polygon, fill=x.type.color_text_nontext(), width=5)
        elif setting.MASK_TYPE is MaskType.TEXT_LINE:
            ImageDraw.Draw(pil_image).polygon(x.polygon, outline=x.type.color_text_nontext(),
                                              fill=x.type.color_text_nontext())

    return pil_image


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
                           choices=['all_types', 'text_nontext', 'baseline', 'textline'],
                           help='select types of region to be included (default: %(default)s)')
    conf_args.add_argument("-e", '--mask-extension',
                           default='png',
                           choices=['png', 'dib', 'eps', 'gif', 'icns', 'ico', 'im', 'jpeg', 'msp',
                                    'pcx', 'ppm', 'sgi', 'tga', 'tiff', 'webp', 'xbm'],
                           metavar='FILE_EXT',
                           help='Filetype to use for masks (any of: %(choices)s)')
    conf_args.add_argument("-p", '--pcgts-version',
                           default='2017',
                           choices=['2017', '2013'],
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
