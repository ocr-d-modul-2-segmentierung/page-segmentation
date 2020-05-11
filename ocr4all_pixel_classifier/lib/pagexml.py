import enum
import os
from lxml import etree
from typing import NamedTuple, List, Tuple, Optional, Set

import numpy as np
from PIL import Image, ImageDraw


class MaskType(enum.Enum):
    ALLTYPES = 'all_types'
    TEXT_GRAPHICS = 'text_nontext'
    BASE_LINE = 'baseline'
    TEXT_LINE = 'textline'
    TEXT_ONLY = 'text_only'

    def get_color(self, region: 'Region', capital_is_text: bool) -> Tuple[int, int, int]:
        def color_tg(x): return x.type.color_text_graphics(capital_is_text)

        f = {
            MaskType.ALLTYPES: lambda x: x.type.color,
            MaskType.TEXT_GRAPHICS: color_tg,
            MaskType.BASE_LINE: color_tg,
            MaskType.TEXT_LINE: color_tg,
            MaskType.TEXT_ONLY: lambda x: x.type.color_text_only(capital_is_text)
        }[self]
        return f(region)


class PCGTSVersion(enum.Enum):
    PCGTS2019 = '2019'
    PCGTS2017 = '2017'
    PCGTS2013 = '2013'
    PCGTS2010 = '2010'

    def get_namespace(self):
        return {
            PCGTSVersion.PCGTS2019: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15',
            PCGTSVersion.PCGTS2017: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15',
            PCGTSVersion.PCGTS2013: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
            PCGTSVersion.PCGTS2010: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19',
        }[self]

    @staticmethod
    def detect(root: etree.Element):
        for ns in root.nsmap.values():
            if ns.startswith('http://schema.primaresearch.org/PAGE/gts/pagecontent'):
                for version in PCGTSVersion:
                    if version.get_namespace() == ns:
                        return version
                else:
                    raise Exception('Unknown Schema Version')
        else:
            raise Exception('No PAGE namespace found')


class MaskSetting(NamedTuple):
    mask_extension: str = 'png'
    mask_type: MaskType = MaskType.ALLTYPES
    pcgts_version: Optional[PCGTSVersion] = None  # autodetect if not given
    line_width: int = 5
    capital_is_text: bool = False
    use_xml_filename: bool = False # if true, use the xml file's base name instead of the PageXML filename attribute for output


class PageXMLTypes(enum.Enum):
    PARAGRAPH = ('paragraph', (255, 0, 0))
    IMAGE = ('ImageRegion', (0, 255, 0))
    GRAPHIC = ('GraphicRegion', (0, 255, 0))
    TABLE = ('TableRegion', (0, 128, 0))
    MATHS = ('MathsRegion', (0, 0, 128))
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
    FOOTER = ('footer', (255, 128, 128))
    FOOTNOTE = ('footnote', (128, 255, 128))
    FOOTNOTE_CONTINUED = ('footnote-continued', (128, 255, 128))
    UNKNOWN = ('', (10, 10, 10))

    def __new__(cls, value, color):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.color = color
        obj.label = value
        return obj

    def color_text_graphics(self, capital_is_text=False) -> Tuple[int, int, int]:
        if self.is_text(capital_is_text):
            return (255, 0, 0)
        else:
            return (0, 255, 0)

    def color_text_only(self, capital_is_text=False) -> Tuple[int, int, int]:
        if self.is_text(capital_is_text):
            return (255, 0, 0)
        else:
            return (255, 255, 255)

    def is_text(self, capital_is_text: bool) -> bool:
        return not (
                self is PageXMLTypes.IMAGE
                or self is PageXMLTypes.GRAPHIC
                or (self is PageXMLTypes.DROP_CAPITAL and not capital_is_text)
        )

    @classmethod
    def image_map(cls, mask_type: MaskType):
        types = {
            MaskType.ALLTYPES: PageXMLTypes,
            MaskType.TEXT_GRAPHICS: [PageXMLTypes.PARAGRAPH, PageXMLTypes.IMAGE],
            MaskType.TEXT_ONLY: [PageXMLTypes.PARAGRAPH],
            MaskType.TEXT_LINE: [PageXMLTypes.PARAGRAPH],
            MaskType.BASE_LINE: [PageXMLTypes.PARAGRAPH],
        }[mask_type]

        map = {
            str(xmltype.color): (i + 1, xmltype.label)
            for (i, xmltype) in enumerate(types)
        }
        map['(255, 255, 255)'] = (0, 'background')
        return map


class Region(NamedTuple):
    polygon: List[Tuple[int, int]]
    type: PageXMLTypes


class PageRegions(NamedTuple):
    image_size: Tuple[int, int]
    xml_regions: List[Region]
    filename: str

    def only_types(self, types: Set[PageXMLTypes]) -> 'PageRegions':
        return PageRegions(image_size=self.image_size,
                           xml_regions=[x for x in self.xml_regions if x.type in types],
                           filename=self.filename)


class MaskGenerator:

    def __init__(self, settings: MaskSetting):
        self.settings = settings

    def save(self, file, output_dir):
        a = get_xml_regions(file, self.settings)
        mask_pil = page_region_to_mask(a, self.settings)
        filename_wo_ext = os.path.splitext(os.path.basename(file if self.settings.use_xml_filename else a.filename))[0]
        os.makedirs(output_dir, exist_ok=True)
        mask_pil.save(os.path.join(output_dir, filename_wo_ext + '.mask.' + self.settings.mask_extension))


def string_to_lp(points: str):
    lp_points: List[Tuple[int, int]] = []
    if points is not None:
        for point in points.split(' '):
            x, y = point.split(',')
            lp_points.append((int(x), int(y)))
    return lp_points


def coords_for_element(element, namespaces, tag: str = 'pcgts:Coords', type: Optional[PageXMLTypes] = None) -> Optional[Region]:
    coords = element.find(tag, namespaces)
    if coords is not None:
        polyline = string_to_lp(coords.get('points'))
        if not type:
            type = PageXMLTypes(element.get('type')) if 'type' in element.attrib else PageXMLTypes('paragraph')
        return Region(polygon=polyline, type=type)
    else:
        return None


def nested_child_regions(child, namespaces, tag: str = 'pcgts:Coords') -> List[Region]:
    return [
        coords_for_element(textline, namespaces, tag)
        for textline in child.findall('pcgts:TextLine', namespaces)
        if textline is not None
    ]


def get_xml_regions(xml_file, setting: MaskSetting) -> PageRegions:
    root = etree.parse(xml_file).getroot()
    if setting.pcgts_version:
        namespaces = {'pcgts': setting.pcgts_version.get_namespace()}
    else:
        namespaces = {'pcgts': PCGTSVersion.detect(root).get_namespace()}

    region_by_types = []
    for child in root.findall('.//pcgts:TextRegion', namespaces):
        if setting.mask_type in [MaskType.ALLTYPES, MaskType.TEXT_GRAPHICS, MaskType.TEXT_ONLY]:
            region_by_types.append(coords_for_element(child, namespaces))
        elif setting.mask_type == setting.mask_type.TEXT_LINE:
            region_by_types += nested_child_regions(child, namespaces, 'pcgts:Coords')
        elif setting.mask_type == setting.mask_type.BASE_LINE:
            region_by_types += nested_child_regions(child, namespaces, 'pcgts:Baseline')

    for region_tag in ["MathsRegion", "TableRegion"]:
        type = PageXMLTypes(region_tag)
        for child in root.findall('.//pcgts:'+region_tag, namespaces):
            if setting.mask_type == MaskType.ALLTYPES:
                region_by_types.append(coords_for_element(child, namespaces, type=type))

    from itertools import chain
    for child in chain(root.findall('.//pcgts:ImageRegion', namespaces),
            root.findall('.//pcgts:GraphicRegion', namespaces)):
        if setting.mask_type == setting.mask_type.TEXT_GRAPHICS or setting.mask_type == setting.mask_type.ALLTYPES:
            coords = child.find('pcgts:Coords', namespaces)
            if coords is not None:
                polyline = string_to_lp(coords.get('points'))
                region_by_types.append(Region(polygon=polyline, type=PageXMLTypes('ImageRegion')))

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


def page_region_to_binary_mask(page_region: PageRegions) -> np.ndarray:
    height, width = page_region.image_size
    pil_image = Image.new('1', (width, height), 0)
    for x in page_region.xml_regions:
        ImageDraw.Draw(pil_image).polygon(x.polygon, outline=1, fill=1)
    return np.asarray(pil_image)


def page_region_to_mask(page_region: PageRegions, setting: MaskSetting) -> Image:
    height, width = page_region.image_size
    pil_image = Image.new('RGB', (width, height), (255, 255, 255))
    canvas = ImageDraw.Draw(pil_image)
    for x in page_region.xml_regions:
        color = setting.mask_type.get_color(x, setting.capital_is_text)

        if (setting.mask_type in [MaskType.ALLTYPES, MaskType.TEXT_GRAPHICS, MaskType.TEXT_ONLY] and len(x.polygon) > 2) \
                or setting.mask_type is MaskType.TEXT_LINE:
            canvas.polygon(x.polygon, outline=color, fill=color)
        elif setting.mask_type is MaskType.BASE_LINE:
            canvas.line(x.polygon, fill=color, width=setting.line_width)

    return pil_image
