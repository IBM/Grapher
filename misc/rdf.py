from typing import List, Any

from os.path import join
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom


import logging

logger = logging.getLogger(__name__)
log = logger


def create_xml(data, categories, ts_header, t_header):
    benchmark = Element('benchmark')
    entries = SubElement(benchmark, 'entries')

    assert len(categories) == len(data)

    for idx, triples in enumerate(data):

        entry = SubElement(entries, 'entry', {'category': categories[idx], 'eid': 'Id%s' % (idx + 1)})
        t_entry = SubElement(entry, ts_header)

        for triple in triples:
            element = SubElement(t_entry, t_header)
            element.text = triple

    return benchmark


def xml_prettify(elem):
    """Return a pretty-printed XML string for the Element.
       source : https://pymotw.com/2/xml/etree/ElementTree/create.html
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def save_webnlg_rdf(hyps: Any,
                    refs: Any,
                    categories: List,
                    out_dir: str,
                    iteration: str,
                    logger: Any = None):
    mprint = logger.info if (logger is not None) else print

    if len(refs) != len(hyps):
        raise Exception(f"reference size {len(refs)} is not same as hypothesis size {len(hyps)}")

    ref_xml = create_xml(refs, categories, "modifiedtripleset", "mtriple")
    hyp_xml = create_xml(hyps, categories, "generatedtripleset", "gtriple")

    ref_fname = join(out_dir, f"ref_{iteration}.xml")
    hyp_fname = join(out_dir, f"hyp_{iteration}.xml")

    mprint(f"creating reference xml  file : [{ref_fname}]")
    mprint(f"creating hypothesis xml file : [{hyp_fname}]")

    with open(ref_fname, 'w', encoding='utf-8') as f:
        f.write(xml_prettify(ref_xml))

    with open(hyp_fname, 'w', encoding='utf-8') as f:
        f.write(xml_prettify(hyp_xml))

    return ref_fname, hyp_fname