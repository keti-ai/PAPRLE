from lxml import etree as ET
import os
import numpy as np
import paprle

def recursive_change(element, prefix='', found_first_body=False, offset=np.array([0,0,0]), keep_floor=False):
    for child in element.getchildren():
        if child.tag == 'body' and 'name' in child.attrib:
            child.attrib['name'] = prefix + child.attrib['name']
            if not found_first_body:
                found_first_body = True
                if 'pos' in child.attrib:
                    pos = np.fromstring(child.attrib['pos'], sep=' ')
                    pos += offset
                    child.attrib['pos'] = ' '.join([str(p) for p in pos])
                else:
                    child.attrib['pos'] = ' '.join([str(p) for p in offset])
            recursive_change(child, prefix)
        elif child.tag == 'geom' and 'name' in child.attrib:
            if child.attrib['name'] != 'floor':
                child.attrib['name'] = prefix + child.attrib['name']
            elif not keep_floor:
                element.remove(child)
        elif child.tag == 'joint' and 'name' in child.attrib:
            child.attrib['name'] = prefix + child.attrib['name']
        elif child.tag == 'site' and 'name' in child.attrib:
            child.attrib['name'] = prefix + child.attrib['name']
        elif child.tag == 'camera' and 'name' in child.attrib:
            child.attrib['name'] = prefix + child.attrib['name']
        elif child.tag == 'light' and not keep_floor:
            element.remove(child)
        elif child.tag == 'exclude':
            if 'body1' in child.attrib:
                if child.attrib['body1'] != 'world':
                    child.attrib['body1'] = prefix + child.attrib['body1']
            if 'body2' in child.attrib:
                if child.attrib['body2'] != 'world':
                    child.attrib['body2'] = prefix + child.attrib['body2']
        elif child.tag == 'contact':
            recursive_change(child, prefix)
        else:
            recursive_change(child, prefix)



def load_multiple_robot(xml_path, num_robots, spacing, prefix='r'):

    tree = ET.parse(os.path.abspath(xml_path))
    original_root = tree.getroot()
    found_floor = False
    for child in original_root.getchildren():
        if child.tag == 'compiler':
            xml_dir = os.path.dirname(os.path.abspath(xml_path))
            child.attrib['meshdir'] = os.path.abspath(os.path.join(xml_dir, child.attrib['meshdir']))
        elif child.tag == 'worldbody':
            recursive_change(child, f'r0_', keep_floor=True)
            for geom in child.findall('geom'):
                if 'name' in geom.attrib and geom.attrib['name'] == 'floor':
                    found_floor = True
        elif child.tag == 'contact':
            recursive_change(child, f'r0_')
        elif child.tag == 'include':
            original_root.remove(child)
        elif child.tag == 'actuator':
            original_root.remove(child)

    all_roots = []
    for i in range(1, num_robots):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tree_to_add = []
        found_first_body, offset = False, np.array([0, i * spacing, 0])
        for child in root.getchildren():
            if child.tag == 'worldbody':
                recursive_change(child, f'r{i}_', found_first_body, offset)
                tree_to_add.append(child)
            elif child.tag == 'contact':
                recursive_change(child, f'r{i}_')
                tree_to_add.append(child)
        all_roots.extend(tree_to_add)

    for new_root in all_roots:
        original_root.append(new_root)

    if not found_floor:
        scene_dir = os.path.join(paprle.__path__[0], '../models/assets/scene/floor_sky.xml')
        floor = ET.parse(os.path.abspath(scene_dir)).getroot()
        for child in floor.getchildren():
            original_root.append(child)
    xml_text = ET.tostring(original_root, encoding="unicode")
    return xml_text