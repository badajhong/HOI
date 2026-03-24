#!/usr/bin/env python3
"""Generate g1_29dof_w_{object}.xml files for all objects in models/objects/.

Each generated XML takes the base g1_29dof.xml and adds:
1. A mesh asset pointing to the object's .obj file
2. A free-floating body in the worldbody for the object
3. Ground contact properties (condim/conaffinity) and a sun light

Usage:
    python generate_object_xmls.py
    python generate_object_xmls.py --objects largebox smallbox suitcase
    python generate_object_xmls.py --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def generate_object_xml(base_xml: str, obj_name: str, obj_rel_path: str) -> str:
    """Generate a combined robot+object XML from the base robot XML.

    Args:
        base_xml: Contents of g1_29dof.xml
        obj_name: Object name (e.g. "largebox", "suitcase")
        obj_rel_path: Relative path from g1/ dir to the object .obj file

    Returns:
        Modified XML string with the object added.
    """
    xml = base_xml

    # 1. Add object mesh asset: insert before </asset> (first occurrence, inside <asset> block near top)
    mesh_line = f'    <mesh name="{obj_name}_mesh" file="{obj_rel_path}" scale="1 1 1"/>\n'
    # Find the closing </asset> that's inside the first <asset> block (robot meshes)
    # The base XML has robot mesh assets followed by </asset>
    # We insert the object mesh right before the first </asset>
    first_asset_close = xml.find("  </asset>")
    if first_asset_close == -1:
        raise ValueError("Could not find </asset> in base XML")
    xml = xml[:first_asset_close] + "\n" + mesh_line + xml[first_asset_close:]

    # 2. Modify ground geom: add contact properties
    xml = re.sub(
        r'(<geom name="ground"[^/]*)(material="MatPlane"\s*)/>',
        r'\1quat="1 0 0 0" \2condim="1" conaffinity="15"/>',
        xml,
    )

    # 3. Add object body and sun light before </worldbody>
    object_body = f"""
    <body name="{obj_name}_link">
        <freejoint/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="0.002 0.002 0.002"/>
        <geom name="{obj_name}" type="mesh" mesh="{obj_name}_mesh"
                contype="1" conaffinity="1"
                pos="0 0 0" quat="1 0 0 0"
                rgba="0.7 0.8 0.9 0.7"
                friction="0.9 0.5 0.5"
                solref="0.02 1"
                solimp="0.9 0.95 0.001"/>
    </body>

    <light name="sun" pos="0 0 5" dir="0 0 -1" directional="true"
         diffuse="1 1 1" ambient="0.2 0.2 0.2" specular="0.2 0.2 0.2"
         castshadow="true"/>
"""
    xml = xml.replace("  </worldbody>", object_body + "  </worldbody>")

    return xml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate g1_29dof_w_{object}.xml files")
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Specific object names to generate. If omitted, generates for all objects with .obj files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which files would be created without writing them.",
    )
    args = parser.parse_args()

    g1_dir = Path(__file__).resolve().parent
    objects_dir = g1_dir.parent / "objects"
    base_xml_path = g1_dir / "g1_29dof.xml"

    if not base_xml_path.exists():
        raise FileNotFoundError(f"Base XML not found: {base_xml_path}")
    if not objects_dir.exists():
        raise FileNotFoundError(f"Objects directory not found: {objects_dir}")

    base_xml = base_xml_path.read_text()

    # Discover objects
    if args.objects:
        obj_names = args.objects
    else:
        obj_names = sorted(
            d.name
            for d in objects_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}.obj").exists()
        )

    print(f"Found {len(obj_names)} objects: {', '.join(obj_names)}")

    created = []
    skipped = []
    for obj_name in obj_names:
        obj_file = objects_dir / obj_name / f"{obj_name}.obj"
        if not obj_file.exists():
            print(f"  SKIP {obj_name}: no {obj_name}.obj found")
            skipped.append(obj_name)
            continue

        # Relative path from g1/ to objects/{obj}/{obj}.obj
        obj_rel_path = f"../../objects/{obj_name}/{obj_name}.obj"
        output_path = g1_dir / f"g1_29dof_w_{obj_name}.xml"

        if args.dry_run:
            print(f"  [DRY-RUN] Would create: {output_path.name}")
            created.append(obj_name)
            continue

        xml_content = generate_object_xml(base_xml, obj_name, obj_rel_path)
        output_path.write_text(xml_content)
        print(f"  Created: {output_path.name}")
        created.append(obj_name)

    print(f"\nDone: {len(created)} created, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
