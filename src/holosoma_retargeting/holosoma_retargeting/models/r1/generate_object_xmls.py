#!/usr/bin/env python3
"""Generate r1_26dof_w_{object}.xml files for objects in models/objects/."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def _insert_attr_if_missing(tag: str, name: str, value: str) -> str:
    if re.search(rf"\b{re.escape(name)}=", tag):
        return tag
    return tag[:-2].rstrip() + f' {name}="{value}"/>'


def _add_ground_contact_properties(xml: str) -> str:
    match = re.search(r'<geom\s+name="ground"[^>]*/>', xml)
    if match is None:
        return xml

    tag = match.group(0)
    tag = _insert_attr_if_missing(tag, "quat", "1 0 0 0")
    tag = _insert_attr_if_missing(tag, "condim", "1")
    tag = _insert_attr_if_missing(tag, "conaffinity", "15")
    return xml[: match.start()] + tag + xml[match.end() :]


def _object_mesh_rel_path(robot_dir: Path, obj_file: Path, base_xml: str) -> str:
    meshdir_match = re.search(r'<compiler\b[^>]*\bmeshdir="([^"]+)"', base_xml)
    meshdir = meshdir_match.group(1) if meshdir_match else "."
    mesh_base_dir = (robot_dir / meshdir).resolve()
    return os.path.relpath(obj_file.resolve(), mesh_base_dir).replace(os.sep, "/")


def _extra_object_geoms_xml(obj_name: str) -> str:
    if obj_name != "suitcase":
        return ""

    wheel_positions = (
        (-0.0236, -0.3, 0.1623),
        (0.1629, -0.3, -0.0188),
        (-0.1629, -0.3, 0.0188),
        (0.0236, -0.3, -0.1623),
    )
    return "".join(
        f"""
        <geom name="suitcase_wheel_{idx}" type="sphere" size="0.02"
                contype="1" conaffinity="1"
                pos="{x:g} {y:g} {z:g}"
                rgba="0.08 0.08 0.08 1"
                friction="0.9 0.5 0.5"
                solref="0.02 1"
                solimp="0.9 0.95 0.001"/>"""
        for idx, (x, y, z) in enumerate(wheel_positions)
    )


def generate_object_xml(base_xml: str, obj_name: str, obj_rel_path: str) -> str:
    """Generate a combined R1 robot + object MuJoCo XML."""
    xml = base_xml

    mesh_name = f"{obj_name}_mesh"
    if f'name="{mesh_name}"' not in xml:
        mesh_line = f'    <mesh name="{mesh_name}" file="{obj_rel_path}" scale="1 1 1"/>\n'
        first_asset_close = xml.find("  </asset>")
        if first_asset_close == -1:
            raise ValueError("Could not find </asset> in base XML")
        xml = xml[:first_asset_close] + "\n" + mesh_line + xml[first_asset_close:]

    xml = _add_ground_contact_properties(xml)

    body_name = f"{obj_name}_link"
    if f'name="{body_name}"' not in xml:
        object_body = f"""
    <body name="{body_name}">
        <freejoint/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="0.002 0.002 0.002"/>
        <geom name="{obj_name}" type="mesh" mesh="{mesh_name}"
                contype="1" conaffinity="1"
                pos="0 0 0" quat="1 0 0 0"
                rgba="0.7 0.8 0.9 0.7"
                friction="0.9 0.5 0.5"
                solref="0.02 1"
                solimp="0.9 0.95 0.001"/>
{_extra_object_geoms_xml(obj_name)}
    </body>

    <light name="sun" pos="0 0 5" dir="0 0 -1" directional="true"
         diffuse="1 1 1" ambient="0.2 0.2 0.2" specular="0.2 0.2 0.2"
         castshadow="true"/>
"""
        xml = xml.replace("  </worldbody>", object_body + "  </worldbody>")

    return xml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate r1_26dof_w_{object}.xml files")
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Specific object names to generate. If omitted, generates for all objects with .obj files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print outputs without writing files.")
    args = parser.parse_args()

    r1_dir = Path(__file__).resolve().parent
    models_dir = r1_dir.parent
    objects_dir = models_dir / "objects"
    base_xml_path = r1_dir / "r1_26dof.xml"

    if not base_xml_path.exists():
        raise FileNotFoundError(f"Base XML not found: {base_xml_path}")
    if not objects_dir.exists():
        raise FileNotFoundError(f"Objects directory not found: {objects_dir}")

    base_xml = base_xml_path.read_text()
    obj_names = (
        args.objects
        if args.objects
        else sorted(d.name for d in objects_dir.iterdir() if d.is_dir() and (d / f"{d.name}.obj").exists())
    )

    print(f"Found {len(obj_names)} objects: {', '.join(obj_names)}")
    created: list[str] = []
    skipped: list[str] = []
    for obj_name in obj_names:
        obj_file = objects_dir / obj_name / f"{obj_name}.obj"
        if not obj_file.exists():
            print(f"  SKIP {obj_name}: no {obj_name}.obj found")
            skipped.append(obj_name)
            continue

        output_path = r1_dir / f"r1_26dof_w_{obj_name}.xml"
        obj_rel_path = _object_mesh_rel_path(r1_dir, obj_file, base_xml)

        if args.dry_run:
            print(f"  [DRY-RUN] Would create: {output_path.name}")
            created.append(obj_name)
            continue

        output_path.write_text(generate_object_xml(base_xml, obj_name, obj_rel_path))
        print(f"  Created: {output_path.name}")
        created.append(obj_name)

    print(f"\nDone: {len(created)} created, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
