#!/usr/bin/env python3
"""
Compare two DOCX files to understand differences in structure and image embedding
"""
import zipfile
import os
from pathlib import Path
import xml.etree.ElementTree as ET

def analyze_docx_structure(docx_path):
    """Extract detailed structure information from a DOCX file"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(docx_path)}")
    print(f"File size: {os.path.getsize(docx_path):,} bytes")
    print(f"{'='*80}")

    with zipfile.ZipFile(docx_path, 'r') as zip_ref:
        # List all files
        print("\n--- File List ---")
        file_list = sorted(zip_ref.namelist())
        for fname in file_list:
            file_info = zip_ref.getinfo(fname)
            print(f"  {fname}: {file_info.file_size:,} bytes (compressed: {file_info.compress_size:,})")

        # Count images
        image_files = [f for f in file_list if f.startswith('word/media/')]
        print(f"\n--- Images: {len(image_files)} ---")
        for img in image_files:
            file_info = zip_ref.getinfo(img)
            print(f"  {img}: {file_info.file_size:,} bytes")

        # Analyze document.xml
        print("\n--- Document Structure (document.xml) ---")
        try:
            doc_xml = zip_ref.read('word/document.xml')
            root = ET.fromstring(doc_xml)

            # Namespace mapping
            namespaces = {
                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
                'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
            }

            # Count paragraphs
            paragraphs = root.findall('.//w:p', namespaces)
            print(f"  Total paragraphs: {len(paragraphs)}")

            # Count inline pictures
            inline_pics = root.findall('.//w:drawing//pic:pic', namespaces)
            print(f"  Inline pictures in document: {len(inline_pics)}")

            # Analyze picture details
            for i, pic in enumerate(inline_pics, 1):
                blip = pic.find('.//a:blip', namespaces)
                if blip is not None:
                    embed_id = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                    print(f"    Picture {i}: rId={embed_id}")

            # Check for alternate content (compatibility)
            alt_content = root.findall('.//w:drawing', namespaces)
            print(f"  Drawing elements: {len(alt_content)}")

        except Exception as e:
            print(f"  Error parsing document.xml: {e}")

        # Analyze document.xml.rels
        print("\n--- Relationships (document.xml.rels) ---")
        try:
            rels_xml = zip_ref.read('word/_rels/document.xml.rels')
            rels_root = ET.fromstring(rels_xml)

            # Namespace for relationships
            rels_ns = {'r': 'http://schemas.openxmlformats.org/package/2006/relationships'}

            relationships = rels_root.findall('.//r:Relationship', rels_ns)
            print(f"  Total relationships: {len(relationships)}")

            # Image relationships
            image_rels = [r for r in relationships if 'image' in r.get('Type', '')]
            print(f"  Image relationships: {len(image_rels)}")
            for rel in image_rels:
                print(f"    Id={rel.get('Id')}, Target={rel.get('Target')}, Type={rel.get('Type')}")

        except Exception as e:
            print(f"  Error parsing document.xml.rels: {e}")

        # Check styles.xml for formatting
        print("\n--- Styles (styles.xml) ---")
        try:
            styles_xml = zip_ref.read('word/styles.xml')
            styles_root = ET.fromstring(styles_xml)

            # Namespace
            w_ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

            # Default paragraph properties
            doc_defaults = styles_root.find('.//w:docDefaults//w:pPrDefault//w:pPr', w_ns)
            if doc_defaults is not None:
                spacing = doc_defaults.find('.//w:spacing', w_ns)
                if spacing is not None:
                    print(f"  Default paragraph spacing:")
                    for attr, val in spacing.attrib.items():
                        print(f"    {attr}: {val}")

            # Normal style
            normal_style = styles_root.find('.//w:style[@w:styleId="Normal"]', w_ns)
            if normal_style is not None:
                print(f"  Normal style found")
                spacing = normal_style.find('.//w:spacing', w_ns)
                if spacing is not None:
                    print(f"    Spacing attributes:")
                    for attr, val in spacing.attrib.items():
                        print(f"      {attr}: {val}")

        except Exception as e:
            print(f"  Error parsing styles.xml: {e}")

if __name__ == "__main__":
    artifacts_dir = Path(__file__).parent / "artifacts"

    file1 = artifacts_dir / "final_report.docx"
    file2 = artifacts_dir / "Moon_Market_판매현황보고서.docx"

    if file1.exists():
        analyze_docx_structure(file1)
    else:
        print(f"File not found: {file1}")

    if file2.exists():
        analyze_docx_structure(file2)
    else:
        print(f"File not found: {file2}")

    print("\n" + "="*80)
    print("Comparison Complete")
    print("="*80)
