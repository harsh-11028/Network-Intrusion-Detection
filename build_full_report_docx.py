import os
import zipfile
import html

def escape_xml(text):
    if text is None:
        return ""
    return html.escape(str(text))

class DocxBuilder:
    def __init__(self, filename="Machine_Learning_Network_Intrusion_Detection_Report.docx"):
        self.filename = filename
        self.paragraphs = []
        self.images = []  # tuples of (rel_id, filename, image_path, width_emu, height_emu)
        self.img_counter = 0

    def add_image(self, image_path, caption, width_in_inches=5.8, height_in_inches=3.5):
        if not os.path.exists(image_path):
            self.add_p(f"[{caption} - Image not found at {image_path}]", italic=True, align="center")
            return
        self.img_counter += 1
        rel_id = f"rIdImg{self.img_counter}"
        img_filename = f"image{self.img_counter}.png"
        cx = int(width_in_inches * 914400)
        cy = int(height_in_inches * 914400)
        self.images.append((rel_id, img_filename, image_path, cx, cy))

        # Add image xml
        img_xml = f"""
        <w:p>
          <w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="80"/></w:pPr>
          <w:r>
            <w:drawing>
              <wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
                <wp:extent cx="{cx}" cy="{cy}"/>
                <wp:docPr id="{self.img_counter}" name="Picture {self.img_counter}"/>
                <wp:cNvGraphicFramePr>
                  <a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>
                </wp:cNvGraphicFramePr>
                <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                  <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                    <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
                      <pic:nvPicPr>
                        <pic:cNvPr id="{self.img_counter}" name="{img_filename}"/>
                        <pic:cNvPicPr/>
                      </pic:nvPicPr>
                      <pic:blipFill>
                        <a:blip r:embed="{rel_id}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
                        <a:stretch><a:fillRect/></a:stretch>
                      </pic:blipFill>
                      <pic:spPr>
                        <a:xfrm>
                          <a:off x="0" y="0"/>
                          <a:ext cx="{cx}" cy="{cy}"/>
                        </a:xfrm>
                        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                      </pic:spPr>
                    </pic:pic>
                  </a:graphicData>
                </a:graphic>
              </wp:inline>
            </w:drawing>
          </w:r>
        </w:p>
        """
        self.paragraphs.append(img_xml)
        if caption:
            self.add_p(caption, italic=True, align="center", sz=21, space_after=180)

    def add_p(self, text="", bold=False, italic=False, underline=False, align="left", sz=24, 
              font="Times New Roman", space_before=60, space_after=100, line_spacing=276, color="000000"):
        jc_xml = f'<w:jc w:val="{align}"/>' if align != "left" else ''
        sp_xml = f'<w:spacing w:before="{space_before}" w:after="{space_after}" w:line="{line_spacing}" w:lineRule="auto"/>'
        
        rpr_elems = [f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>']
        if bold: rpr_elems.append('<w:b/>')
        if italic: rpr_elems.append('<w:i/>')
        if underline: rpr_elems.append('<w:u w:val="single"/>')
        if sz: rpr_elems.append(f'<w:sz w:val="{sz}"/>')
        if color != "000000": rpr_elems.append(f'<w:color w:val="{color}"/>')
        rpr_str = "".join(rpr_elems)

        escaped = escape_xml(text)
        xml = f"""
        <w:p>
          <w:pPr>{jc_xml}{sp_xml}</w:pPr>
          <w:r><w:rPr>{rpr_str}</w:rPr><w:t xml:space="preserve">{escaped}</w:t></w:r>
        </w:p>
        """
        self.paragraphs.append(xml)

    def add_page_break(self):
        self.paragraphs.append("""
        <w:p>
          <w:r><w:br w:type="page"/></w:r>
        </w:p>
        """)

    def add_h1(self, text, align="left"):
        self.add_p(text, bold=True, sz=32, align=align, space_before=240, space_after=120)

    def add_h2(self, text, align="left"):
        self.add_p(text, bold=True, sz=28, align=align, space_before=180, space_after=100)

    def add_h3(self, text, align="left"):
        self.add_p(text, bold=True, sz=24, align=align, space_before=140, space_after=80)

    def add_bullet(self, title, desc="", level=0):
        indent = 360 * (level + 1)
        xml = f"""
        <w:p>
          <w:pPr>
            <w:ind w:left="{indent}" w:hanging="240"/>
            <w:spacing w:before="40" w:after="80" w:line="260" w:lineRule="auto"/>
          </w:pPr>
          <w:r>
            <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
            <w:t>• </w:t>
          </w:r>
          <w:r>
            <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="24"/></w:rPr>
            <w:t xml:space="preserve">{escape_xml(title)} </w:t>
          </w:r>
          <w:r>
            <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
            <w:t xml:space="preserve">{escape_xml(desc)}</w:t>
          </w:r>
        </w:p>
        """
        self.paragraphs.append(xml)

    def add_table(self, headers, rows, col_widths=None):
        if not col_widths:
            # Default equal distribution across 9000 dxa (approx 6.25 in)
            w = int(9000 / len(headers))
            col_widths = [w] * len(headers)

        tbl_xml = ['<w:tbl>',
                   '<w:tblPr>',
                   '  <w:tblW w:w="0" w:type="auto"/>',
                   '  <w:jc w:val="center"/>',
                   '  <w:tblBorders>',
                   '    <w:top w:val="single" w:sz="6" w:space="0" w:color="003366"/>',
                   '    <w:bottom w:val="single" w:sz="6" w:space="0" w:color="003366"/>',
                   '    <w:insideH w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>',
                   '    <w:insideV w:val="none"/>',
                   '    <w:left w:val="none"/>',
                   '    <w:right w:val="none"/>',
                   '  </w:tblBorders>',
                   '  <w:tblCellMar>',
                   '    <w:top w:w="120" w:type="dxa"/>',
                   '    <w:bottom w:w="120" w:type="dxa"/>',
                   '    <w:left w:w="160" w:type="dxa"/>',
                   '    <w:right w:w="160" w:type="dxa"/>',
                   '  </w:tblCellMar>',
                   '</w:tblPr>']

        # Header Row
        tbl_xml.append('<w:tr>')
        tbl_xml.append('  <w:trPr><w:tblHeader/></w:trPr>')
        for idx, h in enumerate(headers):
            cw = col_widths[idx] if idx < len(col_widths) else 2000
            tbl_xml.append(f"""
            <w:tc>
              <w:tcPr>
                <w:tcW w:w="{cw}" w:type="dxa"/>
                <w:shd w:val="clear" w:color="auto" w:fill="003366"/>
              </w:tcPr>
              <w:p>
                <w:pPr><w:spacing w:before="60" w:after="60"/></w:pPr>
                <w:r>
                  <w:rPr>
                    <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
                    <w:b/>
                    <w:color w:val="FFFFFF"/>
                    <w:sz w:val="21"/>
                  </w:rPr>
                  <w:t xml:space="preserve">{escape_xml(h)}</w:t>
                </w:r>
              </w:p>
            </w:tc>
            """)
        tbl_xml.append('</w:tr>')

        # Data Rows
        for r_idx, row in enumerate(rows):
            bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
            tbl_xml.append('<w:tr>')
            for idx, val in enumerate(row):
                cw = col_widths[idx] if idx < len(col_widths) else 2000
                is_bold_col = (idx == 0)
                tbl_xml.append(f"""
                <w:tc>
                  <w:tcPr>
                    <w:tcW w:w="{cw}" w:type="dxa"/>
                    <w:shd w:val="clear" w:color="auto" w:fill="{bg}"/>
                  </w:tcPr>
                  <w:p>
                    <w:pPr><w:spacing w:before="60" w:after="60"/><w:line w:line="240" w:lineRule="auto"/></w:pPr>
                    <w:r>
                      <w:rPr>
                        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
                        {'<w:b/>' if is_bold_col else ''}
                        <w:sz w:val="20"/>
                      </w:rPr>
                      <w:t xml:space="preserve">{escape_xml(val)}</w:t>
                    </w:r>
                  </w:p>
                </w:tc>
                """)
            tbl_xml.append('</w:tr>')

        tbl_xml.append('</w:tbl>')
        tbl_xml.append('<w:p><w:pPr><w:spacing w:before="60" w:after="120"/></w:pPr></w:p>')
        self.paragraphs.append("\n".join(tbl_xml))

    def add_code_block(self, code_str):
        xml = f"""
        <w:tbl>
          <w:tblPr>
            <w:tblW w:w="9200" w:type="dxa"/>
            <w:jc w:val="center"/>
            <w:tblBorders>
              <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
              <w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
              <w:left w:val="single" w:sz="18" w:space="0" w:color="003366"/>
              <w:right w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
            </w:tblBorders>
            <w:tblCellMar>
              <w:top w:w="120" w:type="dxa"/>
              <w:bottom w:w="120" w:type="dxa"/>
              <w:left w:w="180" w:type="dxa"/>
              <w:right w:w="180" w:type="dxa"/>
            </w:tblCellMar>
          </w:tblPr>
          <w:tr>
            <w:tc>
              <w:tcPr>
                <w:tcW w:w="9200" w:type="dxa"/>
                <w:shd w:val="clear" w:color="auto" w:fill="F1F5F9"/>
              </w:tcPr>
        """
        lines = code_str.strip().split("\n")
        for line in lines:
            xml += f"""
              <w:p>
                <w:pPr><w:spacing w:before="20" w:after="20"/><w:line w:line="220" w:lineRule="auto"/></w:pPr>
                <w:r>
                  <w:rPr>
                    <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>
                    <w:color w:val="1E293B"/>
                    <w:sz w:val="18"/>
                  </w:rPr>
                  <w:t xml:space="preserve">{escape_xml(line)}</w:t>
                </w:r>
              </w:p>
            """
        xml += """
            </w:tc>
          </w:tr>
        </w:tbl>
        <w:p><w:pPr><w:spacing w:before="60" w:after="120"/></w:pPr></w:p>
        """
        self.paragraphs.append(xml)

    def save(self):
        z = zipfile.ZipFile(self.filename, 'w', zipfile.ZIP_DEFLATED)
        
        # [Content_Types].xml
        types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                 '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
                 '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                 '  <Default Extension="xml" ContentType="application/xml"/>']
        if self.images:
            types.append('  <Default Extension="png" ContentType="image/png"/>')
        types.append('  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>')
        types.append('  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>')
        types.append('  <Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>')
        types.append('  <Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>')
        types.append('</Types>')
        z.writestr('[Content_Types].xml', "\n".join(types))

        # _rels/.rels
        rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
        z.writestr('_rels/.rels', rels)

        # word/_rels/document.xml.rels
        doc_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                    '  <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
                    '  <Relationship Id="rIdHeader1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>',
                    '  <Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>']
        for rel_id, img_fn, _, _, _ in self.images:
            doc_rels.append(f'  <Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{img_fn}"/>')
        doc_rels.append('</Relationships>')
        z.writestr('word/_rels/document.xml.rels', "\n".join(doc_rels))

        # word/header1.xml
        header1 = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:p>
    <w:pPr>
      <w:jc w:val="right"/>
      <w:spacing w:after="120"/>
      <w:pBdr><w:bottom w:val="single" w:sz="6" w:space="4" w:color="CBD5E1"/></w:pBdr>
    </w:pPr>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
        <w:b/>
        <w:sz w:val="18"/>
        <w:color w:val="475569"/>
      </w:rPr>
      <w:t>Computer Science and Engineering | NIET Greater Noida</w:t>
    </w:r>
  </w:p>
</w:hdr>'''
        z.writestr('word/header1.xml', header1)

        # word/footer1.xml
        footer1 = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:p>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="120"/>
      <w:pBdr><w:top w:val="single" w:sz="4" w:space="4" w:color="E2E8F0"/></w:pBdr>
    </w:pPr>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
        <w:sz w:val="18"/>
        <w:color w:val="64748B"/>
      </w:rPr>
      <w:t xml:space="preserve">Machine Learning-Based NIDS Project Report  |  Page </w:t>
    </w:r>
    <w:fldSimple w:instr="PAGE"/>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
        <w:sz w:val="18"/>
        <w:color w:val="64748B"/>
      </w:rPr>
      <w:t xml:space="preserve"> of </w:t>
    </w:r>
    <w:fldSimple w:instr="NUMPAGES"/>
  </w:p>
</w:ftr>'''
        z.writestr('word/footer1.xml', footer1)

        # word/styles.xml
        styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
        <w:sz w:val="24"/>
        <w:color w:val="000000"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:line="276" w:lineRule="auto" w:after="100"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
</w:styles>'''
        z.writestr('word/styles.xml', styles)

        # Write images
        for _, img_fn, path, _, _ in self.images:
            with open(path, 'rb') as f:
                z.writestr(f"word/media/{img_fn}", f.read())

        # word/document.xml
        doc = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
               '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"',
               '            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"',
               '            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"',
               '            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"',
               '            xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">',
               '  <w:body>']
        doc.extend(self.paragraphs)
        doc.append("""
          <w:sectPr>
            <w:headerReference w:type="default" r:id="rIdHeader1"/>
            <w:footerReference w:type="default" r:id="rIdFooter1"/>
            <w:pgSz w:w="11906" w:h="16838"/>
            <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
          </w:sectPr>
        """)
        doc.append('  </w:body>')
        doc.append('</w:document>')

        z.writestr('word/document.xml', "\n".join(doc))
        z.close()
        print(f"Generated Word Document: {self.filename} ({os.path.getsize(self.filename):,} bytes)")

print("DocxBuilder class defined successfully.")
