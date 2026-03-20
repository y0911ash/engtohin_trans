import sys
import zipfile
import xml.etree.ElementTree as ET

def read_docx(path):
    with zipfile.ZipFile(path) as docx:
        content = docx.read('word/document.xml')
        tree = ET.fromstring(content)
        
        # NS mappings for Word XML
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        text = []
        for p in tree.findall('.//w:p', ns):
            p_text = []
            for t in p.findall('.//w:t', ns):
                p_text.append(t.text)
            if p_text:
                text.append(''.join(p_text))
                
        print('\n'.join(text))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        read_docx(sys.argv[1])
