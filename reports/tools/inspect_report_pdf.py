"""Render every report page for visual QA; text checks do not substitute visual review."""
import argparse,json,math
from pathlib import Path
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw


def main():
    p=argparse.ArgumentParser(); p.add_argument('pdf',type=Path); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); doc=pdfium.PdfDocument(str(a.pdf)); thumbs=[]
    for i,page in enumerate(doc):
        img=page.render(scale=1.5).to_pil().convert('RGB'); img.save(a.out/f'page-{i+1:02d}.png')
        thumb=ImageOps.contain(img,(480,680)); cell=Image.new('RGB',(500,710),'#dce2e8'); cell.paste(thumb,((500-thumb.width)//2,20))
        ImageDraw.Draw(cell).text((12,693),f'Page {i+1}',fill='black'); thumbs.append(cell)
    columns=3; sheet=Image.new('RGB',(columns*500,math.ceil(len(thumbs)/columns)*710),'white')
    for i,img in enumerate(thumbs): sheet.paste(img,((i%columns)*500,(i//columns)*710))
    sheet.save(a.out/'contact-sheet.png')
    reader=PdfReader(a.pdf); text='\n'.join(page.extract_text() for page in reader.pages)
    fonts={}
    for page in reader.pages:
        for font in page['/Resources'].get('/Font',{}).values():
            f=font.get_object(); desc=f.get('/FontDescriptor'); desc=desc.get_object() if desc else {}
            fonts[str(f.get('/BaseFont'))]=any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])
    result={'pages':len(reader.pages),'all_pages_rendered':len(thumbs)==len(reader.pages),
            'characters':len(text),'fonts_embedded':fonts,'contains_replacement_character':'\ufffd' in text,
            'visual_review':'PENDING_HUMAN_OR_AGENT_IMAGE_INSPECTION','pdf':str(a.pdf)}
    (a.out/'inspection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
