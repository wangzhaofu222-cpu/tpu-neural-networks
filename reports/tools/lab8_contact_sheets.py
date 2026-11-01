"""Readable contact sheets for human visual verification of all figure groups."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def main():
    files=sorted(Path('experiments/lab8/figures').glob('*.png'));out=Path('tmp/lab8_qa');out.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    for batch in range(0,len(files),6):
        canvas=Image.new('RGB',(1800,1740),'white');draw=ImageDraw.Draw(canvas)
        for j,p in enumerate(files[batch:batch+6]):
            im=Image.open(p).convert('RGB');im.thumbnail((900,550));x=(j%2)*900+(900-im.width)//2;y=(j//2)*580+30
            canvas.paste(im,(x,y));draw.text(((j%2)*900+10,(j//2)*580+5),p.name,fill='black',font=font)
        canvas.save(out/f'contact_{batch//6+1}.png')
    print('Contact sheets:',(len(files)+5)//6)

if __name__=='__main__':main()
