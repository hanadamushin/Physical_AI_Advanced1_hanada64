"""Labelled top view of the template fitted to the demonstration head."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from demo import electrode_layout, channel_names, topomap_xy

def draw(path='output_v07/montage_1020.png'):
    image=Image.new('RGB',(720,760),'#0e1621');d=ImageDraw.Draw(image)
    font_path=next(x for x in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(x).exists())
    f=lambda n:ImageFont.truetype(font_path,n)
    d.text((38,20),'10-20 STANDARD POSITIONS | 16 SITES',font=f(26),fill='white')
    d.ellipse((95,105,625,635),outline='#8fa6b9',width=3)
    d.line([(337,106),(360,76),(383,106)],fill='#8fa6b9',width=3)
    d.text((300,52),'FRONT / NOSE',font=f(16),fill='#9bb0c5')
    d.text((15,355),'LEFT',font=f(16),fill='#9bb0c5')
    d.text((655,355),'RIGHT',font=f(16),fill='#9bb0c5')
    points,_,_=electrode_layout(16)
    for name,p in zip(channel_names(16),topomap_xy(points)):
        x,y=360+p[0]*250,370+p[1]*250
        d.ellipse((x-7,y-7,x+7,y+7),fill='#52dce1')
        w=d.textlength(name,font=f(22))
        d.text((x-w/2,y-35 if p[1]<-.5 else y+10),name,font=f(22),fill='white')
    d.text((38,675),'Landmark/arc-based model; azimuthal equidistant top view.',font=f(17),fill='#9bb0c5')
    d.text((38,705),'16-position subset; idealised anatomy, not human digitisation.',font=f(17),fill='#9bb0c5')
    Path(path).parent.mkdir(parents=True,exist_ok=True);image.save(path)

if __name__=='__main__':draw()
