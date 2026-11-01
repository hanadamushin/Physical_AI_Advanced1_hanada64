"""Schematic of the open-front base rail and the rear gel station."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from demo import ARC_MIN,ARC_MAX,topomap_xy,electrode_layout,channel_names

def draw(path='output_v07/rear_arc_layout.png'):
    im=Image.new('RGB',(840,880),'#0e1621');d=ImageDraw.Draw(im)
    fp=next(x for x in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(x).exists())
    font=lambda n:ImageFont.truetype(fp,n)
    d.text((35,22),'REAR 270 DEG BASE ARC',font=font(29),fill='white')
    d.text((252,88),'FRONT 90 DEG | NO BASE',font=font(20),fill='#e2a39e')
    cx,cy,r=420,400,235
    pts=[(cx-330*np.cos(a),cy+330*np.sin(a)) for a in np.linspace(ARC_MIN,ARC_MAX,200)]
    d.line(pts,fill='#55c9d7',width=6)
    for x,y in [pts[0],pts[-1]]:d.ellipse((x-9,y-9,x+9,y+9),fill='#9be9ef')
    d.ellipse((cx-r,cy-r,cx+r,cy+r),outline='#8298a9',width=2)
    d.line([(cx-16,cy-r),(cx,cy-r-23),(cx+16,cy-r)],fill='#a1b4c4',width=2)
    points,_,_=electrode_layout(16)
    for name,(u,v) in zip(channel_names(16),topomap_xy(points)):
        x,y=cx+r*u,cy+r*v
        d.ellipse((x-5,y-5,x+5,y+5),fill='#53d9d7')
        tx,ty=(cx+r*u*1.10,cy+r*v*1.10) if np.hypot(u,v)>.8 else (x,y+17)
        d.text((tx-d.textlength(name,font=font(17))/2,ty-8),name,font=font(17),fill='white')
    d.rounded_rectangle((386,660,454,700),radius=8,fill='#247d87',outline='#7ee5e8',width=2)
    d.text((302,741),'GEL STATION (REAR)',font=font(20),fill='#91dbe1')
    d.text((45,770),'Base stays on the open arc; frontal sites are reached from its ends.',font=font(20),fill='#b6c9d9')
    d.text((45,809),'Schematic only. Robot motion and impedance response are simulated.',font=font(18),fill='#8ca5ba')
    Path(path).parent.mkdir(parents=True,exist_ok=True);im.save(path)

if __name__=='__main__':draw()
