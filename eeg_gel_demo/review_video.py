"""Decode the completed video and inspect evenly spaced encoded frames."""
from pathlib import Path
import json
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def review(out='output_v07'):
    out=Path(out);qa=out/'qa_video';qa.mkdir(exist_ok=True)
    summary=json.loads((out/'summary.json').read_text())
    reader=imageio.get_reader(out/'eeg_gel_demo.mp4')
    count=reader.count_frames()
    expected=len(range(0,summary['frames'],summary['playback_speed']))
    assert count==expected,(count,expected)
    sheet=Image.new('RGB',(1536,1456),'#0e1621')
    fp=next(x for x in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(x).exists())
    font=ImageFont.truetype(fp,17)
    try:
        for k,i in enumerate(np.linspace(0,count-1,12,dtype=int)):
            frame=Image.fromarray(reader.get_data(int(i)))
            frame.save(qa/f'frame_{k+1:02d}.png')
            x=(k%3)*512;y=(k//3)*364
            sheet.paste(frame.resize((512,336)),(x,y+28))
            ImageDraw.Draw(sheet).text((x+10,y+4),f'{i/summary["fps"]:.1f} s video',font=font,fill='white')
        phases=np.load(out/'replay.npz')['phase'][::summary['playback_speed']]
        for name,mask in [('piston_aspirate',np.char.find(phases,'ASPIRATE')>=0),('piston_dispense',np.char.startswith(phases,'DISPENSE'))]:
            start=int(np.flatnonzero(mask)[0]);end=start
            while end+1<len(mask) and mask[end+1]:end+=1
            Image.fromarray(reader.get_data((start+end)//2)).save(qa/f'{name}.png')
    finally:reader.close()
    sheet.save(qa/'contact_sheet.png')
    print(f'Full decode passed: {count} frames. Encoded frame contact sheet saved.')

if __name__=='__main__':review()
