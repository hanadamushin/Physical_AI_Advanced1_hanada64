"""Inspect the actual 3D head from front, side, rear and above before rendering."""
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from demo import xml_scene,electrode_layout,channel_names,label_head,HC,home_pose

def run(out='output_v07'):
    out=Path(out);out.mkdir(exist_ok=True)
    points,_,_=electrode_layout(16);names=channel_names(16)
    model=mujoco.MjModel.from_xml_string(xml_scene(points));data=mujoco.MjData(model)
    data.qpos[:]=home_pose();mujoco.mj_forward(model,data)
    for i in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        if model.geom_bodyid[i]>0:model.geom_group[i]=1
        if name=='floor' or name.startswith(('rail','jar_')) or name=='reservoir_gel':model.geom_group[i]=2
    option=mujoco.MjvOption();option.geomgroup[1:3]=0;option.sitegroup[:]=0
    r=mujoco.Renderer(model,height=540,width=720)
    fp=next(x for x in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(x).exists())
    font=ImageFont.truetype(fp,16)
    sheet=Image.new('RGB',(1440,1160),'#0e1621')
    try:
        for k,(title,az,el) in enumerate([('FRONT | PATIENT LEFT ON SCREEN RIGHT',90,-12),('PATIENT LEFT SIDE',180,-12),('REAR',270,-12),('TOP | FRONT UP, PATIENT LEFT ON SCREEN LEFT',270,-89.9)]):
            cam=mujoco.MjvCamera();cam.lookat[:]=HC;cam.distance=1.12;cam.azimuth=az;cam.elevation=el
            if k==3:cam.orthographic=1
            r.update_scene(data,camera=cam,scene_option=option)
            for _ in range(3):r.render()
            image=Image.fromarray(r.render());label_head(image,r.scene,points,names,font)
            image.save(out/f'qa_{k+1}.png')
            x=(k%2)*720;y=(k//2)*580
            sheet.paste(image,(x,y+40));ImageDraw.Draw(sheet).text((x+24,y+12),title,font=font,fill='white')
        sheet.save(out/'placement_review.png')
    finally:r.close()

if __name__=='__main__':
    run()
    import demo
    full_render=demo.render_video
    def preview_render(model,data,frames,out,fps,initial,names,supply_capacity,playback):
        aspirate=[i for i,f in enumerate(frames) if 'ASPIRATE' in f[1]]
        dispense=[i for i,f in enumerate(frames) if f[1].startswith('DISPENSE')]
        first_group=lambda indices:[i for i in indices if i<indices[0]+30 and (not any(j not in indices for j in range(indices[0],i+1)))]
        a=first_group(aspirate);b=first_group(dispense)
        selected=[frames[i] for indices in [a,b] for i in [indices[0],indices[len(indices)//2],indices[-1]]]
        full_render(model,data,selected,out,fps,initial,names,supply_capacity,playback)
        sheet=Image.new('RGB',(900,420),'#0e1621')
        for i in range(6):
            image=Image.open(out/f'preview_detail_{i:02d}.png').crop((418,88,714,259))
            sheet.paste(image,(i%3*300,30+i//3*210))
        ImageDraw.Draw(sheet).text((12,10),'ASPIRATION: START / MIDDLE / END',fill='white')
        ImageDraw.Draw(sheet).text((12,220),'INJECTION: START / MIDDLE / END',fill='white')
        sheet.save(out.parent/'piston_review.png')
    demo.render_video=preview_render
    demo.run('output_v07/preflight')
