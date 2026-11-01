"""EEG gel assistance: kinematic MuJoCo demo with synthetic feedback only."""
from pathlib import Path
import argparse, csv, json, math
from functools import lru_cache
import numpy as np

THRESHOLD = 10.0  # Arbitrary simulation criterion; not a clinical recommendation.
RADIUS, SHOULDER_Z, L1, L2, TOOL = .66, .72, .50, .46, .15
HC = np.array([0., 0., .80])
HS = np.array([.204, .228, .276])
SYRINGE_CAPACITY, RESERVOIR_CAPACITY = .48, 2.4
PISTON_STROKE = .075  # Visually scaled model stroke, not a calibrated syringe.
ARC_MIN, ARC_MAX = -np.pi/4, 5*np.pi/4
SUPPLY_ANGLE, SUPPLY_RADIUS = np.pi/2, .44
SUPPLY_X, SUPPLY_Y = SUPPLY_RADIUS*np.cos(SUPPLY_ANGLE), SUPPLY_RADIUS*np.sin(SUPPLY_ANGLE)

def home_pose(angle=ARC_MIN):
    return np.array([angle,0.,1.8,-2.8,0.,np.pi/2+1.0,0.])

def piston_extension(volume):
    return PISTON_STROKE*float(np.clip(volume/SYRINGE_CAPACITY,0.,1.))

def base_for_target(point):
    if np.linalg.norm(point[:2])<1e-6:return (ARC_MIN+ARC_MAX)/2
    a=(np.arctan2(point[1],point[0])-ARC_MIN)%(2*np.pi)+ARC_MIN
    if a<=ARC_MAX:return a
    return ARC_MAX if a-ARC_MAX < ARC_MIN+2*np.pi-a else ARC_MIN
CHANNELS_16=['Fp1','Fp2','F7','F3','F4','F8','T7','C3','C4','T8','P7','P3','P4','P8','O1','O2']
CHANNELS_32=CHANNELS_16+['Fz','Cz','Pz','AF3','AF4','FC5','FC1','FC2','FC6','CP5','CP1','CP2','CP6','PO3','PO4','Oz']
def channel_names(count):
    return (CHANNELS_16 if count==16 else CHANNELS_32 if count==32 else ['Fp1','Fp2','F3','F4','C3','C4','O1','O2']).copy()
def electrode_conditions(count):
    initial=np.resize(np.array([7.,25.,18.,31.,8.,23.,28.,20.]),count)
    rates=np.resize(np.array([6.,6.8,5.5,7.5,6.,4.8,6.2,8.]),count)
    rates[6]=0.  # T7 is deliberately nonresponsive in the 16/32 position sets.
    return initial,rates,channel_names(count)

def features(z, d):
    z, d = np.broadcast_arrays(np.asarray(z)/30., np.asarray(d)/.16)
    return np.stack([np.ones_like(z), z, d, z*d, d*d, z*d*d, z*d*d*d], axis=-1)

def response(z, dose, rate):
    return 3.5 + (z-3.5)*np.exp(-rate*dose)

class DoseModel:
    """Ridge regression trained on synthetic intervention examples, then frozen."""
    def __init__(self):
        rng = np.random.default_rng(103)
        z=rng.uniform(10,36,1600); d=rng.uniform(.02,.16,1600)
        y=z-response(z,d,rng.uniform(4,9,1600))+rng.normal(0,.1,1600)
        x=features(z,d)
        self.w=np.linalg.solve(x.T@x+1e-4*np.eye(x.shape[1]),x.T@y)
        rng=np.random.default_rng(904)
        z=rng.uniform(10,36,500); d=rng.uniform(.02,.16,500)
        actual=z-response(z,d,rng.uniform(4,9,500))
        self.test_mae=float(np.mean(np.abs(features(z,d)@self.w-actual)))

    def choose(self,z,scale=1.):
        choices=np.array([.04,.08,.12,.16])
        improvement=np.maximum(0,features(z,choices)@self.w)*scale
        ok=np.where(z-improvement<=8.8)[0]
        i=int(ok[0]) if len(ok) else 3
        return float(choices[i]),float(improvement[i])

def arc_sample(curve,fractions):
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
    return np.array([np.interp(np.asarray(fractions)*lengths[-1],lengths,curve[:,axis]) for axis in range(3)]).T

@lru_cache(maxsize=1)
def montage_geometry():
    """Idealised 10-20 construction from model landmarks and measured scalp arcs.

    Nasion/inion and preauricular landmarks define equatorial endpoints. Arc
    distances are measured on the ellipsoid, rather than an MRI sphere fitted
    without anatomical registration. This is a model, not human digitisation.
    """
    t=np.linspace(0,np.pi,12001)
    sagittal=HC+np.c_[np.zeros_like(t),-HS[1]*np.cos(t),HS[2]*np.sin(t)]
    coronal=HC+np.c_[-HS[0]*np.cos(t),np.zeros_like(t),HS[2]*np.sin(t)]
    xyz=dict(zip(['Fpz','Fz','Cz','Pz','Oz'],arc_sample(sagittal,[.1,.3,.5,.7,.9])))
    xyz.update(zip(['T7','C3','Cz','C4','T8'],arc_sample(coronal,[.1,.3,.5,.7,.9])))
    phis=np.linspace(-np.pi/2,-np.pi/2-2*np.pi,1601)
    rim=[]
    theta=np.linspace(0,np.pi/2,2001)
    for phi in phis:
        meridian=HC+np.c_[HS[0]*np.sin(theta)*np.cos(phi),HS[1]*np.sin(theta)*np.sin(phi),HS[2]*np.cos(theta)]
        rim.append(arc_sample(meridian,[.8])[0])
    rim=np.array(rim)
    rim_names=['Fpz','Fp1','F7','T7','P7','O1','Oz','O2','P8','T8','F8','Fp2']
    rim_fracs=[0,.05,.15,.25,.35,.45,.5,.55,.65,.75,.85,.95]
    xyz.update(zip(rim_names,arc_sample(rim,rim_fracs)))
    lateral=[]
    for side in (1,2):
        a,b,c=[(xyz[n]-HC)/HS for n in [f'Fp{side}',f'C{3 if side==1 else 4}',f'O{side}']]
        normal=np.cross(b-a,c-a);normal/=np.linalg.norm(normal)
        centre=normal*np.dot(normal,a);radius=np.linalg.norm(a-centre)
        u=(a-centre)/radius;v=np.cross(normal,u)
        angle=lambda p:np.arctan2(np.dot(p-centre,v),np.dot(p-centre,u))%(2*np.pi)
        if angle(b)>angle(c):v=-v
        tt=np.linspace(0,angle(c),12001)
        curve=HC+HS*(centre+radius*(np.cos(tt)[:,None]*u+np.sin(tt)[:,None]*v))
        lateral.append(curve)
        xyz[f'F{3 if side==1 else 4}'],xyz[f'P{3 if side==1 else 4}']=arc_sample(curve,[.25,.75])
        xyz[f'AF{3 if side==1 else 4}'],xyz[f'PO{3 if side==1 else 4}']=arc_sample(curve,[.125,.875])
    def midpoint(names):
        unit=np.mean([(xyz[n]-HC)/HS for n in names],axis=0)
        return HC+HS*unit/np.linalg.norm(unit)
    for name,anchors in {
        'FC5':['F7','F3','T7','C3'],'FC1':['F3','Fz','C3','Cz'],
        'FC2':['Fz','F4','Cz','C4'],'FC6':['F4','F8','C4','T8'],
        'CP5':['T7','C3','P7','P3'],'CP1':['C3','Cz','P3','Pz'],
        'CP2':['Cz','C4','Pz','P4'],'CP6':['C4','T8','P4','P8']}.items():xyz[name]=midpoint(anchors)
    guides=[arc_sample(sagittal,np.linspace(.1,.9,73)),arc_sample(coronal,np.linspace(.1,.9,73)),rim[::20],*[c[::160] for c in lateral]]
    # Model anterior is -Y, so patient LEFT must be +X to keep the anatomical
    # frame right-handed. A front camera therefore sees Fp1 on screen RIGHT.
    def to_world(v):
        result=np.array(v,copy=True);result[...,0]*=-1;return result
    audit={'sagittal':sagittal,'coronal':coronal,'rim':rim,'lateral_left':lateral[0],'lateral_right':lateral[1]}
    return {k:to_world(v) for k,v in xyz.items()},[to_world(g) for g in guides],{k:to_world(v) for k,v in audit.items()}

def electrode_layout(count=16):
    if count not in (8,16,32): raise ValueError('Choose 8, 16 or 32 electrodes')
    xyz,_,_=montage_geometry()
    points=np.array([xyz[n] for n in channel_names(count)])
    normals=(points-HC)/(HS*HS);normals/=np.linalg.norm(normals,axis=1)[:,None]
    return points,normals,np.arctan2(points[:,1],points[:,0])

def topomap_xy(points):
    """Azimuthal equidistant projection: nasion/front is at the top of the map."""
    unit=(np.asarray(points)-HC)/HS
    polar=np.arccos(np.clip(unit[:,2],-1,1));phi=np.arctan2(unit[:,1],unit[:,0])
    rim_polar=np.arccos((montage_geometry()[0]['T7'][2]-HC[2])/HS[2])
    return np.c_[-polar/rim_polar*np.cos(phi),polar/rim_polar*np.sin(phi)]

def label_head(image,scene,points,names,font,offset=(0,0)):
    from PIL import ImageDraw
    d=ImageDraw.Draw(image)
    cams=scene.camera
    pos=(np.array(cams[0].pos)+np.array(cams[1].pos))/2
    forward=np.array(cams[0].forward);up=np.array(cams[0].up);right=np.cross(forward,up)
    near=cams[0].frustum_near;bottom=cams[0].frustum_bottom;top=cams[0].frustum_top
    height=540;width=720;frustum_width=(top-bottom)*width/height
    normals=(points-HC)/(HS*HS)
    for name,p,n in zip(names,points,normals):
        if np.dot(n,pos-p)<=0:continue
        v=p-pos;depth=np.dot(v,forward)
        scale=1. if cams[0].orthographic else near/depth
        x=(.5+np.dot(v,right)*scale/frustum_width)*width+offset[0]
        y=(top-np.dot(v,up)*scale)/(top-bottom)*height+offset[1]
        text_width=d.textlength(name,font=font)
        d.rounded_rectangle((x+8,y-9,x+12+text_width,y+7),radius=2,fill='#14202c')
        d.text((x+10,y-8),name,font=font,fill='white')

def draw_topomap(d,points,names,values,failed,active,angle,font):
    cx,cy,r=124,221,77
    d.rounded_rectangle((14,88,235,330),radius=8,fill='#152332')
    d.text((27,99),'270 DEG | FRONT UP',font=font,fill='#cadbe7')
    d.ellipse((cx-r,cy-r,cx+r,cy+r),outline='#657e92',width=1)
    d.line([(cx-9,cy-r),(cx,cy-r-11),(cx+9,cy-r)],fill='#9bb0c5',width=1)
    rail=[(cx-102*np.cos(a),cy+102*np.sin(a)) for a in np.linspace(ARC_MIN,ARC_MAX,100)]
    d.line(rail,fill='#6babb9',width=2)
    for a in np.linspace(ARC_MAX,ARC_MIN+2*np.pi,16):
        x,y=cx-102*np.cos(a),cy+102*np.sin(a)
        d.ellipse((x-1,y-1,x+1,y+1),fill='#b16a65')
    d.text((96,118),'NO BASE',font=font,fill='#d9a19a')
    for i,(x,y) in enumerate(topomap_xy(points)):
        px,py=cx+r*x,cy+r*y
        color='#ffb547' if failed[i] else ('#35d499' if values[i]<=THRESHOLD else '#fa5c62')
        d.ellipse((px-3,py-3,px+3,py+3),fill=color,outline='white' if i==active else color)
        tx,ty=(cx+r*x*1.15,cy+r*y*1.15) if np.hypot(x,y)>.8 else (px,py+10)
        d.text((tx-d.textlength(names[i],font=font)/2,ty-6),names[i],font=font,fill='white')
    x,y=cx-102*np.cos(angle),cy+102*np.sin(angle)
    d.ellipse((x-4,y-4,x+4,y+4),fill='#6ce2ee')

def xml_scene(points):
    normals=(points-HC)/(HS*HS);normals/=np.linalg.norm(normals,axis=1)[:,None]
    electrodes='';gel='';guide_geoms=''
    for i,(p,n) in enumerate(zip(points,normals)):
        pos=p+.004*n
        quat=np.r_[1+n[2],-n[1],n[0],0.];quat/=np.linalg.norm(quat)
        electrodes+=f'<geom name="e{i}" type="cylinder" pos="{" ".join(map(str,pos))}" quat="{" ".join(map(str,quat))}" size=".010 .003" rgba="1 .4 .2 1" contype="0" conaffinity="0"/>'
        gel+=f'<geom name="gel{i}" type="sphere" pos="{" ".join(map(str,p+.009*n))}" size=".004" rgba=".1 .9 .95 0" contype="0" conaffinity="0"/>'
    for guide in montage_geometry()[1]:
        for a,b in zip(guide[:-1],guide[1:]):
            a=HC+(a-HC)*1.006;b=HC+(b-HC)*1.006
            guide_geoms+=f'<geom type="capsule" fromto="{" ".join(map(str,np.r_[a,b]))}" size=".0011" rgba=".43 .53 .61 .55" contype="0" conaffinity="0"/>'
    rail_angles=np.linspace(ARC_MIN,ARC_MAX,74)
    rail=''.join(f'<geom name="rail{i}" type="capsule" fromto="{RADIUS*np.cos(a)} {RADIUS*np.sin(a)} .035 {RADIUS*np.cos(b)} {RADIUS*np.sin(b)} .035" size=".008" rgba=".2 .35 .42 1" contype="0" conaffinity="0"/>' for i,(a,b) in enumerate(zip(rail_angles[:-1],rail_angles[1:])))
    rail+=''.join(f'<geom name="rail_end{i}" type="sphere" pos="{RADIUS*np.cos(a)} {RADIUS*np.sin(a)} .035" size=".018" rgba=".2 .8 .85 1" contype="0" conaffinity="0"/>' for i,a in enumerate([ARC_MIN,ARC_MAX]))
    jar_walls=''.join(f'<geom name="jar_wall{i}" type="capsule" fromto="{SUPPLY_X+.046*np.cos(a)} {SUPPLY_Y+.046*np.sin(a)} .59 {SUPPLY_X+.046*np.cos(a)} {SUPPLY_Y+.046*np.sin(a)} .69" size=".003" rgba=".6 .8 .9 .5"/>' for i,a in enumerate(np.linspace(0,2*np.pi,24,endpoint=False)))
    return f'''<mujoco model="EEG gel assistance">
    <compiler angle="radian"/><option gravity="0 0 0"/>
    <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/><headlight ambient=".4 .4 .4" diffuse=".7 .7 .7" specular=".2 .2 .2"/></visual>
    <default><geom friction=".8 .1 .1"/><joint damping="5"/></default>
    <worldbody><light pos="1 -2 4" diffuse=".9 .9 .9"/><light pos="-2 1 3" diffuse=".5 .6 .7"/>
    <geom name="floor" type="plane" size="3 3 .1" rgba=".09 .13 .18 1"/>
    {rail}
    <geom name="head" type="ellipsoid" pos="0 0 .8" size=".204 .228 .276" rgba=".85 .69 .56 1"/>
    <geom name="nose" type="ellipsoid" pos="0 -.233 .775" size=".026 .045 .04" rgba=".9 .74 .6 1"/>
    <geom name="earL" type="ellipsoid" pos=".198 0 .79" size=".026 .035 .052" rgba=".85 .65 .52 1"/>
    <geom name="earR" type="ellipsoid" pos="-.198 0 .79" size=".026 .035 .052" rgba=".85 .65 .52 1"/>
    <geom type="ellipsoid" pos="-.07 -.215 .825" size=".024 .013 .010" rgba=".2 .24 .26 1" contype="0" conaffinity="0"/>
    <geom type="ellipsoid" pos=".07 -.215 .825" size=".024 .013 .010" rgba=".2 .24 .26 1" contype="0" conaffinity="0"/>
    <geom type="capsule" fromto="-.035 -.21 .685 .035 -.21 .685" size=".003" rgba=".44 .25 .21 1" contype="0" conaffinity="0"/>
    <geom name="jar_bottom" type="cylinder" pos="{SUPPLY_X} {SUPPLY_Y} .585" size=".048 .005" rgba=".65 .8 .88 1"/>
    {jar_walls}
    <geom name="reservoir_gel" type="cylinder" pos="{SUPPLY_X} {SUPPLY_Y} .63" size=".042 .04" rgba=".1 .85 .9 .85" contype="0" conaffinity="0"/>
    {guide_geoms}{electrodes}{gel}
    <body name="orbit"><joint name="orbit" type="hinge" axis="0 0 1" range="{ARC_MIN} {ARC_MAX}" limited="true"/>
      <body name="carriage" pos=".66 0 .12" quat="0 0 0 1">
      <geom name="base" type="box" size=".12 .10 .07" rgba=".15 .4 .5 1"/>
      <geom name="mast" type="capsule" fromto="0 0 .07 0 0 {SHOULDER_Z-.12}" size=".035" rgba=".65 .7 .75 1"/>
      <body name="shoulder_yaw" pos="0 0 {SHOULDER_Z-.12}"><joint name="shoulder_yaw" axis="0 0 1" range="-3.1416 3.1416"/>
      <inertial pos="0 0 0" mass=".1" diaginertia=".001 .001 .001"/>
      <body name="shoulder"><joint name="shoulder" axis="0 -1 0" range="-1.8 2.9"/>
        <geom name="upper" type="capsule" fromto="0 0 0 {L1} 0 0" size=".026" rgba=".1 .65 .75 1"/>
        <body name="elbow" pos="{L1} 0 0"><joint name="elbow" axis="0 -1 0" range="-2.9 2.9"/>
          <geom name="forearm" type="capsule" fromto="0 0 0 {L2} 0 0" size=".022" rgba=".7 .78 .83 1"/>
          <body name="wrist_yaw" pos="{L2} 0 0"><joint name="wrist_yaw" axis="0 0 1" range="-1.6 1.6"/>
          <inertial pos="0 0 0" mass=".05" diaginertia=".0001 .0001 .0001"/>
          <body name="wrist"><joint name="wrist" axis="0 -1 0" range="-3.1 3.1"/>
            <geom name="grip" type="box" pos=".012 0 -.028" size=".027 .025 .009" rgba=".12 .2 .25 1" group="3"/>
            <geom name="syringe" type="cylinder" pos=".057 0 0" quat=".70710678 0 .70710678 0" size=".015 .045" rgba=".7 .88 .95 .24" group="3"/>
            <geom name="barrel_flange" type="cylinder" pos=".012 0 0" quat=".70710678 0 .70710678 0" size=".025 .002" rgba=".75 .85 .9 1" group="3"/>
            <geom name="barrel_tip" type="cylinder" pos=".103 0 0" quat=".70710678 0 .70710678 0" size=".013 .003" rgba=".75 .85 .9 1" group="3"/>
            <geom name="barrel_gel" type="cylinder" pos=".096 0 0" quat=".70710678 0 .70710678 0" size=".0105 .00001" rgba=".08 .85 .9 0" contype="0" conaffinity="0" group="3"/>
            {''.join(f'<geom name="tick{i}" type="capsule" fromto="{x} -.015 -.006 {x} -.015 .006" size=".0005" rgba=".15 .25 .3 1" contype="0" conaffinity="0" group="3"/>' for i,x in enumerate(np.linspace(.023,.096,6)))}
            <geom name="nozzle" type="capsule" fromto=".10 0 0 .15 0 0" size=".004" rgba=".9 .93 .94 1" group="3"/>
            <camera name="syringe_detail" pos=".022 -.22 .044" xyaxes="1 0 0 0 .196116 .980581" fovy="34"/>
            <site name="tip" pos=".15 0 0" size=".005" rgba="1 1 1 1"/>
            <body name="plunger" pos=".096 0 0">
              <joint name="plunger_slide" type="slide" axis="-1 0 0" range="0 {PISTON_STROKE}" limited="true"/>
              <geom name="piston_seal" type="cylinder" quat=".70710678 0 .70710678 0" size=".0115 .0025" rgba=".08 .12 .16 1" group="3"/>
              <geom name="piston_rod" type="capsule" fromto="-.13 0 0 0 0 0" size=".003" rgba=".83 .89 .93 1" group="3"/>
              <geom name="piston_pad" type="cylinder" pos="-.13 0 0" quat=".70710678 0 .70710678 0" size=".021 .003" rgba=".93 .96 .98 1" group="3"/>
            </body>
          </body></body></body></body></body></body></body>
    </worldbody></mujoco>'''

def ik(point, normal, angle, clearance):
    """Spatial IK: shoulder and wrist yaw reach frontal sites from arc endpoints."""
    radial=np.array([np.cos(angle),np.sin(angle),0.])
    n=np.asarray(normal,dtype=float);n=n/np.linalg.norm(n)
    tip=np.asarray(point)+clearance*n;wrist=tip+TOOL*n
    delta=wrist-(RADIUS*radial+np.array([0.,0.,SHOULDER_Z]))
    heading=np.arctan2(delta[1],delta[0])
    yaw=(heading-angle-np.pi+np.pi)%(2*np.pi)-np.pi
    x=np.linalg.norm(delta[:2]);z=delta[2]
    c=(x*x+z*z-L1*L1-L2*L2)/(2*L1*L2)
    if abs(c)>1: raise RuntimeError('Unreachable target')
    elbow=-np.arccos(c)
    shoulder=np.arctan2(z,x)-np.arctan2(L2*np.sin(elbow),L1+L2*np.cos(elbow))
    beta=shoulder+elbow
    ch,sh=np.cos(heading),np.sin(heading);cb,sb=np.cos(beta),np.sin(beta)
    rotation=np.array([[ch*cb,-sh,-ch*sb],[sh*cb,ch,-sh*sb],[sb,0,cb]])
    local=rotation.T@(-n)
    wy=np.arctan2(local[1],local[0]);wp=np.arctan2(local[2],np.linalg.norm(local[:2]))
    if wy>np.pi/2:wy-=np.pi;wp=np.pi-wp if wp>=0 else -np.pi-wp
    if wy<-np.pi/2:wy+=np.pi;wp=np.pi-wp if wp>=0 else -np.pi-wp
    return np.array([angle,yaw,shoulder,elbow,wy,wp,0.]),tip

def run(out='output', fps=12, render=True, seed=42, electrodes=16, playback=2):
    import mujoco
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    points,normals,angles=electrode_layout(electrodes)
    initial,rates,names=electrode_conditions(electrodes)
    xml=xml_scene(points); (out/'scene.xml').write_text(xml,encoding='utf-8')
    model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
    dose_model=DoseModel(); rng=np.random.default_rng(seed)
    z=initial.copy(); measured=z+rng.normal(0,.08,electrodes)
    volumes=np.zeros(electrodes); attempts=np.zeros(electrodes,dtype=int); failed=np.zeros(electrodes,dtype=bool)
    scale=np.ones(electrodes); logs=[]; trajectory=[]; qhome=home_pose()
    data.qpos[:]=qhome; mujoco.mj_forward(model,data)
    tip_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'tip')
    robot_geoms=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,x) for x in ['base','mast','upper','forearm','grip','syringe','nozzle','barrel_flange','barrel_tip','piston_seal','piston_rod','piston_pad']]
    obstacles=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,x) for x in ['head','nose','earL','earR','jar_bottom']+[f'jar_wall{i}' for i in range(24)]]
    min_clearance=10.; max_tip_error=0.; max_velocity=0.; last_q=None
    supply_capacity=RESERVOIR_CAPACITY*electrodes/8
    frames=[]; action={}; syringe=0.; reservoir=supply_capacity; refills=[]
    orbit_travel=0.
    def append(q, phase_name, idx=-1, goal=None):
        nonlocal min_clearance,max_tip_error,last_q,max_velocity
        q=q.copy();q[-1]=piston_extension(syringe)
        data.qpos[:]=q; mujoco.mj_forward(model,data)
        for g in robot_geoms:
            for h in obstacles:
                distance=mujoco.mj_geomDistance(model,data,g,h,2.,None)
                min_clearance=min(min_clearance,distance)
                if distance < .004:
                    raise RuntimeError(f'Clearance violation: {distance:.4f} m, {phase_name}, geom {g}/{h}')
        if np.any(q<model.jnt_range[:,0]) or np.any(q>model.jnt_range[:,1]):
            raise RuntimeError('Joint limit exceeded')
        if goal is not None:
            err=float(np.linalg.norm(data.site_xpos[tip_id]-goal));max_tip_error=max(max_tip_error,err)
            if err>.003: raise RuntimeError(f'Tip tracking error {err:.4f}')
        if last_q is not None: max_velocity=max(max_velocity,float(np.max(np.abs(q[:-1]-last_q[:-1]))*fps))
        last_q=q.copy()
        frames.append((q.copy(),phase_name,idx,measured.copy(),volumes.copy(),failed.copy(),dict(action),syringe,reservoir))
        trajectory.append(q.tolist())
    def hold(seconds, phase_name, idx=-1):
        for _ in range(round(seconds*fps)): append(data.qpos.copy(),phase_name,idx)
    def interpolate(target,seconds,phase_name,idx=-1):
        start=data.qpos.copy()
        for t in np.linspace(0,1,round(seconds*fps)+1)[1:]:
            s=t*t*(3-2*t);append(start+(target-start)*s,phase_name,idx)
    def approach(idx,begin,end,seconds,phase_name):
        angle=float(data.qpos[0]);
        for t in np.linspace(0,1,round(seconds*fps)+1)[1:]:
            s=t*t*(3-2*t);q,goal=ik(points[idx],normals[idx],angle,begin+(end-begin)*s)
            append(q,phase_name,idx,goal)
    def work_lift(angle,reverse=False):
        radial=np.array([np.cos(angle),np.sin(angle),0.])
        low=(RADIUS-L1*np.cos(qhome[2])-L2*np.cos(qhome[2]+qhome[3]))*radial
        low[2]=SHOULDER_Z+L1*np.sin(qhome[2])+L2*np.sin(qhome[2]+qhome[3])+TOOL
        high=.58*radial;high[2]=1.20
        start,end=(high,low) if reverse else (low,high)
        for t in np.linspace(0,1,round(1.2*fps)+1)[1:]:
            pos=start+(end-start)*t*t*(3-2*t)
            q,goal=ik(pos,np.array([0.,0.,-1.]),angle,0.)
            append(q,'LOWER TO TRAVEL POSE' if reverse else 'LIFT TO CLEAR HEAD',goal=goal)
        return ik(high,np.array([0.,0.,-1.]),angle,0.)[0]
    def refill(current_angle):
        nonlocal syringe,reservoir,orbit_travel
        station_angle=SUPPLY_ANGLE
        park=qhome.copy();park[0]=current_angle
        interpolate(park,.9,'RETRACT FOR GEL REFILL')
        park[0]=station_angle
        travel=abs(station_angle-current_angle);orbit_travel+=2*travel
        interpolate(park,max(1.2,travel/.9),'MOVE TO GEL STATION')
        surface=.59+.08*reservoir/supply_capacity
        radial=np.array([np.cos(station_angle),np.sin(station_angle),0.])
        point=SUPPLY_RADIUS*radial;point[2]=surface;normal=np.array([0.,0.,1.])
        low_park_tip=data.site_xpos[tip_id].copy()
        def vertical_park(reverse=False):
            start=lifted_park_tip if reverse else low_park_tip
            end=low_park_tip if reverse else lifted_park_tip
            for t in np.linspace(0,1,round(1.2*fps)+1)[1:]:
                pos=start+(end-start)*t*t*(3-2*t)
                q,goal=ik(pos,np.array([0.,0.,-1.]),station_angle,0.)
                append(q,'LOWER OUTSIDE CONTAINER' if reverse else 'LIFT OUTSIDE CONTAINER',goal=goal)
        lifted_park_tip=.58*radial;lifted_park_tip[2]=1.20
        lifted_park,_=ik(lifted_park_tip,np.array([0.,0.,-1.]),station_angle,0.)
        vertical_park()
        raised_point=SUPPLY_RADIUS*radial;raised_point[2]=1.20
        raised,_=ik(raised_point,np.array([0.,0.,-1.]),station_angle,0.)
        interpolate(raised,1.2,'RAISE TOOL ABOVE GEL STATION')
        pre,_=ik(point,normal,station_angle,.10)
        interpolate(pre,1.2,'ALIGN OVER GEL CONTAINER')
        for t in np.linspace(0,1,fps+1)[1:]:
            q,goal=ik(point,normal,station_angle,.10+(.008-.10)*t*t*(3-2*t))
            append(q,'LOWER INTO GEL CONTAINER',goal=goal)
        amount=min(SYRINGE_CAPACITY-syringe,reservoir)
        if amount<=1e-8: raise RuntimeError('Gel reservoir depleted')
        old_syringe,old_reservoir=syringe,reservoir
        for t in np.linspace(0,1,2*fps+1)[1:]:
            syringe=old_syringe+amount*t;reservoir=old_reservoir-amount*t
            # Keep the nozzle just above the receding virtual gel surface.
            point[2]=.59+.08*reservoir/supply_capacity
            q,goal=ik(point,normal,station_angle,.008)
            append(q,'ASPIRATE GEL / REFILL',goal=goal)
        refills.append({'frame':len(frames),'drawn_ml':amount,'syringe_after_ml':syringe,'reservoir_after_ml':reservoir})
        for t in np.linspace(0,1,fps+1)[1:]:
            q,goal=ik(point,normal,station_angle,.008+(.10-.008)*t*t*(3-2*t))
            append(q,'LIFT FROM GEL CONTAINER',goal=goal)
        interpolate(raised,1.2,'RAISE TOOL FOR RETURN')
        interpolate(lifted_park,1.2,'MOVE OUTSIDE CONTAINER RIM')
        vertical_park(reverse=True)
        park[0]=station_angle;interpolate(park,1.2,'PARK SYRINGE FOR TRAVEL')
        park[0]=current_angle
        interpolate(park,max(1.2,travel/.9),'RETURN TO ELECTRODE SWEEP')
    hold(3,'INITIAL SCAN')
    angle=ARC_MIN; direction=1; max_actions=electrodes*4
    target_bases=np.array([base_for_target(p) for p in points])
    for visit in range(max_actions):
        candidates=np.flatnonzero((measured>THRESHOLD)&(~failed))
        if not len(candidates): break
        # Sweep along the open arc; reverse rather than wrapping across the face.
        delta=(target_bases[candidates]-angle)*direction
        available=delta>=-1e-7
        if not np.any(available):
            direction*=-1;delta=(target_bases[candidates]-angle)*direction;available=delta>=-1e-7
        eligible=candidates[available]
        idx=int(eligible[np.argmin(delta[available])]);next_angle=float(target_bases[idx])
        dose,pred=dose_model.choose(float(measured[idx]),float(scale[idx]))
        dose=min(dose,.48-volumes[idx])
        if syringe+1e-8<dose: refill(angle)
        if syringe+1e-8<dose: raise RuntimeError('Insufficient gel after refill')
        action={'dose_ml':round(dose,3),'pred_drop':round(pred,2),'before':round(float(measured[idx]),2)}
        park=qhome.copy();park[0]=angle
        interpolate(park,1.0,'RETRACT TO TRAVEL',idx)
        park[0]=next_angle
        interpolate(park,max(1.4,abs(next_angle-angle)/.65),'MOVE ALONG REAR 270 DEG ARC',idx)
        orbit_travel+=abs(next_angle-angle)
        angle=next_angle
        raised_work=work_lift(angle)
        pre,_=ik(points[idx],normals[idx],angle,.12)
        interpolate(pre,1.3,'ALIGN WITH ELECTRODE',idx)
        approach(idx,.12,.014,1.1,'APPROACH')
        old_syringe=syringe
        for t in np.linspace(0,1,round(.8*fps)+1)[1:]:
            syringe=max(0.,old_syringe-dose*t)
            append(data.qpos.copy(),'DISPENSE / PUSH PISTON',idx)
        before=float(measured[idx])
        z[idx]=response(z[idx],dose,rates[idx]);volumes[idx]+=dose;attempts[idx]+=1
        measured[idx]=z[idx]+rng.normal(0,.08)
        drop=before-measured[idx]
        if pred>.1: scale[idx]=float(np.clip(drop/pred,.15,1.6))
        if (attempts[idx]>=2 and drop<.5) or (volumes[idx]>=.479 and measured[idx]>THRESHOLD):failed[idx]=True
        action['after']=round(float(measured[idx]),2)
        hold(.8,'REMEASURE / UPDATE',idx)
        logs.append({'visit':visit+1,'electrode':names[idx],'base_angle_degrees':round(math.degrees(angle),1),'before_kohm':before,'dose_ml':dose,'predicted_drop_kohm':pred,'true_after_kohm':float(z[idx]),'measured_after_kohm':float(measured[idx]),'cumulative_ml':float(volumes[idx]),'syringe_remaining_ml':syringe,'reservoir_remaining_ml':reservoir,'status':'REVIEW' if failed[idx] else ('PASS' if measured[idx]<=THRESHOLD else 'REVISIT')})
        approach(idx,.014,.12,1.1,'WITHDRAW')
        interpolate(raised_work,1.2,'RAISE CLEAR OF HEAD')
        work_lift(angle,reverse=True)
    if np.any((measured>THRESHOLD)&(~failed)):
        raise RuntimeError('Action limit reached with untreated electrodes')
    park=qhome.copy();park[0]=angle;interpolate(park,1.3,'PARK')
    park[0]=SUPPLY_ANGLE;orbit_travel+=abs(SUPPLY_ANGLE-angle)
    interpolate(park,max(1.2,abs(SUPPLY_ANGLE-angle)/.9),'PARK BEHIND HEAD')
    hold(6,'FINISHED / REVIEW FLAGGED ELECTRODE')
    summary={'seed':seed,'electrodes':electrodes,'synthetic_only':True,'robot_control':'kinematic joint positioning, no actuator dynamics or real impedance sensor','model':'ridge regression, 1600 synthetic training and 500 independent synthetic test examples','synthetic_test_mae_kohm':dose_model.test_mae,'threshold_kohm':THRESHOLD,'initial_kohm':initial.tolist(),'final_measured_kohm':measured.tolist(),'passed':int(np.sum(measured<=THRESHOLD)),'review':[names[i] for i in np.flatnonzero(failed)],'visits':len(logs),'total_gel_ml':float(volumes.sum()),'final_base_angle_degrees':float(np.rad2deg(data.qpos[0])),'minimum_robot_to_manikin_clearance_m':float(min_clearance),'maximum_tip_position_error_m':float(max_tip_error),'maximum_joint_speed_rad_s':max_velocity,'frames':len(frames),'fps':fps,'video_seconds':len(frames)/fps}
    summary.update(refill_count=len(refills),syringe_capacity_ml=SYRINGE_CAPACITY,syringe_remaining_ml=syringe,reservoir_capacity_ml=supply_capacity,reservoir_remaining_ml=reservoir,drawn_gel_ml=supply_capacity-reservoir,total_base_travel_degrees=float(np.rad2deg(orbit_travel)),collision_check='Every output frame; robot vs enlarged head, nose, ears and refill-container walls/bottom. Self-collision and continuous swept volumes not certified.')
    summary.update(simulation_seconds=len(frames)/fps,playback_speed=playback,video_seconds=len(frames[::playback])/fps)
    summary.update(base_arc_degrees=270,front_exclusion_degrees=90,base_limits_degrees=[-45,225],base_observed_degrees=[float(np.rad2deg(np.min(np.array(trajectory)[:,0]))),float(np.rad2deg(np.max(np.array(trajectory)[:,0])))],supply_location='behind head',frontal_reach='from arc endpoints using shoulder and wrist yaw')
    summary.update(piston_model='prismatic joint linked to virtual syringe volume',piston_stroke_m=PISTON_STROKE,piston_extension_final_m=piston_extension(syringe))
    summary.update(channel_names=names,montage='Idealised landmark/arc-based 10-20 subset (16); interpolated additional 10-10 sites (32)',montage_source='https://pubmed.ncbi.nlm.nih.gov/28436791/')
    with (out/'electrode_positions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['channel','model_x_m','model_y_m','model_z_m'])
        for name,point in zip(names,points):w.writerow([name,*point.tolist()])
    if abs((supply_capacity-reservoir)-volumes.sum()-syringe)>1e-8:raise RuntimeError('Gel conservation error')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'refills.json').write_text(json.dumps(refills,indent=2),encoding='utf-8')
    with (out/'events.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=logs[0].keys());w.writeheader();w.writerows(logs)
    np.savez(out/'replay.npz',qpos=np.array(trajectory),weights=dose_model.w,syringe_ml=np.array([f[-2] for f in frames]),reservoir_ml=np.array([f[-1] for f in frames]),piston_extension_m=np.array(trajectory)[:,-1],phase=np.array([f[1] for f in frames]))
    if render: render_video(model,data,frames[::playback],out,fps,initial,names,supply_capacity,playback)
    print(json.dumps(summary,indent=2));return summary

def render_video(model,data,frames,out,fps,initial,names,supply_capacity,playback):
    import mujoco, imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    font_path=next((x for x in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(x).exists()),None)
    font=lambda n:ImageFont.truetype(font_path,n) if font_path else ImageFont.load_default()
    f18,f14,f12,f24=font(18),font(14),font(12),font(24)
    renderer=mujoco.Renderer(model,height=540,width=720)
    detail_renderer=mujoco.Renderer(model,height=120,width=288)
    detail_option=mujoco.MjvOption();detail_option.geomgroup[:]=0;detail_option.geomgroup[3]=1;detail_option.sitegroup[:]=0
    camera=mujoco.MjvCamera(); camera.lookat[:]=[0,.03,.80];camera.distance=1.75;camera.azimuth=185;camera.elevation=-30
    count=len(names)
    points,_,_=electrode_layout(count)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'e{i}') for i in range(count)]
    gel_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'gel{i}') for i in range(count)]
    reservoir_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'reservoir_gel')
    barrel_gel_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'barrel_gel')
    writer=imageio.get_writer(str(out/'eeg_gel_demo.mp4'),fps=fps,codec='libx264',quality=8,macro_block_size=16)
    snapshots={0:'preview_initial.png',len(frames)//2:'preview_action.png',len(frames)-1:'preview_final.png'}
    refill_snapshot_saved=False
    try:
        for frame,(q,phase,idx,z,vol,failed,action,syringe,reservoir) in enumerate(frames):
            height=.08*reservoir/supply_capacity
            model.geom_pos[reservoir_id,2]=.59+height/2
            model.geom_size[reservoir_id,1]=max(.001,height/2)
            fill=piston_extension(syringe)
            model.geom_pos[barrel_gel_id,0]=.096-fill/2
            model.geom_size[barrel_gel_id,1]=max(.00001,fill/2)
            model.geom_rgba[barrel_gel_id,3]=.88 if fill>1e-8 else 0.
            data.qpos[:]=q;mujoco.mj_forward(model,data)
            for i,g in enumerate(ids):model.geom_rgba[g]=[.99,.65,.16,1] if failed[i] else ([.15,.83,.57,1] if z[i]<=THRESHOLD else [.97,.27,.27,1])
            if idx>=0: model.geom_rgba[ids[idx]]=[1,1,.65,1]
            for i,g in enumerate(gel_ids):
                model.geom_rgba[g,3]=.8 if vol[i]>0 else 0
                model.geom_size[g,0]=.003+.003*np.sqrt(vol[i]/.48)
            renderer.update_scene(data,camera=camera)
            if frame==0:
                for _ in range(3):renderer.render()
            image=Image.new('RGB',(1024,672),(14,22,33));image.paste(Image.fromarray(renderer.render()),(0,76))
            label_head(image,renderer.scene,points,names,f12,(0,76))
            d=ImageDraw.Draw(image)
            draw_topomap(d,points,names,z,failed,idx,q[0],font(11))
            detail_renderer.update_scene(data,camera='syringe_detail',scene_option=detail_option)
            if frame==0:
                for _ in range(3):detail_renderer.render()
            d.rounded_rectangle((418,88,713,258),radius=7,fill='#152332',outline='#365464',width=1)
            d.text((431,95),'SYRINGE | 3D CLOSE-UP',font=f12,fill='#cadbe7')
            image.paste(Image.fromarray(detail_renderer.render()),(422,113))
            pulling='ASPIRATE' in phase;pushing=phase.startswith('DISPENSE')
            mode='PULL / ASPIRATE' if pulling else 'PUSH / DISPENSE' if pushing else 'PISTON HELD'
            d.text((431,239),mode,font=f12,fill='#6ce2ee')
            d.text((632,239),f'{syringe:.2f} mL',font=f12,fill='white')
            if pulling or pushing:
                x0,x1=(554,502) if pulling else (502,554)
                d.line((x0,218,x1,218),fill='#6ce2ee',width=2)
                sign=1 if x1>x0 else -1
                d.polygon([(x1,218),(x1-8*sign,213),(x1-8*sign,223)],fill='#6ce2ee')
            d.text((24,14),f'EEG GEL ASSIST | {count} ELECTRODES',font=f24,fill='#f2f6fb')
            d.text((24,46),f'REAR 270 DEG BASE ARC | SIMULATION ONLY | {playback}x PLAYBACK',font=f12,fill='#8ea6ba')
            d.rectangle((720,76,1024,672),fill='#152332')
            d.text((738,91),'ELECTRODE IMPEDANCE',font=f18,fill='white')
            d.text((738,119),'Initial -> current (simulated kOhm)',font=f12,fill='#9bb0c5')
            columns=2 if count>16 else 1
            row_height=21 if count>=16 else 42
            for i in range(count):
                col,row=divmod(i,16) if columns==2 else (0,i)
                x=738+col*143;y=146+row*row_height
                color='#ffb547' if failed[i] else ('#35d499' if z[i]<=THRESHOLD else '#fa5c62')
                if idx==i:d.rounded_rectangle((x-5,y-2,x+(138 if columns==2 else 277),y+19),radius=3,fill='#26394c')
                d.text((x,y),names[i],font=f12,fill='white')
                value=f'{z[i]:.1f}' if columns==2 else f'{initial[i]:.0f} -> {z[i]:.1f}'
                d.text((x+35,y),value,font=f12,fill=color)
                status='CHK' if failed[i] else ('OK' if z[i]<=THRESHOLD else 'HIGH')
                d.text((x+(100 if columns==2 else 207),y),status,font=f12,fill=color)
                d.rectangle((x+35,y+17,x+35+min(60 if columns==2 else 145,z[i]/35*(60 if columns==2 else 145)),y+18),fill=color)
            y=505
            d.text((738,y),f'PASS {sum(z<=THRESHOLD)}/{count}   REVIEW {sum(failed)}',font=f18,fill='white')
            d.text((738,y+29),f'Total gel: {sum(vol):.2f} mL (virtual)',font=f14,fill='#9bb0c5')
            d.text((738,y+55),f'Syringe: {syringe:.2f} / {SYRINGE_CAPACITY:.2f} mL',font=f14,fill='#6ce2ee')
            d.rectangle((739,y+79,1003,y+88),fill='#30495b')
            d.rectangle((739,y+79,739+264*max(0,syringe)/SYRINGE_CAPACITY,y+88),fill='#36cdd2')
            d.text((738,y+99),f'Reservoir: {reservoir:.2f} mL',font=f14,fill='#9bb0c5')
            d.text((738,y+123),'Demo target <= 10 (not clinical)',font=f12,fill='#9bb0c5')
            d.rectangle((0,616,720,672),fill='#102a36')
            label=f'{names[idx]} | {phase}' if idx>=0 else phase
            d.text((18,622),label,font=f18,fill='#d7f7f5')
            detail=f'Model dose: {action.get("dose_ml",0):.2f} mL | Predicted drop: {action.get("pred_drop",0):.1f} kOhm'
            if idx<0:detail=f'Syringe: {syringe:.2f} mL | Gel reservoir: {reservoir:.2f} mL'
            d.text((18,648),detail,font=f12,fill='#9bbcc7')
            d.text((590,18),f'{frame*playback/fps:05.1f}s sim',font=f14,fill='white')
            if frame in snapshots:image.save(out/snapshots[frame])
            if len(frames)<=12:image.save(out/f'preview_detail_{frame:02d}.png')
            if phase=='ASPIRATE GEL / REFILL' and not refill_snapshot_saved:
                image.save(out/'preview_refill.png');refill_snapshot_saved=True
            writer.append_data(np.asarray(image))
            if frame%(fps*20)==0:print(f'Rendered {frame}/{len(frames)} frames',flush=True)
    finally:writer.close();detail_renderer.close();renderer.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='output');parser.add_argument('--fps',type=int,default=12);parser.add_argument('--no-video',action='store_true');parser.add_argument('--seed',type=int,default=42);parser.add_argument('--electrodes',type=int,choices=[8,16,32],default=16);parser.add_argument('--playback',type=int,choices=[1,2],default=2)
    args=parser.parse_args();run(args.out,args.fps,not args.no_video,args.seed,args.electrodes,args.playback)
