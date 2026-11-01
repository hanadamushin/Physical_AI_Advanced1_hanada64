"""Check feedback behavior and geometric clearance with independent noise seeds."""
import csv, json
import numpy as np
from pathlib import Path
from demo import run, THRESHOLD, channel_names, electrode_layout, HS, HC, montage_geometry, arc_sample, topomap_xy, ARC_MIN, ARC_MAX, base_for_target, ik, xml_scene, PISTON_STROKE, SYRINGE_CAPACITY

def verify():
    root=Path('verification_v07')
    xyz,_,curves=montage_geometry()
    for curve,names in [('sagittal',['Fpz','Fz','Cz','Pz','Oz']),('coronal',['T7','C3','Cz','C4','T8'])]:
        expected=arc_sample(curves[curve],[.1,.3,.5,.7,.9])
        assert np.max(np.linalg.norm(expected-np.array([xyz[n] for n in names]),axis=1))<1e-5
    for side in [1,2]:
        expected=arc_sample(curves['lateral_left' if side==1 else 'lateral_right'],[.25,.5,.75])
        assert np.allclose(expected,[xyz[n] for n in (['F3','C3','P3'] if side==1 else ['F4','C4','P4'])],atol=1e-5)
    for left,right in [('Fp1','Fp2'),('F7','F8'),('F3','F4'),('T7','T8'),('C3','C4'),('P3','P4'),('P7','P8'),('O1','O2')]:
        assert np.allclose(xyz[left]*[-1,1,1],xyz[right],atol=1e-5)
        assert xyz[left][0]>0 and xyz[right][0]<0
    assert xyz['Fp1'][2]>.825 and xyz['T7'][2]>.842
    assert xyz['C3'][2]>xyz['T7'][2] and xyz['F3'][2]>xyz['Fp1'][2]
    mapped=topomap_xy(np.array([xyz['Fp1'],xyz['Fp2'],xyz['O1']]))
    assert mapped[0,0]<0<mapped[1,0] and mapped[0,1]<0<mapped[2,1]
    results=[]
    for count,seed in [(16,42),(16,43),(32,42)]:
        dest=root/f'{count}_{seed}'
        summary=run(dest,render=False,seed=seed,electrodes=count)
        names=channel_names(count)
        assert summary['passed']==count-1 and summary['review']==['T7']
        points,normals,angles=electrode_layout(count)
        assert np.allclose(np.sum(((points-HC)/HS)**2,axis=1),1.)
        for name,point in zip(names,points):
            if name.endswith('1') or name in ['F3','F7','C3','T7','P3','P7']:assert point[0]>0
            if name.startswith('Fp'):assert point[1]<0
            if name in ['O1','O2']:assert point[1]>0
        assert len(set(names))==count
        assert summary['minimum_robot_to_manikin_clearance_m']>=.004
        assert summary['maximum_tip_position_error_m']<.003
        if count==16:assert 60<=summary['video_seconds']<=180
        assert summary['refill_count']>=count//4
        assert abs(summary['drawn_gel_ml']-summary['total_gel_ml']-summary['syringe_remaining_ml'])<1e-8
        replay=np.load(dest/'replay.npz')
        assert np.all(replay['qpos'][:,0]>=ARC_MIN-1e-9) and np.all(replay['qpos'][:,0]<=ARC_MAX+1e-9)
        assert replay['qpos'].shape[1]==7
        extension=replay['piston_extension_m']
        assert np.allclose(extension,PISTON_STROKE*replay['syringe_ml']/SYRINGE_CAPACITY,atol=1e-10)
        assert np.allclose(extension,replay['qpos'][:,-1])
        assert extension.min()>=0 and extension.max()<=PISTON_STROKE+1e-10
        movement=np.diff(extension);phase=replay['phase'][1:]
        pulling=np.char.find(phase,'ASPIRATE')>=0;pushing=np.char.startswith(phase,'DISPENSE')
        assert np.all(movement[pulling]>=-1e-10) and np.any(movement[pulling]>1e-5)
        assert np.all(movement[pushing]<=1e-10) and np.any(movement[pushing]<-1e-5)
        assert np.all(np.abs(movement[~(pulling|pushing)])<1e-10)
        assert np.all(replay['syringe_ml']>=-1e-8) and np.all(replay['syringe_ml']<=.48000001)
        assert np.all(replay['reservoir_ml']>=0) and np.all(np.diff(replay['reservoir_ml'])<=1e-8)
        assert replay['syringe_ml'][0]==0
        rows=list(csv.DictReader((dest/'events.csv').open(encoding='utf-8-sig')))
        assert not any(r['electrode'] in [names[i] for i,z in enumerate(summary['initial_kohm']) if z<=THRESHOLD] for r in rows)
        assert all(0<float(r['dose_ml'])<=.16 for r in rows)
        assert all(float(r['cumulative_ml'])<=.48001 for r in rows)
        e7=[r for r in rows if r['electrode']=='T7']
        assert len(e7)==2 and e7[-1]['status']=='REVIEW'
        results.append({'electrodes':count,'seed':seed,'passed':True,'visits':summary['visits']})
        # Every site, including initially good ones, must be reachable with
        # both tip position and nozzle direction preserved from the allowed arc.
        import mujoco
        model=mujoco.MjModel.from_xml_string(xml_scene(points));data=mujoco.MjData(model)
        joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'plunger_slide')
        assert model.jnt_type[joint]==mujoco.mjtJoint.mjJNT_SLIDE
        sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'tip')
        for point,normal in zip(points,normals):
            angle=base_for_target(point);assert ARC_MIN<=angle<=ARC_MAX
            q,goal=ik(point,normal,angle,.014);data.qpos[:]=q;mujoco.mj_forward(model,data)
            assert np.linalg.norm(data.site_xpos[sid]-goal)<1e-6
            assert np.dot(data.site_xmat[sid].reshape(3,3)[:,0],-normal)>1-1e-8
    # Exercise trajectories to initially healthy sites as well, including Fp1.
    import demo
    original_conditions=demo.electrode_conditions
    def all_high(count):
        _,rates,names=original_conditions(count)
        return np.full(count,24.),rates,names
    try:
        demo.electrode_conditions=all_high
        extra=demo.run(root/'all_high_16',render=False,electrodes=16)
        assert extra['passed']==15 and extra['review']==['T7']
        events=list(csv.DictReader((root/'all_high_16/events.csv').open(encoding='utf-8-sig')))
        assert {r['electrode'] for r in events}==set(channel_names(16))
        results.append({'electrodes':16,'all_initially_high':True,'passed':True,'visits':extra['visits']})
    finally:demo.electrode_conditions=original_conditions
    (root/'checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('All electrode-count and feedback scenarios passed.')

if __name__=='__main__':verify()
