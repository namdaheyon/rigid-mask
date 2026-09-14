#!/usr/bin/env python3
"""Offline checks of saved VCN flow against image evidence. No ROS or model loads.

LK is an independent consistency check, NOT ground-truth motion. Photometric
errors can come from illumination/occlusion. D/S regions are model predictions.
All pixel errors below are in the saved network grid, not full-frame pixels.
"""
import argparse
import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import yaml


def static_depth_check(p, q, cnn_tau, camera_matrix, rotation_10, translation_10):
    """Two-view triangulation under a static-scene hypothesis, not depth truth."""
    if len(p) == 0:
        return dict(reason='no_reliable_LK_matches', accepted=0)
    inverse = np.linalg.inv(camera_matrix)
    rays0 = np.c_[p, np.ones(len(p))] @ inverse.T
    rays1 = np.c_[q, np.ones(len(q))] @ inverse.T
    P0 = np.c_[np.eye(3), np.zeros(3)]
    P1 = np.c_[rotation_10, translation_10]
    homogeneous = cv2.triangulatePoints(P0, P1, rays0[:,:2].T, rays1[:,:2].T)
    safe = np.abs(homogeneous[3]) > 1e-10
    xyz0 = np.full((len(p),3),np.nan)
    xyz0[safe] = (homogeneous[:3,safe]/homogeneous[3,safe]).T
    xyz1 = xyz0 @ rotation_10.T + translation_10
    with np.errstate(divide='ignore',invalid='ignore'):
        uv0 = xyz0 @ camera_matrix.T; uv0 = uv0[:,:2]/uv0[:,2:]
        uv1 = xyz1 @ camera_matrix.T; uv1 = uv1[:,:2]/uv1[:,2:]
        ratio = xyz0[:,2]/xyz1[:,2]
    reprojection = np.maximum(np.linalg.norm(uv0-p,axis=1),np.linalg.norm(uv1-q,axis=1))
    r0 = rays0/np.linalg.norm(rays0,axis=1,keepdims=True)
    r1 = rays1 @ rotation_10
    r1 /= np.linalg.norm(r1,axis=1,keepdims=True)
    parallax = np.arctan2(np.linalg.norm(np.cross(r0,r1),axis=1),np.sum(r0*r1,axis=1))
    good = safe & np.isfinite(ratio) & np.isfinite(cnn_tau) & (cnn_tau>0)
    good &= (xyz0[:,2]>.1)&(xyz1[:,2]>.1)&(reprojection<=1.)&(parallax>=.001)
    return dict(reason='static_hypothesis_only', matches=len(p), accepted=int(good.sum()),
                min_parallax_rad=.001,max_reprojection_px=1.,
                triangulated_Z0_over_Z1=percentiles(ratio[good]),
                cnn_Z0_over_Z1=percentiles(cnn_tau[good]),
                absolute_log_ratio_error=percentiles(np.abs(np.log(cnn_tau[good]/ratio[good]))),
                reprojection_px=percentiles(reprojection[good]),
                warning='Depends on static ROI, LK and planar vehicle pose; not independent metric depth ground truth')


def camera_motion(data, lidar_origin, width, height):
    metadata=json.loads(str(data['metadata_json']));pair=metadata['ego_pair']
    if pair.get('reason')!='valid':raise ValueError('Vehicle pose not bracketed')
    a,b=pair['previous_xy_yaw'],pair['current_xy_yaw']
    if a[2] is None or b[2] is None:raise ValueError('Missing yaw')
    a,b=np.asarray(a),np.asarray(b)
    if not np.isfinite([a,b]).all():raise ValueError('Nonfinite vehicle pose')
    if np.linalg.norm(b[:2]-a[:2])<.05:raise ValueError('Less than 5 cm vehicle translation')
    def rz(x):
        c,s=np.cos(x),np.sin(x)
        return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    cal=yaml.safe_load(str(data['calibration_yaml']))['cameras'][str(metadata['camera_id'])]
    R=np.asarray(cal['rotation_matrix']);t=np.asarray(cal['translation_vector'])
    offset=np.asarray(lidar_origin)-R.T@t
    R01=R@rz(b[2]-a[2])@R.T
    displacement=R@(rz(a[2]).T@np.r_[b[:2]-a[:2],0.]+(rz(b[2]-a[2])-np.eye(3))@offset)
    K=np.asarray(data['camera_matrix'],dtype=float).copy()
    original_h,original_w=data['previous_bgr'].shape[:2]
    K[0]*=width/original_w;K[1]*=height/original_h
    return K,R01.T,-R01.T@displacement


def final_term_audit(data, roi=None):
    if 'foreground_contributions' not in data:
        return dict(reason='spatial_contributions_missing')
    import torch
    terms=torch.from_numpy(np.array(data['foreground_contributions'],copy=True))
    residual=torch.from_numpy(np.array(data['foreground_residual'],copy=True))
    h,w=data['gated_mask'].shape
    selection=data['gated_mask']!=127
    if roi is not None:
        # Manually selected region: include UNKNOWN to avoid selecting only
        # confident model predictions when assessing a static object.
        selection=np.zeros((h,w),dtype=bool)
        x0,y0,x1,y1=roi;selection[y0:y1,x0:x1]=True
    names=['symmetric_transfer','epipolar','angular_2d','distance_3d','angular_3d','depth_contrast']
    variants={'original':terms.sum(1,keepdim=True)+residual,
              'without_residual':terms.sum(1,keepdim=True)}
    for i,name in enumerate(names):variants['without_'+name]=variants['original']-terms[:,i:i+1]
    nh,nw=data['vcn_flow_x_network_px'].shape[-2:]
    result={'warning':'Frozen final-term ablation, NOT full CNN re-inference or a proposed fix',
            'selection':'manual ROI' if roi else 'LiDAR-supported confident pixels','variants':{}}
    for name,logits in variants.items():
        p=torch.nn.functional.interpolate(logits,(nh,nw),mode='bilinear',align_corners=False).sigmoid()[0,0].numpy()
        p=cv2.resize(p,(w,h))[selection]
        result['variants'][name]=dict(pixels=int(p.size),probability=percentiles(p),
                                      dynamic_fraction=float(np.mean(p>=.6)) if p.size else None)
    return result


def percentiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(count=int(values.size), p05_p50_p95=np.percentile(values,[5,50,95]).tolist()
                if values.size else None)


def audit(data, roi=None, assume_static=False, lidar_origin=None):
    if assume_static and (roi is None or lidar_origin is None):
        raise ValueError('Static depth check requires a manually verified ROI and LiDAR origin')
    if lidar_origin is not None and (len(lidar_origin)!=3 or not np.isfinite(lidar_origin).all()):
        raise ValueError('LiDAR origin must contain three finite coordinates')
    fx, fy = data['vcn_flow_x_network_px'][0], data['vcn_flow_y_network_px'][0]
    h, w = fx.shape
    previous = cv2.resize(data['previous_bgr'], (w,h))
    current = cv2.resize(data['current_bgr'], (w,h))
    gray0 = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(data['gated_mask'], (w,h), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.mgrid[:h,:w].astype(np.float32)
    mapx, mapy = xx+fx, yy+fy
    valid = np.isfinite(mapx)&np.isfinite(mapy)&(mapx>=0)&(mapx<w-1)&(mapy>=0)&(mapy<h-1)
    roi_mask = np.ones((h,w),dtype=bool)
    if roi is not None:
        oh,ow=data['previous_bgr'].shape[:2]
        x0,y0,x1,y1=roi
        if not 0<=x0<x1<=ow or not 0<=y0<y1<=oh:
            raise ValueError('ROI must lie inside original full frame')
        roi_mask = (xx*ow/w>=x0)&(xx*ow/w<x1)&(yy*oh/h>=y0)&(yy*oh/h<y1)
    warped = cv2.remap(gray1,np.where(valid,mapx,0).astype(np.float32),
                       np.where(valid,mapy,0).astype(np.float32),cv2.INTER_LINEAR)
    error = np.abs(warped.astype(float)-gray0.astype(float))
    tau = cv2.resize(data['depth_ratio_Z0_over_Z1'][0],(w,h))
    report = dict(units='network pixels; photometric error in grayscale 0..255',
                  warning='Consistency checks, not ground truth; D/S are model labels', regions={})
    feature_mask = (roi_mask if assume_static else ((mask!=127)&roi_mask)).astype(np.uint8)*255
    points = cv2.goodFeaturesToTrack(gray0,maxCorners=500,qualityLevel=.01,minDistance=5,mask=feature_mask)
    report['lk'] = dict(detected=0 if points is None else len(points), accepted=0)
    if points is not None:
        following, forward_status, _ = cv2.calcOpticalFlowPyrLK(gray0,gray1,points,None)
        if following is not None:
            back, backward_status, _ = cv2.calcOpticalFlowPyrLK(gray1,gray0,following,None)
            if back is not None:
                p, q = points[:,0], following[:,0]
                fb = np.linalg.norm(back[:,0]-p,axis=1)
                keep = (forward_status[:,0]!=0)&(backward_status[:,0]!=0)&(fb<=1.)
                keep &= np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
                x,y = p[:,0].reshape(-1,1),p[:,1].reshape(-1,1)
                cnn = np.column_stack([cv2.remap(f,x,y,cv2.INTER_LINEAR)[:,0] for f in [fx,fy]])
                keep &= np.isfinite(cnn).all(1)
                discrepancy=np.linalg.norm(cnn-(q-p),axis=1)
                labels=mask[np.rint(p[:,1]).astype(int),np.rint(p[:,0]).astype(int)]
                report['lk'] = dict(detected=len(points),accepted=int(keep.sum()),fb_limit_px=1.,
                    cnn_minus_lk_px=percentiles(discrepancy[keep]),
                    dynamic=percentiles(discrepancy[keep&(labels==255)]),
                    static=percentiles(discrepancy[keep&(labels==0)]))
                if assume_static:
                    try:
                        K,R10,t10=camera_motion(data,lidar_origin,w,h)
                        cnn_tau=cv2.remap(tau,x,y,cv2.INTER_LINEAR)[:,0]
                        report['static_depth_check']=static_depth_check(p[keep],q[keep],cnn_tau[keep],K,R10,t10)
                    except ValueError as error:
                        report['static_depth_check']=dict(reason=str(error),accepted=0)
    for name, selection in [('dynamic',mask==255),('static',mask==0)]:
        selected=selection&roi_mask
        usable=selected&valid
        report['regions'][name] = dict(pixels=int(selected.sum()),
            flow_endpoint_outside_or_nonfinite=int((selected&~valid).sum()),
            photometric_error=percentiles(error[usable]),
            flow_magnitude=percentiles(np.hypot(fx,fy)[usable]),
            depth_ratio_Z0_over_Z1=percentiles(tau[selected]))
    if assume_static:
        report.setdefault('static_depth_check',dict(reason='no_reliable_LK_matches',accepted=0))
    report['final_term_audit']=final_term_audit(data,roi)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures',nargs='+',type=Path)
    parser.add_argument('--roi',nargs=4,type=int,metavar=('X0','Y0','X1','Y1'),help='Optional full-frame rectangle')
    parser.add_argument('--assume-static',action='store_true',help='User-verified static ROI; enables triangulated depth-ratio check')
    parser.add_argument('--lidar-origin-vehicle',nargs=3,type=float,metavar=('X','Y','Z'))
    args=parser.parse_args()
    if len(args.captures)>30:parser.error('At most 30 captures per run')
    output=Path(tempfile.mkdtemp(prefix='rigid_flow_audit_'))
    print('OUTPUT',output,flush=True)
    for index,path in enumerate(args.captures):
        with np.load(path,allow_pickle=False) as data:
            report=audit(data,args.roi,args.assume_static,args.lidar_origin_vehicle)
            report.update(capture=str(path.resolve()),metadata=json.loads(str(data['metadata_json'])),roi=args.roi,
                          assume_static=args.assume_static,lidar_origin_vehicle=args.lidar_origin_vehicle)
        with open(output/f'{index:02d}.json','x',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
        print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
