import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
def rot(axis, ang):
    a=np.asarray(axis,float); a/=np.linalg.norm(a); c,s=np.cos(ang),np.sin(ang); x,y,z=a
    return np.array([[c+x*x*(1-c),x*y*(1-c)-z*s,x*z*(1-c)+y*s],[y*x*(1-c)+z*s,c+y*y*(1-c),y*z*(1-c)-x*s],[z*x*(1-c)-y*s,z*y*(1-c)+x*s,c+z*z*(1-c)]])
# frame F columns: d (pipe dir), e1 (bend-plane normal reference), e2
def key(F): return tuple(np.round(F,6).flatten())
def dkey(F): return tuple(np.round(F[:,0],6))
start=np.eye(3)
frames={key(start):start}; front=[start]
for k in range(1,6):
    new=[]
    for F in front:
        d,e1,e2=F[:,0],F[:,1],F[:,2]
        for r in range(4):              # roll multiples of 90 about d
            G=rot(d,r*np.pi/2)@F
            for b in (np.pi/4,np.pi/2):  # bend about e2 of rolled frame
                H=rot(G[:,2],b)@G
                kk=key(H)
                if kk not in frames: frames[kk]=H; new.append(H)
    front=new
    ds={dkey(F) for F in frames.values()}
    offgrid=[d for d in ds if not all(abs(abs(v)-t)<1e-5 for v in d for t in [0] ) and sorted(np.round(np.abs(d),4)) not in ([0,0,1],[0,0.7071,0.7071])]
    print(f"弯头数≤{k}: 方向数={len(ds)}, 非'轴/面对角'方向数={len(offgrid)}", sorted(offgrid)[:2])
