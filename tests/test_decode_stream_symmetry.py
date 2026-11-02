import numpy as np, re
from pathlib import Path
from hatTrick.decode.stream import decode_stream
from hatTrick._helpers import generate_s_matrix
def chunk(fname, refl):
    s=f"----- Begin chunk -----\nImage filename: {fname}\nEvent: //0\nImage serial number: 1\nhit = 1\n--- Begin crystal\nCell parameters 6.2 6.2 11.1 nm, 90 90 120 deg\nnum_reflections = {len(refl)}\nReflections measured after indexing\n   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel\n"
    for (h,k,l),I in refl.items(): s+=f"{h:4d} {k:4d} {l:4d} {I:10.2f} {1.0:10.2f} {1.0:10.2f} {1.0:10.2f} {1.0:6.1f} {1.0:6.1f} p0\n"
    return s+"End of reflections\n--- End crystal\n----- End chunk -----\n"
def test_symmetry_alignment(tmp_path):
    rng=np.random.default_rng(0); hkls=[(h,k,l) for h in range(1,6) for k in range(0,5) for l in range(1,4)]
    truth={x:rng.uniform(50,500,3) for x in hkls}; S=generate_s_matrix(3); tags=["".join(map(str,r)) for r in S]
    ops=[lambda h,k,l:(h,k,l), lambda h,k,l:(-k,h+k,l), lambda h,k,l:(-h,-k,-l)]   # pattern p relabelled by op p
    txt="CrystFEL stream format 2.3\n"
    for p,t in enumerate(tags):
        txt+=chunk(f"run_{t}.h5",{ops[p](*x):float(S[p]@truth[x]) for x in hkls})
    (tmp_path/"in.stream").write_text(txt)
    st0=decode_stream(str(tmp_path/"in.stream"),str(tmp_path/"o0.stream"),3)
    st=decode_stream(str(tmp_path/"in.stream"),str(tmp_path/"o.stream"),3,symmetry="6/m")
    assert st0["decoded"]==0 and st["decoded"]==1 and st["refl_decoded"]==len(hkls)
    out=Path(tmp_path/"o.stream").read_text().split("----- Begin chunk -----\n")[1:]
    for f,c in enumerate(out):
        for line in c.split("Reflections measured after indexing\n")[1].split("End of reflections")[0].splitlines()[1:]:
            p=line.split(); assert abs(float(p[3])-truth[(int(p[0]),int(p[1]),int(p[2]))][f])<0.05
