import os
import numpy as np
import torch
import torch.optim as optim
from geometry import compute_operators

def read_surf(fname):
    """
    Read FreeSurfer's surface.

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    vertex_coords : 2D array, shape = [n_vertex, 3]
        Vertex coordinates.
    faces : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    """

    TRIANGLE_FILE_MAGIC_NUMBER = 0xFFFFFE
    QUAD_FILE_MAGIC_NUMBER = 0xFFFFFF
    NEW_QUAD_FILE_MAGIC_NUMBER = 0xFFFFFD

    with open(fname, "rb") as f:
        h0, h1, h2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
        magic = (h0 << 16) + (h1 << 8) + h2

        if (magic == QUAD_FILE_MAGIC_NUMBER) | (magic == NEW_QUAD_FILE_MAGIC_NUMBER):
            # need to be verified
            h0, h1, h2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
            vnum = (h0 << 16) + (h1 << 8) + h2

            h0, h1, h2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
            fnum = (h0 << 16) + (h1 << 8) + h2

            vertex_coords = np.fromfile(f, dtype=np.dtype(">i2"), count=3 * vnum) / 100
            vertex_coords = vertex_coords.reshape(-1, 3)
            arr = np.fromfile(f, dtype=np.dtype("B"), count=9 * fnum)
            faces = (arr[0::3] << 16) + (arr[0::3] << 8) + arr[0::3]
            faces = faces.reshape(-1, 3)

            return vertex_coords, faces

        elif magic == TRIANGLE_FILE_MAGIC_NUMBER:
            f.readline()
            f.readline().strip()

            vnum, fnum = np.fromfile(f, dtype=np.dtype(">i4"), count=2)
            vertex_coords = np.fromfile(f, dtype=np.dtype(">f4"), count=3 * vnum)
            faces = np.fromfile(f, dtype=np.dtype(">i4"), count=3 * fnum)

            vertex_coords = vertex_coords.reshape(vnum, 3)
            faces = faces.reshape(fnum, 3)

            return vertex_coords, faces

        else:
            raise Exception("SurfReaderError: unknown format!")

def read_vtk(fname):
    """
    Read a vtk file (ASCII version).

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the the input mesh.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    """

    with open(fname, "rb") as fd:
        lines = iter(l for l in fd)

        ver = next(d for d in lines if b"Version" in d)
        ver = float(ver.split()[-1])

        nVert = next(d for d in lines if b"POINTS" in d)
        nVert = int(nVert.split()[1])
        v = np.fromfile(fd, dtype=float, count=nVert * 3, sep=" ").reshape(nVert, 3)

        nFace = next(d for d in lines if b"POLYGONS" in d)
        nFace = int(nFace.split()[1])
        if ver < 5:
            f = np.fromfile(fd, dtype=int, count=nFace * 4, sep=" ").reshape(nFace, 4)
            f = f[:, 1:]
        else:
            nFace -= 1
            next(d for d in lines if b"CONNECTIVITY" in d)
            f = np.fromfile(fd, dtype=int, count=nFace * 3, sep=" ").reshape(nFace, 3)

    return v, f

data_dir = "/data/human/Mindboggle/FreeSurfer/scan"
label_dir = "/data/human/Mindboggle/labels"
subjs = os.listdir(data_dir)
for subj in subjs:
    white_dir = os.path.join(data_dir, subj, "surf", "lh.white")
    white_v, white_f = read_surf(white_dir)
    white_v = torch.from_numpy(white_v.astype(np.float32))
    white_f = torch.from_numpy(white_f.astype(np.int32))
    _, mass, L, evals, evecs, gradX, gradY = compute_operators(white_v, white_f)
    parc_dir = os.path.join(label_dir, subj+".lh.parc.txt")
    label = torch.from_numpy(np.loadtxt(parc_dir).astype(np.int16))
    d = dict()
    d["vertices"] = white_v
    d["label"] = label
    d["massvec"] = mass
    d["evals"] = evals
    d["evecs"] = evecs
    d["gradX"] = gradX
    d["gradY"] = gradY
    torch.save(d, os.path.join("/data/lfs/kimjongmin8/Develop/CortexDiffusion/Mindboggle_dataset",subj+".pt"))