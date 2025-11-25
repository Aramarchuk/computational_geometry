import numpy as np

def load_off(filename):
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    assert lines[0] == "OFF"

    nv, nf, *_ = map(int, lines[1].split())
    V = np.array([list(map(float, lines[i+2].split())) for i in range(nv)])
    F = []

    idx = 2 + nv
    for _ in range(nf):
        parts = lines[idx].split()
        assert parts[0] == "3"
        F.append(list(map(int, parts[1:4])))
        idx += 1

    return V, np.array(F)

def face_normals(V, F):
    v0 = V[F[:,0]]
    v1 = V[F[:,1]]
    v2 = V[F[:,2]]

    N = np.cross(v1 - v0, v2 - v0)
    L = np.linalg.norm(N, axis=1, keepdims=True)
    return N / (L + 1e-12)

def ray_intersections(E, dir, V, F):
    v0 = V[F[:,0]]
    v1 = V[F[:,1]]
    v2 = V[F[:,2]]

    e1 = v1 - v0
    e2 = v2 - v0

    p = np.cross(np.tile(dir, (len(F),1)), e2)
    det = np.sum(e1 * p, axis=1)

    eps = 1e-9
    mask = np.abs(det) > eps

    inv_det = np.zeros_like(det)
    inv_det[mask] = 1.0 / det[mask]

    tvec = E - v0
    u = np.sum(tvec * p, axis=1) * inv_det

    q = np.cross(tvec, e1)
    v = np.sum(dir * q, axis=1) * inv_det

    t = np.sum(e2 * q, axis=1) * inv_det

    hit = (mask & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 0))
    return hit, t


def orient_normals_off(V, F, N):
    mins = V.min(0)
    maxs = V.max(0)

    E = maxs + (maxs - mins) * 10.0  # точка вне модели
    centers = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]]) / 3.0
    dirs = centers - E

    for i in range(len(F)):
        dir = dirs[i]
        hit, t = ray_intersections(E, dir, V, F)

        t_face = t[i]
        inside = np.sum(hit & (t < t_face)) % 2 == 1

        toE = E - centers[i]

        # Если inside == True → центр грани внутри
        if inside:
            if np.dot(toE, N[i]) > 0:
                N[i] = -N[i]
        else:
            if np.dot(toE, N[i]) < 0:
                N[i] = -N[i]

    return N


def save_off(filename, V, F, N=None):
    with open(filename, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")

        for x, y, z in V:
            f.write(f"{x} {y} {z}\n")

        for i, (a, b, c) in enumerate(F):
            f.write(f"3 {a} {b} {c}")
            if N is not None:
                nx, ny, nz = N[i]
                f.write(f"   {nx} {ny} {nz}")
            f.write("\n")


V, F = load_off("input.off")
N = face_normals(V, F)
N = orient_normals_off(V, F, N)
save_off("output.off", V, F, N)
