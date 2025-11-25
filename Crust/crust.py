import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def delauney(points: np.ndarray) -> np.ndarray:
    """Compute 2D Delaunay triangulation via 3D lifting and convex hull.

    Returns an array of triangles as integer point indices with shape (m, 3).
    """
    if len(points) < 3:
        return np.empty((0, 3), dtype=int)

    X, Y = points[:, 0], points[:, 1]
    Z = X * X + Y * Y
    pts3 = np.column_stack([X, Y, Z])

    try:
        ch = ConvexHull(pts3)
    except Exception as e:
        print(f"ConvexHull failed: {e}")
        return np.empty((0, 3), dtype=int)

    lower_triangles = []
    for simplex, eq in zip(ch.simplices, ch.equations):
        # Check if normal points down (negative Z component)
        if eq[2] < -1e-10:
            lower_triangles.append(simplex)

    if not lower_triangles:
        return np.empty((0, 3), dtype=int)

    return np.array(lower_triangles, dtype=int)


def circumcenters(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute circumcenters for triangles given by point indices.

    points: (n, 2) array of point coordinates.
    triangles: (m, 3) array of integer indices into `points`.
    Returns: (m, 2) array of circumcenter coordinates.
    """
    if triangles.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    pts = points.astype(np.float64, copy=False)
    tri = triangles.astype(int, copy=False)

    A = pts[tri[:, 0]]
    B = pts[tri[:, 1]]
    C = pts[tri[:, 2]]

    Ax, Ay = A[:, 0], A[:, 1]
    Bx, By = B[:, 0], B[:, 1]
    Cx, Cy = C[:, 0], C[:, 1]

    a2 = Ax * Ax + Ay * Ay
    b2 = Bx * Bx + By * By
    c2 = Cx * Cx + Cy * Cy

    D = 2.0 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))

    eps = 1e-12
    mask = np.abs(D) > eps

    Ux = np.empty_like(Ax)
    Uy = np.empty_like(Ay)

    if np.any(mask):
        D_safe = D[mask]
        num_x = (a2[mask] * (By[mask] - Cy[mask]) +
                 b2[mask] * (Cy[mask] - Ay[mask]) +
                 c2[mask] * (Ay[mask] - By[mask]))
        num_y = (a2[mask] * (Cx[mask] - Bx[mask]) +
                 b2[mask] * (Ax[mask] - Cx[mask]) +
                 c2[mask] * (Bx[mask] - Ax[mask]))

        Ux[mask] = num_x / D_safe
        Uy[mask] = num_y / D_safe

    if not np.all(mask):
        centroid_x = (Ax + Bx + Cx) / 3.0
        centroid_y = (Ay + By + Cy) / 3.0
        Ux[~mask] = centroid_x[~mask]
        Uy[~mask] = centroid_y[~mask]

    return np.column_stack((Ux, Uy))


def extract_edges_from_triangles(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Extract undirected edges as pairs of point indices from triangles.

    triangles: (m, 3) array of integer point indices.
    Returns: (k, 2) array of unique edges (i, j) with i < j.
    """
    if triangles.size == 0:
        return np.empty((0, 2), dtype=int)
    tri = triangles.astype(int, copy=False)
    edges = np.concatenate([
        tri[:, [0, 1]],
        tri[:, [1, 2]],
        tri[:, [2, 0]],
    ], axis=0)

    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges


def plot_crust(points: np.ndarray, edges: np.ndarray):
    """Plot input points and the reconstructed crust in separate subplots.

    points: (n, 2) array of point coordinates.
    edges: (k, 2) array of integer indices into `points`.
    """
    fig, (ax_points, ax_crust) = plt.subplots(1, 2, figsize=(10, 5))

    ax_points.scatter(points[:, 0], points[:, 1], s=10, color="black")
    ax_points.set_aspect("equal", adjustable="box")
    ax_points.set_title("Sample points")

    for i, j in edges:
        x = [points[i, 0], points[j, 0]]
        y = [points[i, 1], points[j, 1]]
        ax_crust.plot(x, y, color="red", linewidth=1.0)
    ax_crust.set_aspect("equal", adjustable="box")
    ax_crust.set_title("Crust reconstruction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_points = np.reshape(np.fromfile("points.txt", sep=" "), (-1, 2))

    delauney_idxes = delauney(input_points)
    circumcenters_array = circumcenters(input_points, delauney_idxes)
    P_union_C = np.vstack([input_points, circumcenters_array])


    tri_coords_2 = delauney(P_union_C)

    crust_edges = extract_edges_from_triangles(input_points, tri_coords_2)

    # Filter edges: keep only edges where both endpoints are in P (indices < len(input_points))
    num_p = len(input_points)
    mask = (crust_edges[:, 0] < num_p) & (crust_edges[:, 1] < num_p)
    crust_edges = crust_edges[mask]

    plot_crust(input_points, crust_edges)
