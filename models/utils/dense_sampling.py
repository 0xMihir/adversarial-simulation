import numpy as np

def densify(points, step_distance=1.0):
    """Interpolates points along a polyline so they are spaced no more than step_distance apart."""
    if len(points) < 2:
        return np.array(points)

    new_points = [points[0]]
    for i in range(1, len(points)):
        p1 = np.array(points[i - 1])
        p2 = np.array(points[i])
        dist = np.linalg.norm(p2 - p1)

        if dist > step_distance:
            num_points = int(np.ceil(dist / step_distance))
            interpolated = np.linspace(p1, p2, num_points + 1)[1:]
            new_points.extend(interpolated)
        else:
            new_points.append(p2)

    return np.array(new_points)