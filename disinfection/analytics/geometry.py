def is_point_on_segment(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b

    if (min(ax, bx) - 1 <= px <= max(ax, bx) + 1 and
        min(ay, by) - 1 <= py <= max(ay, by) + 1):
        if abs((bx - ax) * (py - ay) - (by - ay) * (px - ax)) < 1e-2:
            return True
    return False


def is_point_in_region(point, region):
    """
    region: shape (1, N, 2)
    """
    poly = region[0]
    x, y = point
    n = len(poly)
    inside = False

    for i in range(n):
        j = (i + 1) % n
        xi, yi = poly[i]
        xj, yj = poly[j]

        if is_point_on_segment((x, y), (xi, yi), (xj, yj)):
            return True

        if ((yi > y) != (yj > y)):
            x_intersect = ((y - yi) * (xj - xi)) / (yj - yi + 1e-8) + xi
            if x <= x_intersect:
                inside = not inside

    return inside


def is_smaller_y_point_up_of_polygon(points, polygon):
    smaller_y_point = min(points, key=lambda p: p[1])
    polygon_y_coords = [v[1] for v in polygon]
    max_polygon_y = max(polygon_y_coords)
    return smaller_y_point[1] > max_polygon_y


def is_bigger_y_point_in_polygon(points, polygon):
    bigger_y_point = max(points, key=lambda p: p[1])
    polygon_y_coords = [v[1] for v in polygon]
    min_polygon_y = min(polygon_y_coords)
    return bigger_y_point[1] > min_polygon_y


def is_beside_of_polygon(points, region):
    if len(points) != 2:
        return False
    bigger_y_point = max(points, key=lambda p: p[1])
    if is_point_in_region(bigger_y_point, region):
        return False

    poly = region[0]
    polygon_y_coords = [v[1] for v in poly]
    min_polygon_y = min(polygon_y_coords)
    return bigger_y_point[1] > min_polygon_y


def calculate_vertical_distance(foot_positions, last_positions):
    if len(foot_positions) != 2 or len(last_positions) != 2:
        raise ValueError("Both lists of positions must contain exactly two coordinates")
    vertical_distances = []
    for i in range(2):
        vertical_distances.append(foot_positions[i][1] - last_positions[i][1])
    return sum(vertical_distances) / len(vertical_distances)
