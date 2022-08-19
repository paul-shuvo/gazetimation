def calc_vertexes(start_cor: tuple, end_cor: tuple):
    """
    Calculate line segments of the vector arrows to be drawn.
    Parameters
    ----------
    start_cor : numpy.ndarray
        Base point of the arrow.
    end_cor : numpy.ndarray
        End point of the arrow.
    Returns
    -------
    list
        Location of the edge of arrow segments.
    """
    start_x, start_y = start_cor
    end_x, end_y = end_cor
    angle = np.arctan2(end_y - start_y, end_x - start_x) + np.pi
    arrow_length = 15
    arrow_degrees_ = 70

    x1 = int(end_x + arrow_length * np.cos(angle - arrow_degrees_))
    y1 = int(end_y + arrow_length * np.sin(angle - arrow_degrees_))
    x2 = int(end_x + arrow_length * np.cos(angle + arrow_degrees_))
    y2 = int(end_y + arrow_length * np.sin(angle + arrow_degrees_))

    return (x1, y1), (x2, y2)