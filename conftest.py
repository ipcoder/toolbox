import pytest


@pytest.fixture()
def multi_label_data_table():
    """
    Create Table:
    ::
                              data
        view kind  alg
        L    image cam   [10×10]f8
        R    image cam   [10×10]f8
                   GT    [10×10]f8
                   rand  [10×10]f8

    """
    import numpy as np
    from toolbox.utils.pdtools import DataTable, pd
    from toolbox.utils.label import Keys

    shape = (10, 10)
    imL, imR = (np.ones(shape) * 1, np.ones(shape) * 2)
    best_image = lambda a, b: int(b.mean() > a.mean())
    best_view, best_data = [*zip('LR', [imL, imR])][best_image(imL, imR)]

    db = DataTable(map(Keys('data', 'view', 'kind', 'alg').label, [
        (imL, 'L', 'image', 'cam'),
        (imR, 'R', 'image', 'cam'),
        (best_data, best_view, 'disp', 'GT'),
        (np.random.rand(*shape), pd.NA, 'image', 'rand')
    ]))

    db = db.set_index([*filter('data'.__ne__, db.columns)])
    return db
