import numpy as np
from utils.fields import Field


def descartes_folium_function(t, x):
    """
    Descartes folium function.

    """
    t = float(t); x = [float(z) for z in x]
    a, b, c, d = 2.5, -2, 2, -2
    alpha = 0.2
    return alpha * (a*x[0] + b*x[1] + 5), alpha * (c*x[0] + d*x[1] + 5)


# generate fields from functions
field_1 = Field.generate_zero(shape=(20, 20, 1, 1, 2))

for i in range(0, 20):
        for j in range(0, 20):
            field_1.field[i, j, 0, 0, :] = descartes_folium_function(1, [i, j])