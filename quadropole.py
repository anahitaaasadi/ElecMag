"""Plots field lines for a quadrupole."""

from matplotlib import pyplot
from numpy import radians

import electrostatics
from electrostatics import PointCharge
from electrostatics import ElectricField, Potential, GaussianCircle
from electrostatics import finalize_plot


def quad(ch,pos):
    # pylint: disable=invalid-name

    XMIN, XMAX = -50, 400
    YMIN, YMAX = -50, 300
    ZOOM = 33
    XOFFSET = 0

    electrostatics.init(XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET)

    # Set up the charges and electric field
    charges = [PointCharge(ch[p], pos[p]) for p in range(len(pos))]
    field = ElectricField(charges)
    potential = Potential(charges)

    # Set up the Gaussian surfaces
    g = [GaussianCircle(charges[i].x, 0.1) for i in range(len(charges))]


    # Create the field lines
    fieldlines = []
    for g_ in g:
        for x in g_.fluxpoints(field, 12):
            fieldlines.append(field.line(x))


    field.plot()
    potential.plot()
    for fieldline in fieldlines:
        fieldline.plot()
    for charge in charges:
        charge.plot()
    finalize_plot()
