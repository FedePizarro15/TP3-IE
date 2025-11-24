from math import radians
import utm

POS = utm.from_latlon(-34.446628, -58.529361)

REFERENCES = {
    "Biblioteca":{"pos": utm.from_latlon(-34.446246, -58.530031),"angulo": radians(304.67)},
    "Comedor":{"pos": utm.from_latlon(-34.446271, -58.529260),"angulo": radians(12.94)},
    "Hirsch":{"pos": utm.from_latlon(-34.446625, -58.528782),"angulo": radians(89.57)}
}