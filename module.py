import scipy as sp
from scipy.integrate import quad
import sys

eps = 1e-17
Ro = 1  # assign Ro


def classical_first_model(t, k, rdt):
    print(k * t)
    return Ro * (1 - sp.exp(-k * t)) * rdt


def kienetic_model(t, k, rtd):
    return (Ro ** 2 * k * t) / (1 + Ro * k * t) * rtd


def first_flotabililtis(t, k, rtd):
    return Ro * (1 - (1 - sp.exp(-k * t))) * rtd


def second_flotabililtis(t, k, rtd):
    return Ro * (1 - (sp.log(1 + k * t)) / (k * t)) * rtd


def second_flotabililtisfully_mixed_rector(t, k, rtd):
    return Ro * (1 - 1 / (1 + (t / k))) * rtd


def improved_gas_solid(t, k, rtd):
    return Ro * ((k * t) / (1 + k * t)) * rtd


rtd_defualt = 1  # assign RTD

# insert function name from above in first argument without ()
# the range of integrate o , 2500 # if you want infinity --> sp.inf
result = quad(second_flotabililtis, 0, 2500, args=(1, rtd_defualt))  # first integrate

R = 0.5  # R label (actual R)
in_k = 1  # assign first K

while not R - result[0] > eps:
    a = result[0]
    result = quad(second_flotabililtis, 0, 2500, args=(in_k, rtd_defualt))

    in_k -= 1e-2  # size of steps # just change number from 2 to anything you want

    if result[0] < R:
        break

error = (result[0] - R) ** 2

print("first_flotabililtis")
print("zrib K = {:.2f}".format(in_k))
print("error : {:.2f}".format(error))
