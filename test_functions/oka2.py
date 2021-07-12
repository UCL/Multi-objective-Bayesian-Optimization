import numpy as np
def problem( x ):

    f1 = x[0]
    f2 = 1 - 1/(4*np.pi**2) * (x[0]+np.pi)**2 +\
           (abs(x[1] - 5*np.cos(x[0]) ))**(1/3.)\
         + (abs( x[2]- 5*np.sin(x[0]) ))**(1/3.)
    return [f1, f2]

def bounds():

    return np.array([[-np.pi, -5.,-5.],[np.pi,5.,5.]])