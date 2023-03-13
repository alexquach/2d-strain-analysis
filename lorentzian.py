from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

raman_parameters = ["offset", "slope", "x0_1", "a1", "gam1", "x0_2", "a2", "gam2"]
PL_parameters = ["offset", "x0_1", "a1", "gam1", "x0_2", "a2", "gam2"]

def lorentzian(x, x0, a, gam):
    return a / (1 + (2 * (x - x0)/gam)**2 )

def mse(popt, x, y, multi_lorentz):
    # off, x0_1, a1, gam1, x0_2, a2, gam2 = popt
    y_pred = [multi_lorentz(x_, *popt) for x_ in x]
    return ((y-y_pred)**2).mean()

def two_lorentz_no_slope(x, off, x0_1, a1, gam1, x0_2, a2, gam2):
        return off + lorentzian(x, x0_1, a1, gam1) + lorentzian(x, x0_2, a2, gam2)

def two_lorentz_with_slope(x, off, slope, x0_1, a1, gam1, x0_2, a2, gam2):
        return off + slope * x + lorentzian(x, x0_1, a1, gam1) + lorentzian(x, x0_2, a2, gam2)

def fit_two_lorentzians_given_yData(yData, xData, startValues, bounds): 
    label_name = yData.name
    yData = yData / max(yData)

    multi_lorentz = two_lorentz_no_slope if len(startValues) == 7 else two_lorentz_with_slope

    try:
        popt, pcov = curve_fit( multi_lorentz, xData, yData.squeeze(), startValues, bounds=bounds, maxfev=100000 )   # Fit multiple Lorentzian functions to the data
    except RuntimeError as e:
        print(e)
        print(label_name)
        return [0, 0, 0, 0, 0, 0, 0, 420, 421] if len(startValues) == 7 else [0, 1, 0, 0, 0, 0, 0, 0, 420, 421]

    # append MSE and R2 values to return result
    y_pred = [multi_lorentz(x, *popt) for x in xData.values]
    res_mse = mean_squared_error(yData.values, y_pred)
    res_r2 = r2_score(yData.values, y_pred)

    res = np.append(popt, [res_mse, res_r2])
    return res


def six_lorentz_no_slope(x, off, x0_1, a1, gam1, x0_2, a2, gam2, x0_3, a3, gam3, x0_4, a4, gam4, x0_5, a5, gam5, x0_6, a6, gam6):
        return off + lorentzian(x, x0_1, a1, gam1) + lorentzian(x, x0_2, a2, gam2) + lorentzian(x, x0_3, a3, gam3) + lorentzian(x, x0_4, a4, gam4) + lorentzian(x, x0_5, a5, gam5) + lorentzian(x, x0_6, a6, gam6)

def six_lorentz_with_slope(x, off, slope, x0_1, a1, gam1, x0_2, a2, gam2, x0_3, a3, gam3, x0_4, a4, gam4, x0_5, a5, gam5, x0_6, a6, gam6):
        return off + slope * x + lorentzian(x, x0_1, a1, gam1) + lorentzian(x, x0_2, a2, gam2) + lorentzian(x, x0_3, a3, gam3) + lorentzian(x, x0_4, a4, gam4) + lorentzian(x, x0_5, a5, gam5) + lorentzian(x, x0_6, a6, gam6)

def fit_six_lorentzians_given_yData(yData, xData, startValues, bounds): 
    label_name = yData.name
    yData = yData / max(yData)

    multi_lorentz = six_lorentz_no_slope if len(startValues) == 1 + 3 * 6 else six_lorentz_with_slope

    try:
        popt, pcov = curve_fit( multi_lorentz, xData, yData.squeeze(), startValues, bounds=bounds, maxfev=100000 )   # Fit multiple Lorentzian functions to the data
    except RuntimeError as e:
        print(e)
        print(label_name)
        return [0] + [0, 0, 0] * 6 + [420, 421] if len(startValues) == 1 + 3 * 6 else [0, 1] + [0, 0, 0] * 6 + [420, 421]

    # append MSE and R2 values to return result
    y_pred = [multi_lorentz(x, *popt) for x in xData.values]
    res_mse = mean_squared_error(yData.values, y_pred)
    res_r2 = r2_score(yData.values, y_pred)

    res = np.append(popt, [res_mse, res_r2])
    return res