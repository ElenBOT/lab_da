########## Data slot: store data with unit and name to keep tract ##########
########## Data slot: store data with unit and name to keep tract ##########
########## Data slot: store data with unit and name to keep tract ##########
import numpy as np
from copy import copy
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from copy import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import inspect
from scipy.optimize import curve_fit

## Configuration
PREFIX_FACTOR = {
    "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "K": 1e3,
    "": 1.0,
    "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15
}

PREFIXABLE_UNITS = {"A", "Hz", "V", "s", "W", "m", "F", "H", "T", "Gauss", "eV", "J"}

# Registry for unit-to-unit conversions (e.g., linear to log)
UNIT_TRANSFORMS = {
    ("dBm", "mW"): lambda x: 10**(x / 10),
    ("mW", "dBm"): lambda x: 10 * np.log10(x),
    ("dB", "linear"): lambda x: 10**(np.abs(x) / 20),
    ("linear", "dB"): lambda x: 20 * np.log10(np.abs(x)),
    ("T", "Gauss"): lambda x: x * 1e4,
    ("Gauss", "T"): lambda x: x / 1e4,
}

def recognize_unit(unit_str):
    """Split unit string into (prefix, base_unit, factor)."""
    if not isinstance(unit_str, str):
        raise TypeError(f"Unit must be string, not {type(unit_str)}")

    # Sort by length descending to match 'Gauss' before 's'
    for base in sorted(PREFIXABLE_UNITS, key=len, reverse=True):
        if unit_str.endswith(base):
            prefix = unit_str[:-len(base)]
            if prefix in PREFIX_FACTOR:
                return prefix, base, PREFIX_FACTOR[prefix]
            elif not prefix:
                return "", base, 1.0
    
    # Handle non-prefixable units (like dBm or dB)
    return "", unit_str, 1.0

def convert_value_into_newunit(values, old_unit, new_unit="auto"):
    """Convert numeric values between units and SI prefixes.
    
    Example:
        convert_value_into_newunit(0.002, "A", "mA")  -> (2, 'mA')
        convert_value_into_newunit([4e+9, 5e+9], "Hz") -> ([4, 5], 'GHz')
        convert_value_into_newunit(100, "dBm", "mW") -> (10, 'mW')
    """
    ## Input validation and normalization
    if not isinstance(new_unit, str):
        raise TypeError("Target unit must be a string.")
    
    vals = np.asarray(values, dtype=complex)
    old_pre, old_base, old_fact = recognize_unit(old_unit)
    
    # Convert input to the absolute base value
    base_vals = vals * old_fact

    ## Determine target base and prefix
    if new_unit == "auto":
        target_pre, target_base = "auto", old_base
    else:
        target_pre, target_base, _ = recognize_unit(new_unit)

    ## Unit transformation (e.g., mW -> dBm)
    if old_base != target_base:
        transform_key = (old_base, target_base)
        if transform_key not in UNIT_TRANSFORMS:
            raise ValueError(f"Incompatible conversion: {old_base} to {target_base}")
        
        # Check for phase loss warning
        if target_base in ["dB", "dBm"] and not np.all(np.isreal(vals)):
            print("Warning: Converting complex values to dB tosses phase info.")
            
        base_vals = UNIT_TRANSFORMS[transform_key](base_vals)

    ## Prefix scaling
    if target_pre == "auto":
        # Only auto-prefix if the base unit supports it
        if target_base in PREFIXABLE_UNITS:
            max_val = np.max(np.abs(base_vals)) if base_vals.size > 0 else 0
            # Find largest prefix where scaled value is >= 1
            sorted_prefixes = sorted(PREFIX_FACTOR.items(), key=lambda x: x[1], reverse=True)
            target_pre, target_fact = "", 1.0
            for p, f in sorted_prefixes:
                if f != 0 and max_val / f >= 1:
                    target_pre, target_fact = p, f
                    break
        else:
            target_pre, target_fact = "", 1.0
    else:
        target_fact = PREFIX_FACTOR.get(target_pre, 1.0)

    ## Finalize output
    out_vals = base_vals / target_fact
    if np.all(np.isreal(out_vals)):
        out_vals = out_vals.real.astype(float)
        
    return out_vals, f"{target_pre}{target_base}"


class DataSlot:
    def __init__(self, id, dim=1, data=None, name=None, unit=None, dtype='auto'):
        self.id = id
        self.dim = dim
        self.dtype = dtype
        
        if dtype == 'auto':
            self.data = np.empty((0,) * dim, dtype=float) # nD empty array
        else:
            self.data = np.empty((0,) * dim, dtype=dtype) # nD empty array
        if data is None:
            pass
        else: 
            self.set_data(data)
        
        if name is None:
           self.name = id
        else:
            self.set_name(name)
        
        if unit is None:
            self.unit = 'arb.'
        else:
            self.set_unit(unit)

    def set_data(self, input_array):
        ## type checking
        try:
            data_array = np.array(input_array) 
        except Exception as e:
            raise TypeError(f"Input could not be converted to a numpy array: {e}")
        if data_array.ndim != self.dim:
            raise ValueError(f"Input data for slot '{self.id}' must be {self.dim}D, but it has {data_array.ndim} dimensions.")        
        
        if self.dtype == 'auto':
            # check float or complex array
            if np.all(np.isreal(data_array)):
                self.data = np.array(np.real(data_array), dtype=float)
            else:
                self.data = np.array(data_array, dtype=complex)
        else:
            self.data = data_array
        
    def get_data(self):
        return copy(self.data)
    
    def set_unit(self, input_unit):
        try:
            unit_str = str(input_unit)
        except Exception as e:
            raise TypeError(f"Input unit could not be converted to a string: {e}")
        self.unit = unit_str
    def get_unit(self):
        return copy(self.unit)
    
    def set_name(self, input_name):
        try:
            name_str = str(input_name)
        except Exception as e:
            raise TypeError(f"Input name could not be converted to a string: {e}")
        self.name = name_str
    def get_name(self):
            return copy(self.name)
    
    def convert_unit(self, new_unit='auto'):
        old_array = self.data
        old_unit = self.unit
        new_array, new_unit = convert_value_into_newunit(old_array, old_unit, new_unit)
        self.set_data(new_array)
        self.set_unit(new_unit)


########## Data Holder: a holder to hold data slots ##########
########## Data Holder: a holder to hold data slots ##########
########## Data Holder: a holder to hold data slots ##########
class DataHolder:
    def __init__(self, id="dh"):
        self.id = id
        self._slots: dict[str, DataSlot] = {}

    def add_dataslot(self, dataslot: DataSlot):
        if type(dataslot) != DataSlot:
            raise TypeError(f"dataslot must be of type DataSlot, but got {type(dataslot)}")
        self._slots[dataslot.id] = dataslot

    def get_dataslot(self, slot_id):
        # id checking
        if slot_id not in self._slots.keys():
            raise KeyError(f'No such slot id: {slot_id}')
        else:
            slot = self._slots[slot_id]
        return  slot
        
    def set_data_name_unit(self, slot_id, data='keep', name='keep', unit='keep'):
        slot = self.get_dataslot(slot_id)
        
        # data
        if data is None:
            data = np.empty((0,) * slot.dim, dtype=complex) # nD empty array
        if type(data) is str and data == 'keep':
            pass
        else:
            slot.set_data(data)

        # name
        if name is None: 
            name = slot_id
        if type(name) is str and name == 'keep':
            pass
        else:
            slot.set_name(name)
        
        # unit
        if unit is None: 
            unit = 'arb.'
        if type(unit) is str and unit == 'keep':
            pass
        else:
            slot.set_unit(unit)
    
    def get_data_name_unit(self, slot_id):
        slot = self.get_dataslot(slot_id)
        
        # get infos
        data = slot.get_data()
        name = slot.get_name()
        unit = slot.get_unit()
        return data, name, unit

    def convert_unit(self, slot_id, new_unit='auto'):
        slot = self.get_dataslot(slot_id)
        slot.convert_unit(new_unit)

    def vu2ivu(self, slot_id, value, unit='sofar'):
        """Return index, value, unit"""
        slot = self.get_dataslot(slot_id)
        
        ## unit converting
        if unit == 'sofar':
            pass
        else:
            data, unit = convert_value_into_newunit(slot.get_data(), slot.get_unit(), unit)
        
        ## find cloest and return
        def findiv(array, target):
            """Find for the closest index and value
            
            Example usage:
            >>> findiv([4, 3, 2, 1], 3.2)
            OUTPUT:
            | (1, 3)
            """
            diff = [abs(x - target) for x in array]
            index = diff.index(min(diff))
            value = array[index]
            return index, value
        index, value = findiv(data, value)
        return index, value, unit
        
    def i2vu(self, slot_id, index, unit='sofar'):
        """Return value, unit"""
        slot = self.get_dataslot(slot_id)

        ## unit converting and get value
        if unit == 'sofar':
            unit = slot.get_unit()
            data = slot.get_data()
        else:
            data, unit = convert_value_into_newunit(slot.get_data(), slot.get_unit(), unit)
        value = data[index]
        return value, unit


########## Data HolderXY: a holder with some tools to deal with x-y data pair ##########
########## Data HolderXY: a holder with some tools to deal with x-y data pair ##########
########## Data HolderXY: a holder with some tools to deal with x-y data pair ##########
class DataHolderXY(DataHolder):
    def __init__(self, id="dhxy", xdata=None, ydata=None):
        ## Initialize the base class first
        super().__init__(id) 
        
        ## Now self._slots exists, so add_dataslot will work
        self.add_dataslot(DataSlot('x', dim=1, data=xdata, dtype='auto'))
        self.add_dataslot(DataSlot('y', dim=1, data=ydata, dtype='auto'))
    
    @property
    def xdata(self):
        return self.get_dataslot('x').get_data()
    @property
    def ydata(self):
        return self.get_dataslot('y').get_data()
    @property
    def xunit(self):
        return self.get_dataslot('x').get_unit()
    @property
    def yunit(self):
        return self.get_dataslot('y').get_unit()
    @property
    def xname(self):
        return self.get_dataslot('x').get_name()
    @property
    def yname(self):
        return self.get_dataslot('y').get_name()

    def remove_nan(self):
        """remove nan points, return the number of point removed"""
        x = self.xdata
        y = self.ydata
        mask = ~np.isnan(y)
        removed_n = len(y) - np.count_nonzero(mask)
        self.set_data_name_unit('x', x[mask], self.xname, self.xunit)
        self.set_data_name_unit('y', y[mask], self.yname, self.yunit)
        return removed_n

    ## Updated plot method with styling control
    def plot(self, style_str='', *, 
            xlim=None, ylim=None, ax=None, xunit='auto', yunit='auto',
            xfunc=None, yfunc=None, **kwargs):
        """Plot data with support for both plot styles and axis styling.
        
        kwargs:
            label_size, tick_size, title_size
        """
        
        ## create ax if not provided
        plt_show = False
        if ax is None:
            fig, ax = plt.subplots()
            plt_show = True

        ## extract styling kwargs before passing to ax.plot
        # .pop(key, default) removes the key from kwargs and returns the value
        label_size = kwargs.pop('label_size', 12)
        tick_size = kwargs.pop('tick_size', 10)
        title_size = kwargs.pop('title_size', 14)

        ## detecting func bracket
        def detecting_func_bracket(func):
            bracket = '', ''
            if func in [np.abs, abs]: bracket = '|', '|'
            elif func == np.real: bracket = r'Re{', r'}'
            elif func == np.imag: bracket = r'Im{', r'}'
            elif func == np.angle: bracket = r'phase{', r'}'
            elif func is not None: bracket = 'func(', ')'
            return bracket
        
        xbk = detecting_func_bracket(xfunc)
        ybk = detecting_func_bracket(yfunc)

        ## get data and unit
        xdata = xfunc(self.xdata) if xfunc is not None else self.xdata
        ydata = yfunc(self.ydata) if yfunc is not None else self.ydata
        xdata, xunit = convert_value_into_newunit(xdata, self.xunit, xunit)
        ydata, yunit = convert_value_into_newunit(ydata, self.yunit, yunit)

        ## plotting
        # kwargs now only contains valid ax.plot arguments
        ax.plot(xdata, ydata, style_str, **kwargs) if style_str else ax.plot(xdata, ydata, **kwargs)
        
        if xlim is None:
            xlim = [np.min(xdata), np.max(xdata)]
            
        ## applying styles
        ax.set_title(self.id, fontsize=title_size)
        ax.set_xlabel(f'{xbk[0]}{self.xname}{xbk[1]} ({xunit})', fontsize=label_size)
        ax.set_ylabel(f'{ybk[0]}{self.yname}{ybk[1]} ({yunit})', fontsize=label_size)
        ax.tick_params(labelsize=tick_size)
        
        ax.set(xlim=xlim, ylim=ylim)
        ax.grid(True)
        
        if plt_show:
            plt.show()

    def get_sub_dhxy(self, xlim=None, ylim=None, xfunc=None, yfunc=None):
        """ Return a new DataHolderXY with data points within given xlim and ylim.
        
        """
        ## detecting func bracket
        def detecting_func_bracket(func):
            bracket = '', ''
            if func is not None:
                bracket = 'func(', ')'
            if func == np.abs or func == abs:
                bracket = '|', '|'
            if func == np.real:
                bracket = r'Re{', r'}'
            if func == np.imag:
                bracket = r'Im{', r'}'
            if func == np.angle:
                bracket = r'phase{', r'}'
            return bracket
        xbk = detecting_func_bracket(xfunc)
        ybk = detecting_func_bracket(yfunc)
        x = xfunc(self.xdata) if xfunc is not None else self.xdata
        y = yfunc(self.ydata) if yfunc is not None else self.ydata

        # default
        if xlim is None: xlim = [np.min(x), np.max(x)]
        if ylim is None: ylim = [np.min(y), np.max(y)]

        # create mask
        mask = (
            (x >= xlim[0]) & (x <= xlim[1]) &
            (y >= ylim[0]) & (y <= ylim[1])
        )
        # filter data 
        x_sub = x[mask]
        y_sub = y[mask]

        # create new DataHolderXY with same metadata
        sub_dh = DataHolderXY(id=f"{self.id} sub", xdata=x_sub, ydata=y_sub)
        sub_dh.set_data_name_unit('x', x_sub, f'{xbk[0]}{self.xname}{xbk[1]}', self.xunit)
        sub_dh.set_data_name_unit('y', y_sub, f'{ybk[0]}{self.yname}{ybk[1]}', self.yunit)
        
        return sub_dh

    def swap_xy(self):
        xdnu = self.get_data_name_unit('x')
        ydnu = self.get_data_name_unit('y')
        self.set_data_name_unit('x', *ydnu)
        self.set_data_name_unit('y', *xdnu)

    def general_fit(
        self, 
        func,
        *,
        strpts = None,
        bds = (-np.inf, np.inf),
        guess_mode = False,
        print_result = True,
        ax = None,
        ):
        """Fit x-y data and plot it. Returns coeffs, Rsq.s

        Example usage:
        >>> def quad(x, a, b, c):
        >>>     return a*x**2 + b*x + c
        >>> x_data = np.array([1, 2, 3, 4, 5, 6])
        >>> y_data = np.array([1.1, 2.4, 3.2, 4, 5.4, 7])
        >>> dh = DataHolderXY('fitting example', x_data, y_data)
        >>> strpts = [1, 1, 0.2] # a, b, c
        >>> bds = ([0, 0, 0], [10, 10, 10]) # lowers, uppers
        >>> coeffs, Rsq = dh.fit_and_plot(
        >>>     quad,
        >>>     strpts=strpts,
        >>>     bds=bds,
        >>>     guess_mode=True, # guess until the strpts is a close enough
        >>>     print_result=True
        >>> )

        For guess mode, it plot the guessed function along with data.
        """
        def r2_score(y_true, y_pred):
            """Evaluate the R-squred value."""
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        
        if guess_mode:
            plt.plot(self.xdata, self.ydata, 'k s', label='data')
            samt = np.linspace(np.min(self.xdata), np.max(self.xdata), 4000)
            plt.plot(samt, func(samt, *strpts), label='guess')
            plt.legend()
            plt.grid()
            plt.show()
            return None, None
        
        ## fit using abs of y
        if not np.all(np.isreal([self.ydata])):
            ydata = np.abs(self.ydata)
            print('notification: fit using abs of y. For more control over complex data please use `get_sub_hdxy`.')
        else:
            ydata = self.ydata
        
        # fitting
        if not guess_mode:
            coeffs, cov = curve_fit(
                func, self.xdata, ydata,
                p0=strpts, bounds=bds
            )
            
            # compute R^2
            y_pred = func(self.xdata, *coeffs)
            Rsquared = r2_score(ydata, y_pred)
            
            # print fitted result
            if print_result:
                param_names = list(inspect.signature(func).parameters.keys())[1:]
                print('fitted result:')
                for name, value in zip(param_names, coeffs):
                    print(f">> {name} = {value:.5e}")    
                print(f'with R-squared value R^2 = {Rsquared:.5f}')

                # plot
                plt_show = False
                if ax is None:
                    fig, ax = plt.subplots()
                    plt_show = True
                
                ax.plot(self.xdata, ydata, 'k s', label='data')
                samx = np.linspace(np.min(self.xdata), np.max(self.xdata), 4000)
                ax.plot(samx, func(samx, *coeffs), 'r-',label='fitted')
                plt.xlim(np.min(self.xdata), np.max(self.xdata))
                ax.set(
                    title=f'{self.id} fitting result',
                    xlabel=f'{self.xname} ({self.xunit})',
                    ylabel=f'{self.yname} ({self.yunit})',
                    xlim=[np.min(self.xdata), np.max(self.xdata)]
                )
                ax.legend()
                ax.grid()
                
                if plt_show:  # if we created a figure inside
                    plt.show()

            return coeffs, Rsquared

    def linear_fit(dhxy, print_result=True, ax=None):
        """Perform a linear fit: y = a*x + b. Returns coeffs, Rsq.

        (Write by AI)
        """
        ## func
        def linear(x, a, b):
            return a * x + b

        ## fit using abs of y
        if not np.all(np.isreal([dhxy.ydata])):
            ydata = np.abs(dhxy.ydata)
            print('notification: fit using abs of y. For more control over complex data please use `get_sub_hdxy`.')
        else:
            ydata = dhxy.ydata
        
        ## Estimate initial guess
        x, y = dhxy.xdata, ydata
        A = np.vstack([x, np.ones_like(x)]).T
        # least squares estimate
        a_est, b_est = np.linalg.lstsq(A, y, rcond=None)[0]  
        
        ## call method for plotting
        strpts = [a_est, b_est]
        bds = (-np.inf, np.inf)  # allow any values
        coeffs, Rsq = dhxy.general_fit(
            linear,
            strpts=strpts,
            bds=bds,
            guess_mode=False,
            print_result=print_result,
            ax=ax
        )

        return coeffs, Rsq
    
    def exp_decay_fit(dhxy, print_result=True, ax=None):
        """Perform an exp decay fit: y = A*exp(-x / tau) + C. Returns coeffs, Rsq.

        (Write by AI)
        """
        ## func
        def exp_decay(t, A, tau, y0):
            return A * np.exp(-t / tau) + y0

        ## fit using abs of y
        if not np.all(np.isreal([dhxy.ydata])):
            ydata = np.abs(dhxy.ydata)
            print('notification: fit using abs of y. For more control over complex data please use `get_sub_hdxy`.')
        else:
            ydata = dhxy.ydata
        
        ## Estimate offset (y0) as last point, amplitude (A) as first - last
        x, y = dhxy.xdata, ydata
        y0_est = y[-1]
        A_est = y[0] - y0_est

        ## Estimate tau: find x where y has dropped to ~1/e of its initial amplitude
        try:
            target = y0_est + A_est / np.e
            idx = np.argmin(np.abs(y - target))
            tau_est = np.abs(x[idx] - x[0]) if idx > 0 else (x[-1] - x[0]) / 2
        except Exception:
            tau_est = (x[-1] - x[0]) / 2  # fallback

        ## call method for plotting
        strpts = [A_est, tau_est, y0_est]
        bds = ((-np.inf, 0, -np.inf), (np.inf, np.inf, np.inf))  # tau > 0
        coeffs, Rsq = dhxy.general_fit(
            exp_decay,
            strpts=strpts,
            bds=bds,
            guess_mode=False,
            print_result=print_result,
            ax=ax
        )

        return coeffs, Rsq


########## Data HolderXYZ: a holder with some tools to deal with 2D z-data with x-y axis ##########
########## Data HolderXYZ: a holder with some tools to deal with 2D z-data with x-y axis ##########
########## Data HolderXYZ: a holder with some tools to deal with 2D z-data with x-y axis ##########
class DataHolderXYZ(DataHolder):
    def __init__(self, id="dhxyz", xdata=None, ydata=None, zdata=None):
        ## Initialize the base class first
        super().__init__(id) 
        
        self.id = id
        self.add_dataslot(DataSlot('x', dim=1, data=xdata, dtype='auto'))
        self.add_dataslot(DataSlot('y', dim=1, data=ydata, dtype='auto'))
        self.add_dataslot(DataSlot('z', dim=2, data=zdata, dtype='auto'))

    @property
    def xdata(self):
        return self.get_dataslot('x').get_data()
    @property
    def ydata(self):
        return self.get_dataslot('y').get_data()
    @property
    def zdata(self):
        return self.get_dataslot('z').get_data()
    @property
    def xunit(self):
        return self.get_dataslot('x').get_unit()
    @property
    def yunit(self):
        return self.get_dataslot('y').get_unit()
    @property
    def zunit(self):
        return self.get_dataslot('z').get_unit()
    @property
    def xname(self):
        return self.get_dataslot('x').get_name()
    @property
    def yname(self):
        return self.get_dataslot('y').get_name()
    @property
    def zname(self):
        return self.get_dataslot('z').get_name()

    def swap_xy(self):
        xdnu = self.get_data_name_unit('x')
        ydnu = self.get_data_name_unit('y')
        self.set_data_name_unit('x', *ydnu)
        self.set_data_name_unit('y', *xdnu)
        self.set_data_name_unit('z', self.zdata.T)

    def get_sub_dhxyz(self, xindlim=[0, -1], yindlim=[0, -1],
                      xfunc=None, yfunc=None, zfunc=None):
        ## Validate input
        if not (isinstance(xindlim, (list, tuple)) and len(xindlim) == 2):
            raise ValueError("xindlim must be a 2-element tuple or list (start, stop).")
        if not (isinstance(yindlim, (list, tuple)) and len(yindlim) == 2):
            raise ValueError("yindlim must be a 2-element tuple or list (start, stop).")

        ## detecting func bracket
        def detecting_func_bracket(func):
            bracket = '', ''
            if func is not None: bracket = 'func(', ')'
            if func in [np.abs, abs]: bracket = '|', '|'
            if func == np.real: bracket = r'Re{', r'}'
            if func == np.imag: bracket = r'Im{', r'}'
            if func == np.angle: bracket = r'phase{', r'}'
            return bracket
            
        xbk = detecting_func_bracket(xfunc)
        ybk = detecting_func_bracket(yfunc)
        zbk = detecting_func_bracket(zfunc)

        ## helper for inclusive slicing
        # logic: if stop is -1, we want the end of the array (None). 
        # Otherwise, we add 1 to include the stop index.
        def get_slice(lim):
            start, stop = lim
            stop_idx = None if stop == -1 else stop + 1
            return slice(start, stop_idx)

        sl_x = get_slice(xindlim)
        sl_y = get_slice(yindlim)

        ## Extract data and apply slicing
        x = xfunc(self.xdata) if xfunc is not None else self.xdata
        y = yfunc(self.ydata) if yfunc is not None else self.ydata
        z = zfunc(self.zdata) if zfunc is not None else self.zdata
        
        x_sub = x[sl_x]
        y_sub = y[sl_y]
        z_sub = z[sl_y, sl_x]
        
        ## create new dhxyz
        sub_dh = DataHolderXYZ(id=f"{self.zname} sub [{xindlim}, {yindlim}]")
        sub_dh.set_data_name_unit('x', x_sub, f'{xbk[0]}{self.xname}{xbk[1]}', self.xunit)
        sub_dh.set_data_name_unit('y', y_sub, f'{ybk[0]}{self.yname}{ybk[1]}', self.yunit)
        sub_dh.set_data_name_unit('z', z_sub, f'{zbk[0]}{self.zname}{zbk[1]}', self.zunit)
        
        return sub_dh
        
    def get_sliced_dhxy(self, slot_id, slice_index):
        ## get sliced axis
        if slot_id == 'x':
            new_x_id = 'y'
            new_ydata = self.get_dataslot('z').get_data()[:, slice_index]
        elif slot_id == 'y':
            new_x_id = 'x'
            new_ydata = self.get_dataslot('z').get_data()[slice_index, :]
        else: 
            raise Exception()
        
        ## obtain the new XY data
        new_xdata, new_xname, new_xunit = self.get_data_name_unit(new_x_id)
        _, new_yname, new_yunit = self.get_data_name_unit('z')
        
        ## make new dh and display info
        value, unit = self.i2vu(slot_id, index=slice_index)
        dh = DataHolderXY(f'{self.id} slice @ {value}{unit}')
        dh.set_data_name_unit('x', new_xdata, new_xname, new_xunit)
        dh.set_data_name_unit('y', new_ydata, new_yname, new_yunit)
        return dh

    def sliced_val_and_dhxy(self, slot_id):
        """Iterate through (value, DataHolderXY) pairs for each slice along the given slot_id ('x' or 'y').
        """
        z = self.get_dataslot('z').get_data()

        if slot_id == 'x':
            values = self.xdata
            num_slices = z.shape[1]
        elif slot_id == 'y':
            values = self.ydata
            num_slices = z.shape[0]
        else:
            raise ValueError("slot_id must be 'x' or 'y'.")

        for idx in range(num_slices):
            yield values[idx], self.get_sliced_dhxy(slot_id, idx)

    ## flip methods
    def flip_x(self):
        ## flip x array and z array columns
        new_x = self.xdata[::-1]
        new_z = np.flip(self.zdata, axis=1)
        self.set_data_name_unit('x', data=new_x)
        self.set_data_name_unit('z', data=new_z)

    def flip_y(self):
        ## flip y array and z array rows
        new_y = self.ydata[::-1]
        new_z = np.flip(self.zdata, axis=0)
        self.set_data_name_unit('y', data=new_y)
        self.set_data_name_unit('z', data=new_z)

    def plot(
            self, ax=None, 
            xunit='auto', yunit='auto', zunit='auto',
            origin='auto', xlim=None, ylim=None,
            xfunc=None, yfunc=None, zfunc=None, **imshow_kwarg
        ):
        from matplotlib.ticker import MaxNLocator

        ## detecting func bracket
        def detecting_func_bracket(func):
            bracket = '', ''
            if func is not None: bracket = 'func(', ')'
            if func in [np.abs, abs]: bracket = '|', '|'
            if func == np.real: bracket = r'Re{', r'}'
            if func == np.imag: bracket = r'Im{', r'}'
            if func == np.angle: bracket = r'phase{', r'}'
            return bracket
            
        xbk = detecting_func_bracket(xfunc)
        ybk = detecting_func_bracket(yfunc)
        zbk = detecting_func_bracket(zfunc)
            
        ## get data and unit to be plot
        xdata = xfunc(self.xdata) if xfunc is not None else self.xdata
        ydata = yfunc(self.ydata) if yfunc is not None else self.ydata
        zdata = zfunc(self.zdata) if zfunc is not None else self.zdata
        xdata, xunit = convert_value_into_newunit(xdata, self.xunit, xunit)
        ydata, yunit = convert_value_into_newunit(ydata, self.yunit, yunit)
        zdata, zunit = convert_value_into_newunit(zdata, self.zunit, zunit)

        ## shape matching check
        if zdata.shape != (len(ydata), len(xdata)):
            raise ValueError(
                f"Shape mismatch: z array shape {zdata.shape} does not match" 
                f"y array length ({len(ydata)}) and x array length ({len(xdata)})."
        )

        ## configure extent based on values
        if type(origin) == str and origin == 'auto':
            if len(ydata) > 1 and ydata[0] < ydata[-1]: origin = 'lower'
            else: origin = 'upper'

        dx = (xdata[-1] - xdata[0]) / (len(xdata) - 1) if len(xdata) > 1 else 1.0
        dy = (ydata[-1] - ydata[0]) / (len(ydata) - 1) if len(ydata) > 1 else 1.0

        left = xdata[0] - dx / 2
        right = xdata[-1] + dx / 2

        if origin == 'upper':
            top = ydata[0] - dy / 2
            bottom = ydata[-1] + dy / 2
        else:
            bottom = ydata[0] - dy / 2
            top = ydata[-1] + dy / 2
            
        extent = [left, right, bottom, top]
        
        ## create ax if not provided
        plt_show = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
            plt_show = True
        else:
            fig = ax.get_figure()

        ## ploting
        im = ax.imshow(
            zdata, aspect='auto', origin=origin, extent=extent,
            **imshow_kwarg
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.8)
        fig.colorbar(im, cax=cax, label=f'{zbk[0]}{self.zname}{zbk[1]} ({zunit})')
        
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)

        ## Primary axes (values)
        ax.set_xlabel(f'{xbk[0]}{self.xname}{xbk[1]} ({xunit})')
        ax.set_ylabel(f'{ybk[0]}{self.yname}{ybk[1]} ({yunit})')

        ## Secondary axes (indices)
        # Replaced np.interp with strict linear mathematical mappings to prevent 
        # edge-clipping, ensuring index 0 aligns exactly with xdata[0].
        slope_x = (len(xdata) - 1) / (xdata[-1] - xdata[0]) if len(xdata) > 1 else 1.0
        def val_to_xindex(v): return (v - xdata[0]) * slope_x
        def xindex_to_val(i): return xdata[0] + i / slope_x

        secax_x = ax.secondary_xaxis('top', functions=(val_to_xindex, xindex_to_val))
        secax_x.set_xlabel('x index')
        secax_x.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        slope_y = (len(ydata) - 1) / (ydata[-1] - ydata[0]) if len(ydata) > 1 else 1.0
        def val_to_yindex(v): return (v - ydata[0]) * slope_y
        def yindex_to_val(i): return ydata[0] + i / slope_y

        secax_y = ax.secondary_yaxis('right', functions=(val_to_yindex, yindex_to_val))
        secax_y.set_ylabel('y index')
        secax_y.yaxis.set_major_locator(MaxNLocator(integer=True))

        ## show ax if not provided
        if plt_show:
            plt.tight_layout()
            plt.show()