import math
import numpy as np
import pandas as pd

import hkPhysicalConstants as hpc
import hkAstronomicalConstants as hac
import hkAstrophysics as hap

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

def CreateHRDiagram(X, Y, type='O'):
	"""
	Creates a Hertzsprung-Russell Diagram.

	IN:
	X: temperatures or color indices
	L: luminosities or absolute magnitudes
	type:
		O = Observational
		T = Theoretical

	OUT:
	-
	"""

	plt.figure()

	X_min = min(X)
	X_max = max(X)
	Y_min = min(Y)
	Y_max = max(Y)
	plt.xlim(X_min, X_max)
	plt.ylim(Y_min, Y_max)
#	print 'min(X) = {0:.3E}; max(X) = {1:.3E}'.format(X_min, X_max)
#	print 'min(Y) = {0:.3E}; max(Y) = {1:.3E}'.format(Y_min, Y_max)

	if(type=='O'):
		# Create observational HR diagram
		plt.plot(X, Y, 'b.')

		axes = plt.gca()
		axes.invert_yaxis()

		plt.xlabel('B - V')
		plt.ylabel('Mv')
		plt.title('Observational Hertzsprung-Russell Diagram')
	else:
		# Create theoratical HR diagram

		# Plot radius info
		R_min = hap.CalculateLuminosity(T=X_max, L=Y_min)[1]
		R_max = hap.CalculateLuminosity(T=X_min, L=Y_max)[1]
#		print 'R_min = {0:.3E}; R_max = {1:.3E}'.format(R_min, R_max)

		# Determine the number the 10-base factor of the radius with respect to the radius of the Sun
		RR = R_min/hac.R_Sun
#		print 'RR_min = {0:.3E}'.format(RR)
		if(RR < 1):
			power_min = -1 * (len(str(int(round(1/RR)))))
		else:
			power_min = len(str(int(round(RR))))
		power_min = power_min - 1

		RR = R_max/hac.R_Sun
#		print 'RR_max = {0:.3E}'.format(RR)
		if(RR < 1):
			power_max = -1 * (len(str(int(round(1/RR)))))
		else:
			power_max = len(str(int(round(RR))))
		power_max = power_max + 1

#		print 'power_min = {0}, power_max = {1}'.format(power_min, power_max)
#		print 'range: {0}'.format(range(power_min, power_max+1))

		# Plot iso-radius lines
		for p in range(power_min, power_max+1):
			R = hac.R_Sun * 10**p

			T1 = X_min
			L1 = hap.CalculateLuminosity(R=R, T=T1)[0]
			if(L1 > Y_max):
				L1 = Y_max
				T1 = hap.CalculateLuminosity(R=R, L=L1)[2]
			elif(L1 < Y_min):
				L1 = Y_min
				T1 = hap.CalculateLuminosity(R=R, L=L1)[2]

			T2 = X_max
			L2 = hap.CalculateLuminosity(R=R, T=T2)[0]
			if(L2 > Y_max):
				L2 = Y_max
				T2 = hap.CalculateLuminosity(R=R, L=L2)[2]
			elif(L2 < Y_min):
				L2 = Y_min
				T2 = hap.CalculateLuminosity(R=R, L=L2)[2]

#			print 'R = {0:.3E}; T1 = {1:.3E}; L1 = {2:.3E}; T2 = {3:.3E}; L2={4:.3E}'.format(R, T1, L1, T2, L2)

			if(T1 >= X_min and T2 <= X_max and L1 >= Y_min and L2 <= Y_max):
				# Plot the iso-radius line
				plt.loglog([T1, T2], [L1, L2], color='0.85')

				# Add a label to it
				rot = -52 #-1 * math.atan(math.log10(L2 - L1)/math.log10(T2 - T1))*180.0/math.pi # Determine the slope of the iso-radius line
				scale = math.log(X_max - X_min)/math.log(Y_max - Y_min) # Determine the scaling of the whole plot
				angle = round(scale * rot)

#				print 'rot = {0}; scale = {1}; angle = {2}'.format(rot, scale, angle)

				text = r'${:G}'.format(10**p) + r' R_{\odot}$'
				plt.annotate(text, xy=(T1, L1), xycoords='data', xytext=(-50, +25), textcoords='offset points', rotation=angle, fontsize=10) #, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

		# TODO: Also add colorbar at bottom

		# Get an appropriate colormap
		cmap = cmx.rainbow

		# Boundaries of the visible part of the electromagnetic spectrum
		tmin = 2.9E-3 / 700E-9
		tmax = 2.9E-3 / 400E-9
		cnorm  = colors.Normalize(vmin=tmin, vmax=tmax)
		scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

		# Get the color
		C = scalarmap.to_rgba(X)

		# Plot the data
#		plt.loglog(X, Y, 'b.')
		plt.scatter(X, Y, color=C)
		plt.colorbar

		axes = plt.gca()
		axes.invert_xaxis()

		plt.xlabel('Temperature / K')
		plt.ylabel('Luminosity / W')
		plt.title('Theoretical Hertzsprung-Russell Diagram')

#	plt.grid(True)
	plt.show(block=False)

def PlotSpectralRadiance(temperatureArray, wavelengthArray):
	"""
	Plots the Planck function for the given temperatures.

	IN:
	temperatureArray: array of temperatures, in K
	wavelengthArray: array of wavelengths, in m

	OUT:
	"""

	x_min = min(wavelengthArray)
	x_max = max(wavelengthArray)
	x_delta = x_max - x_min
	x_min -= 0.1*x_delta
	x_max += 0.1*x_delta

	y_min = 0.
	y_max = 0.

	# Plot values
	fig = plt.figure()
	ax0 = fig.add_subplot(1, 1, 1)

	for T in temperatureArray:
		B = []

		for l in wavelengthArray:
			b = hap.SpectralRadiance(T=T, l=l)[0]
			B.append(b)

		if min(B)<y_min:
			y_min = min(B)

		if max(B)>y_max:
			y_max = max(B)

		y_delta = y_max - y_min
		y_min -= 0.1*y_delta
		y_max += 0.1*y_delta

		ax0.loglog(wavelengthArray, B, 'b-')
		# ax0.loglog(wavelengthArray, B, color='blue', linewidth = 1, linestyle = 'line')

		# Create annotationa for the temperature
		x = hap.WiensDisplacementLaw(T=T)[0]
		y = hap.SpectralRadiance(T=T, l=x)[0]
		text = 'T = {0:.3G} K'.format(T)
		plt.annotate(text, xy=(x, y), arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.3'), xytext=(+25, +15), xycoords = 'data', textcoords='offset points')

	ax0.set_xlim(x_min, x_max)
	ax0.set_ylim(1E-15, 1E45)
	# ax0.set_ylim(y_min, y_max)

	ax0.set_ylabel(r'Spectral radiance $B_{\lambda}(T)/W m^{2} m^{-1} sr^{-1}$')
	ax0.set_xlabel(r'Wavelength $\lambda / m$')

	ax0.invert_xaxis()

	plt.show()

def LightCurve1():
    """
    Compares three models for the limb-darkening of a host star to estimate the flux deficit caused by an orbiting exoplanet.
    """

    # HD 209458 b
    RS = 1.146 * hac.R_Sun # Radius host star
    RP = 1.359 * hac.R_Jup # Radius orbiting planet
    P = 3.52*24*60*60 # Period of the orbiting planet
    a = 0.04747*hac.AU # Astrocentric semi-major axis
    i = math.radians(86.71) # Inclination of the orbit with respect to observer
    s = 'HD 209458 b'

    # First use a linear model
    u = 0.215 # First parameter of limb-darkening law
    v = 0.0 # Second parameter of limb darkening law
    m = 'lin' # Limb-darkening model to be used

    [timeArray0, areaArray0, fluxArray0, relativeFluxArray0] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m)

    # Second use a logarithmic model
    u = 0.14 # First parameter of limb-darkening law
    v = -0.12 # Second parameter of limb darkening law
    m = 'log' # Limb-darkening model to be used

    [timeArray1, areaArray1, fluxArray1, relativeFluxArray1] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m)

    # Third use a quadratic model
    u = 0.29 # First parameter of limb-darkening law
    v = -0.13 # Second parameter of limb darkening law
    m = 'quad' # Limb-darkening model to be used

    [timeArray2, areaArray2, fluxArray2, relativeFluxArray2] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m)

    # Plot values
    x_min = min(timeArray0)/(60*60*24)
    x_max = max(timeArray0)/(60*60*24)

    fig = plt.figure()

    # Plot a chart with the flux
    y_min = min(fluxArray0)
    y_max = max(fluxArray0)
    y_min -= 0.1*(y_max-y_min)
    y_max += 0.1*(y_max-y_min)

    ax0 = fig.add_subplot(3, 1, 1)

    ax0.plot(timeArray0/(60*60*24), fluxArray0, color='blue', linewidth = 1)
    ax0.plot(timeArray0/(60*60*24), fluxArray1, color='green', linewidth = 1)
    ax0.plot(timeArray0/(60*60*24), fluxArray2, color='red', linewidth = 1)

    ax0.set_xlim(x_min, x_max)
    ax0.set_ylim(y_min, y_max)

    ax0.set_title(s)

    ax0.set_xlabel('time/days')
    ax0.set_ylabel(r'$\Delta F/W m^{-2}$')

    # Plot a chart with the relative flux
    min_y = []
    min_y.append(min(relativeFluxArray0))
    min_y.append(min(relativeFluxArray1))
    min_y.append(min(relativeFluxArray2))
    y_min = min(min_y)

    max_y = []
    max_y.append(max(relativeFluxArray0))
    max_y.append(max(relativeFluxArray1))
    max_y.append(max(relativeFluxArray2))
    y_max = max(max_y)

    y_min -= 0.1*(y_max-y_min)
    y_max += 0.1*(y_max-y_min)

    ax1 = fig.add_subplot(3, 1, 2, sharex=ax0)

    ax1.plot(timeArray0/(60*60*24), relativeFluxArray0, color='blue', linewidth = 1)
    ax1.plot(timeArray0/(60*60*24), relativeFluxArray1, color='green', linewidth = 1)
    ax1.plot(timeArray0/(60*60*24), relativeFluxArray2, color='red', linewidth = 1)

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    ax1.set_xlabel('time/days')
    ax1.set_ylabel('Relative flux')

    # Plot a chart with the occulted area
    y_min = min(areaArray0)
    y_max = max(areaArray0)
    y_min -= 0.1*(y_max-y_min)
    y_max += 0.1*(y_max-y_min)

    ax2 = fig.add_subplot(3, 1, 3, sharex=ax0)

    ax2.plot(timeArray0/(60*60*24), areaArray0, color='blue', linewidth = 1)
    # ax2.plot(timeArray0/(60*60*24), areaArray0/(math.pi*hac.R_Jup**2), color='blue', linewidth = 1)

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('time/days')
    ax2.set_ylabel(r'Occulted area$/m^{2}$')
    # ax2.set_ylabel(r'Occulted area$/A_{Jup}$')

    plt.xlabel('time/days')
    plt.show()

def LightCurve2():
    """
    Tunes the model for a transiting exoplanet until a best fit with measurement data is reached.
    """

    # Load measurement data
    dataframe = pd.read_csv('./Data/hd209458b.csv', comment='#', header=0, names=['time', 'flux', 'error'], delim_whitespace=True)
    dataframe['time'] = dataframe['time']*24*60*60 # Convert from fractions of a day to seconds

    # HD 209458 b
    RS = 1.1 * hac.R_Sun # Radius host star
    RP = 1.27 * hac.R_Jup # Radius orbiting planet
    P = 3.52*24*60*60 # Period of the orbiting planet
    a = 0.0467*hac.AU # Astrocentric semi-major axis
    i = math.radians(87.1) # Inclination of the orbit with respect to observer
    s = 'HD 209458 b'

    # Use a logarithmic model
    u = 0.5 # First parameter of limb-darkening law
    v = -0.12 # Second parameter of limb darkening law
    m = 'lin' # Limb-darkening model to be used

    # Use first estimates and calculate fit quality with measurement data
    [timeArray, areaArray, fluxArray, relativeFluxArray] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m, timeArray = dataframe['time'])
    cur_chiSquared = hap.ChiSquared(relativeFluxArray, dataframe['flux'], dataframe['error'])

    # Start modifying the radius of the exoplanet, and check if quality of fit with measurement data increases
    prev_chiSquared = cur_chiSquared + 1
    while cur_chiSquared<prev_chiSquared:
        prev_chiSquared = cur_chiSquared

        RP *= 1.01
        [timeArray, areaArray, fluxArray, relativeFluxArray] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m, timeArray = dataframe['time'])
        cur_chiSquared = hap.ChiSquared(relativeFluxArray, dataframe['flux'], dataframe['error'])
    cur_chiSquared = prev_chiSquared

    prev_chiSquared = cur_chiSquared + 1
    while cur_chiSquared<prev_chiSquared:
        prev_chiSquared = cur_chiSquared

        RP /= 1.01
        [timeArray, areaArray, fluxArray, relativeFluxArray] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m, timeArray = dataframe['time'])
        cur_chiSquared = hap.ChiSquared(relativeFluxArray, dataframe['flux'], dataframe['error'])
    cur_chiSquared = prev_chiSquared

    # Start modifying the inclination of the exoplanet's orbit, and check if quality of fit with measurement data increases
    prev_chiSquared = cur_chiSquared + 1
    while cur_chiSquared<prev_chiSquared:
        prev_chiSquared = cur_chiSquared

        i *= 1.01
        [timeArray, areaArray, fluxArray, relativeFluxArray] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m, timeArray = dataframe['time'])
        cur_chiSquared = hap.ChiSquared(relativeFluxArray, dataframe['flux'], dataframe['error'])
    cur_chiSquared = prev_chiSquared

    prev_chiSquared = cur_chiSquared + 1
    while cur_chiSquared<prev_chiSquared:
        prev_chiSquared = cur_chiSquared

        i /= 1.01
        [timeArray, areaArray, fluxArray, relativeFluxArray] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m, timeArray = dataframe['time'])
        cur_chiSquared = hap.ChiSquared(relativeFluxArray, dataframe['flux'], dataframe['error'])
    cur_chiSquared = prev_chiSquared

    # Start modifying the u-parameter of the limb-darkening model, and check if quality of fit with measurement data increases
    # prev_chiSquared = cur_chiSquared + 1
    # while cur_chiSquared<prev_chiSquared:
    #     prev_chiSquared = cur_chiSquared
    #
    #     u *= 1.01
    #     [timeArray, areaArray, fluxArray, relativeFluxArray] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m, timeArray = dataframe['time'])
    #     cur_chiSquared = hap.ChiSquared(relativeFluxArray, dataframe['flux'], dataframe['error'])
    # cur_chiSquared = prev_chiSquared
    #
    # prev_chiSquared = cur_chiSquared + 1
    # while cur_chiSquared<prev_chiSquared:
    #     prev_chiSquared = cur_chiSquared
    #
    #     u /= 1.01
    #     [timeArray, areaArray, fluxArray, relativeFluxArray] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m, timeArray = dataframe['time'])
    #     cur_chiSquared = hap.ChiSquared(relativeFluxArray, dataframe['flux'], dataframe['error'])
    # cur_chiSquared = prev_chiSquared
    #
    # # Start modifying the v-parameter of the limb-darkening model, and check if quality of fit with measurement data increases
    # prev_chiSquared = cur_chiSquared + 1
    # while cur_chiSquared<prev_chiSquared:
    #     prev_chiSquared = cur_chiSquared
    #
    #     v *= 1.01
    #     [timeArray, areaArray, fluxArray, relativeFluxArray] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m, timeArray = dataframe['time'])
    #     cur_chiSquared = hap.ChiSquared(relativeFluxArray, dataframe['flux'], dataframe['error'])
    # cur_chiSquared = prev_chiSquared
    #
    # prev_chiSquared = cur_chiSquared + 1
    # while cur_chiSquared<prev_chiSquared:
    #     prev_chiSquared = cur_chiSquared
    #
    #     v /= 1.01
    #     [timeArray, areaArray, fluxArray, relativeFluxArray] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m, timeArray = dataframe['time'])
    #     cur_chiSquared = hap.ChiSquared(relativeFluxArray, dataframe['flux'], dataframe['error'])
    # cur_chiSquared = prev_chiSquared

    print ('Chi^2 = {0:.3G}; u = {1:.3G}; v = {2:.3G}; RP = {3:.3G} * R_Jup; i = {4:.3G}'.format(cur_chiSquared, u, v, RP/hac.R_Jup, math.degrees(i)))

    [timeArray, areaArray, fluxArray, relativeFluxArray] = hap.CreateLightCurve(RS = RS, RP = RP, P = P, a = a, i = i, u = u, v = v, model = m)


    # Plot values
    x_min = min(timeArray)/(60*60*24)
    x_max = max(timeArray)/(60*60*24)

    fig = plt.figure()

    # Plot a chart with the relative flux
    min_y = []
    min_y.append(min(relativeFluxArray))
    min_y.append(min(dataframe['flux']) - max(dataframe['error']))
    y_min = min(min_y)

    max_y = []
    max_y.append(max(relativeFluxArray))
    max_y.append(max(dataframe['flux']) + max(dataframe['error']))
    y_max = max(max_y)

    y_min -= 0.1*(y_max-y_min)
    y_max += 0.1*(y_max-y_min)

    ax0 = fig.add_subplot(1, 1, 1)

    ax0.plot(timeArray/(60*60*24), relativeFluxArray, color='blue', linewidth = 1)
    ax0.errorbar(dataframe['time']/(60*60*24), dataframe['flux'], yerr=dataframe['error'], color='black', fmt='.')
    # ax0.errorbar(dataframe['time']/(60*60*24), dataframe['flux'], yerr=dataframe['error'], color='black', fmt='-.')

    ax0.set_xlim(x_min, x_max)
    ax0.set_ylim(y_min, y_max)

    ax0.set_xlabel('time/days')
    ax0.set_ylabel('Relative flux')

    plt.xlabel('time/days')
    plt.show()

def PlotTransitDiagram(MS = float('nan'), RS = float('nan'), a = float('nan'), P = float('nan'), i = float('nan')):
	"""
	Plots a transit diagram of an exoplanet orbiting a host star

	IN:
	MS: Mass of host star, in kg
	RS: Radius of host star, in m
	P: Period of the orbit

	OUT:
	-
	"""

	# Calculate the orbital angular speed
	omega = hap.OrbitalAngularSpeed(P=P)[0] #Orbital speed
	#    print 'a = {0:.3G} m'.format(a)
	#    print 'P = {0:.3G} s'.format(P)
	#    print 'omega = {0:.3G} rad s^-1'.format(omega)

	# Epoch 1: 1 hour before mid-transit
	t1 = -1.0 * 60 * 60
	phi1 = hap.OrbitalPhase(omega=omega, t=t1)[0]
	s1 = hap.RadialTransitSeparation(a=a, omega=omega, t=t1, i=i)
	alpha1 = math.acos(a*math.sin(omega*t1)/s1)
	#    print 't = {0:.3G} s'.format(t1)
	#    print 'phi = {0:.3G} rad'.format(phi1)
	#    print 's = {0:.3G} m'.format(s1)
	#    print 'alpha = {0:.3G} rad'.format(alpha1)

	# Epoch 2: mid-transit
	t2 = 0.0
	phi2 = hap.OrbitalPhase(omega=omega, t=t2)[0]
	s2 = hap.RadialTransitSeparation(a=a, omega=omega, t=t2, i=i)
	alpha2 = math.acos(a*math.sin(omega*t2)/s2)
	#    print 't = {0:.3G} s'.format(t2)
	#    print 'phi = {0:.3G} rad'.format(phi2)
	#    print 's = {0:.3G} m'.format(s2)
	#    print 'alpha = {0:.3G} rad'.format(alpha2)

	# Epoch 3: 1 hour after mid-transit
	t3 = +1.0 * 60 * 60
	phi3 = hap.OrbitalPhase(omega=omega, t=t3)[0]
	s3 = hap.RadialTransitSeparation(a=a, omega=omega, t=t3, i=i)
	alpha3 = math.acos(a*math.sin(omega*t3)/s3)
	#    print 't = {0:.3G} s'.format(t3)
	#    print 'phi = {0:.3G} rad'.format(phi3)
	#    print 's = {0:.3G} m'.format(s3)
	#    print 'alpha = {0:.3G} rad'.format(alpha3)

	# Arrange data
	RP = 10 #1.0 * hac.R_Earth # Assume Earth-like planet
	radius = [RS, RP, RP, RP]
	area = []
	for r in radius:
	    area.append(math.pi*r**2)

	colors = ['yellow', 'blue', 'blue', 'blue']
	theta = [0.0, alpha1, alpha2, alpha3]
	r = [0, s1, s2, s3]

	# Plot
	ax = plt.subplot(111, projection='polar')
	c = plt.scatter(theta, r, c=colors, s=area, cmap=plt.cm.hsv)
	c.set_alpha(0.75)

	plt.show()

def Exoplanet():
    dataframe = pd.read_csv('./Data/exoplanet.eu_catalog.csv')
    plt.loglog(dataframe['mass'], dataframe['orbital_period'], 'b.')
    plt.ylabel('Orbital period / days')
    plt.xlabel('Planetary mass / $M_{J}$')
    plt.title('Exoplanet.eu')
    plt.show()

def TryFITS():
    dataframe = pd.read_fits('./Data/hlsp_clash_hst_acs_a383_f814w_v1_drz.fits')
    print (dataframe)

def VelocityEarth():
	C = 2*math.pi*hac.AU
	P = hap.KeplerThirdLaw(MS=hac.M_Sun, MP=hac.M_Earth, a=hac.AU)[1]
	v = C/P
	print ('V = {0:.3G} ms^-1'.format(v))

def VelocityMoon():
	C = 2*math.pi*hac.AU
	P = hap.KeplerThirdLaw(MS=hac.M_Earth, MP=hac.M_Earth, a=hac.AU)[1]
	v = C/P
	print ('V = {0:.3G} ms^-1'.format(v))
