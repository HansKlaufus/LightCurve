import math
import numpy as np

import hkPhysicalConstants as hpc
import hkAstronomicalConstants as hac

def GamowEnergy(mr, ZA, ZB):
	"""Calculates the Gamow Energy

	IN:
	mr: Reduced mass
	ZA: Charge of particle A
	ZB: Charge of particle B

	OUT:
	E_G: Gamow energy in J
	"""

	gamowEnergy = 2*mr*hpc.c**2 * (math.pi*1/137*ZA*ZB)**2
	return gamowEnergy

def GamowPeak(EG, T):
	"""Calculates the Gamow peak

	IN:
	EG: Gamow Energy
	T: Temperature in K

	OUT:
	E0: Gamow peak in J
	"""

	gamowPeak = (EG*(hpc.k*T/2)**2)**(1.0/3.0)
	return gamowPeak

def GamowWidth(EG, T):
	"""Calculates the Gamow width

	IN:
	EG: Gamow Energy
	T: Temperature in K

	OUT:
	Delta: Gamow width in J
	"""

	gamowWidth = 4.0/(3.0**0.5 * 2.0**(1.0/3.0)) * EG**(1.0/6.0) * (hpc.k*T)**(5.0/6.0)
	return gamowWidth

def ReducedMass(mA, mB):
	"""Calculates the apparent reduced mass of two particles accelerating towards eachother.

	IN:
	mA: Mass of particle A
	mB: Mass of particle B

	OUT:
	mr: reduced mass
	"""

	mA = mA * 1.0
	mB = mB * 1.0

	reducedMass = (mA*mB)/(mA+mB)
	return reducedMass

def CrossSection(E, EG, SE):
	"""Calculates the cross section of a fusion process

	IN:
	E: Energy
	EG: Gamow Energy
	SE: S-factor at energy E
	"""

	crossSection = SE/E * math.exp(-1*(EG/E)**0.5)
	return crossSection

def FusionRate(AA, AB, ZA, ZB, XA, XB, rho, T, S0, verbose=False):
	"""Calculates the rate of two nuclei fusing per mass fraction B.

	AA: Atomic mass of particle A
	AB: Atomic mass of particle B
	ZA: Charge of particle A
	ZB: Charge of particle B
	XA: Mass fraction of particle A
	XB: Mass fraction of particle B
	rho: Density
	T: Temperature
	S0: S(E0): S-Factor: Strength of fusion interaction, keV barns
	"""

	Ar = ReducedMass(AA, AB)     # Reduced Atomic Mass

	mA = AA * hpc.u		     # Mass of particle A
	mB = AB * hpc.u		     # Mass of particle B
	mr = ReducedMass(mA, mB)      # Reduced Mass

	EG = GamowEnergy(mr, ZA, ZB) # Gamow Energy
	temp = EG / (4*hpc.k*T)

	if verbose:
		print ('Ar: {0:.3G}'.format(Ar))
		print ('mA: {0:.3G}'.format(mA))
		print ('mB: {0:.3G}'.format(mB))
		print ('mr: {0:.3G}'.format(mr))
		print ('EG: {0:.3G}'.format(EG))
		print ('temp: {0:.3G}'.format(temp))

	fusionRate = 6.48E-24/(Ar*ZA*ZB) * (rho * XA)/mA * (rho * XB)/mB * S0 * (temp)**(2.0/3.0) * math.exp(-3*temp**(1.0/3.0))
	if (AA==AB and ZA==ZB):
		# Correct number counts for identical particles (See SEN page 61)
		fusionRate = fusionRate/2

	return fusionRate

def DerivativeFusionRateToTemperature(AA, AB, ZA, ZB, T):
	"""Calculates the dependence of the temperature-exponent nu on the mass and charge of the nuclei.

	AA: Atomic mass of particle A
	AB: Atomic mass of particle B
	ZA: Charge of particle A
	ZB: Charge of particle B
	T: Temperature
	"""

	mA = AA * hpc.u		     # Mass of particle A
	mB = AB * hpc.u		     # Mass of particle B
	mr = ReducedMass(mA, mB)      # Reduced Mass

	EG = GamowEnergy(mr, ZA, ZB) # Gamow Energy

	nu = (EG/(4*hpc.k*T))**(1.0/3.0) - 2.0/3.0
	return nu

def ApproximateHydrogenBurningLifetime(M):
	"""Calculates an approximate to the hydrogen burning lifetimes of stars

	IN:
	M: Mass of star, in kg

	OUT:
	lifeTime of star
	"""

	factor = -2.5 # For low mass stars
	if (M > M_Sun):
		factor = -2.0

	lifeTime = M**factor
	return lifeTime

def QuantumConcentration(m, T):
	"""Calculates the non-relativistic quantum concentration for a particle.

	IN:
	m: mass of the particle, in kg
	T: Temperature, in K

	OUT:
	n_QNR
	"""

	n_QNR = (2*math.pi*hpc.k*T/hpc.h**2)**(3.0/2.0)
	return n_QNR

def ChemicalPotential(m, T, n):
	"""Calculates teh chemical potential of a particle.

	IN:
	m: mass of the particle, in kg
	T: Temperature, in K
	n: number density, in m^-3
	g_s: number of polarizations

	OUT:
	chemicalPotential, in J
	"""

	n_QNR = QuantumConcentration(m, T)
	chemicalPotential = m*hpc.c**2 - hpc.k * T * math.log(g_s * n_QNR/n)
	return chemicalPotential

def DeBroglieWavelength(m, T):
	deBroglieWavelength = hpc.h / (3.0 * m * hpc.k * T)**0.5
	return deBroglieWavelength

def JeansMass(R, T, m):
	"""Calculates the Jeans mass of a gas cloud

	IN:
	R: Radius of the gas cloud, in m
	T: Temperature of the gas cloud, in K
	m: average particle mass, in kg

	OUT:
	M_J: Jeans mass in kg
	"""

	M_J = (3*hpc.k*T)/(2*hpc.G*m)*R
	return M_J

def JeansDensity(M, T, m):
	"""Calculates the Jeans density of a gas cloud

	IN:
	M: Mass of the gas cloud, in m
	T: Temperature of the gas cloud, in K
	m: average particle mass, in kg

	OUT:
	Rho_J: Jeans mass in kg
	"""

	Rho_J = ((3*hpc.k*T)/(2*hpc.G*m))**3 * 3/(4*math.pi*M**2)
	return Rho_J

def FreeFallTime(rho):
	"""Calculates the free fall time of a collapsing gas cloud.

	IN:
	rho: density of the gas cloud; in kg m^-1

	OUT:
	t: the free fall time
	"""

	t = ((3*math.pi)/(32*hpc.G*rho))**0.5
	return t

def CalculateLuminosity2(R, T):
	"""Calculates the luminosity of a spherical black body

	IN:
	R: radius, in m
	T: temperature, in K

	OUT:
	L: luminosity, in W
	"""

	L = 4 * math.pi * R**2 * hpc.sigma * T**4
	return L

def CalculateLuminosity(L = float('nan'), R = float('nan'), T = float('nan')):
	"""Calculates the luminosity of a spherical black body

	IN:
	L: luminosity, in W
	R: radius, in m
	T: temperature, in K

	OUT:
	[L, R, T]: list of luminosity, radius, temperature
	"""

	if (math.isnan(L)):
		# Calculate L
		L = 4*math.pi*R**2*hpc.sigma*T**4
	elif (math.isnan(R)):
		# Calculate R
		R = (L/(4*math.pi*hpc.sigma*T**4))**(0.5)
	elif (math.isnan(T)):
		# Calculate T
		T = (L/(4*math.pi*R**2*hpc.sigma))**(0.25)
	else:
		print ('CalculateLuminosity: WARNING: Nothing to solve')

	if(math.isnan(L) or math.isnan(R) or math.isnan(T)):
		print ('CalculateLuminosity: ERROR: Could not solve')

	return [L, R, T]

def WiensDisplacementLaw(T = float('nan'), l = float('nan')):
	"""Calculates the wavelength based on temperature for a black body

	IN:
	T: Temperature, in K
	l: Lambda; wavelength, in m

	OUT:
	[l, T]: list of wavelength, temperature
	"""

	if(math.isnan(l)):
		l = 2.9E-3 / T
	elif(mat.isnan(T)):
		T = 2.9E-3 / l
	else:
		print ('WiensDisplacementLaw: WARNING: Nothing to solve')

	if(math.isnan(l) or math.isnan(T)):
		print ('WiensDisplacementLaw: ERROR: Could not solve')

	return [l, T]

def KeplerThirdLaw(a = float('nan'), P = float('nan'), MS = float('nan'), MP = float('nan')):
	""" Calculates orbital periods, masses, etc. using Kepler's third law.

	IN:
	a: semi-major axis of the orbit, in m
	P: period of the orbid, in s
	MS: mass of the host star, in kg
	MP: mass of the planet, in kg

	OUT:
	[a, P, MS, MP]: list of all the above
	"""

	if(math.isnan(a)):
		a = ((P**2 * hpc.G*(MS+MP))/(4*math.pi**2))**(1.0/3)
	elif(math.isnan(P)):
		P = ((a**3 * 4*math.pi**2)/(hpc.G*(MS+MP)))**(1.0/2)
	elif(math.isnan(MS)):
		MS = (a**3 * 4*math.pi**2)/(hpc.G) - MP
	elif(math.isnan(MP)):
		MP = (a**3 * 4*math.pi**2)/(hpc.G) - MS
	else:
		print ('KeplerThirdLaw: WARNING: Nothing to solve')

	if(math.isnan(a) or math.isnan(P) or math.isnan(MS) or math.isnan(MP)):
		print ('KeplerThirdLaw: ERROR: Could not solve')

	return [a, P, MS, MP]

def TransitProbability(p = float('nan'), RS = float('nan'), RP = float('nan'), a = float('nan'), e = float('nan')):
	"""
	Calculates the transit probability of a planet around a host star

	IN:
	p: probability, dimensionless factor within [0, 1]
	RS: radius of the host star, in m
	RP: radius of the planet, in m
	a: semi-major axis of the orbit, in m
	e: eccentricity of the orbit

	OUT:
	[p, RS, a, RP, e]: list of all of the above
	"""

	if(math.isnan(p)):
		p = (RS+RP)/(a*(1-e**2))
	elif(math.isnan(RS)):
		RS = p*a*(1-e**2)-RP
	elif(math.isnan(a)):
		a = (RS+RP)/(p*(1-e**2))
	elif(math.isnan(RP)):
		RP = p*a*(1-e**2)-RS
	elif(math.isnan(a)):
		e = (1 - (RS+RP)/(p*a))**0.5
	else:
		print ('TransitProbability: WARNING: Nothing to solve')

	if(math.isnan(p) or math.isnan(RS) or math.isnan(a) or math.isnan(RP) or math.isnan(e)):
		print ('TransitProbability: ERROR: Could not solve')

	return [p, RS, a, RP, e]

def ApproximateTransitTime(t = float('nan'), P = float('nan'), RS = float('nan'), a = float('nan')):
	"""Calculates the approximate transit time, using equation 2.6 in TE

	IN:
	t: transit duration, in s
	P: period of the planet, in s
	RS: radius of the host star, in m
	a: semi-major axis of the orbit, in m

	OUT:
	[t, P, RS, a]: list of all of the above
	"""

	#TODO: Extend this formula according to equation 3.4 in TE

	if(math.isnan(t)):
		t = P/math.pi * math.sin(RS/a)
	elif(math.isnan(P)):
		P = t * math.pi * math.sin(RS/a)
	elif(math.isnan(RS)):
		RS = a * math.asin(t*math.pi/P)
	elif(math.isnan(a)):
		a = RS / math.asin(t*math.pi/P)
	else:
		print ('TransitTime: WARNING: Nothing to solve')

	if(math.isnan(t) or math.isnan(P) or math.isnan(RS) or math.isnan(a)):
		print ('TransitTime: ERROR: Could not solve')

	return [t, P, RS, a]

def AmplitudeReflexRadialVelocity(ARV = float('nan'), MP= float('nan'), MS= float('nan'), a = float('nan'), i= float('nan'), P= float('nan'), e= float('nan')):
	"""
	Calculates the smplitude of the reflex radial velocity of a star induced by an orbiting planet.

	IN:
	ARV: Amplitude of the radial reflex velocity
	MP: mass of planet, in kg
	MS: mass of host star, in kg
	a: semi-major axis of bary-centric orbit, in m
	i: inclination of bary-centric orbit, in radians.
	P: period of bary-centric orbit, in s
	e: eccentricity of bary-centric orbit, in m

	OUT:
	[ARV, MP, MS, a, i, P, e]: list of all the above
	"""

	ARV = (2*math.pi*a*MP*math.sin(i))/((MP+MS)*P*math.sqrt(1 - e**2))
	return [ARV, MP, MS, a, i, P, e]

def CircularOrbitalSpeed(v = float('nan'), a = float('nan'), P = float('nan')):
	"""
	Calculates the orbital speed in case of a circular orbit

	IN:
	v: velocity, km s^-1
	a: semi-major axis, m
	P: period, s

	OUT:
	[v, a, P]: List of all of the above
	"""

	if(math.isnan(v)):
		v = 2*math.pi*a/P
	elif(math.isnan(a)):
		a = v*P / (2*math.pi)
	elif(math.isnan(P)):
		P = 2*math.pi*a/v
	else:
		print ('CircularOrbitalSpeed: WARNING: Nothing to solve')

	if(math.isnan(v) or math.isnan(a) or math.isnan(P)):
		print ('CircularOrbitalSpeed: ERROR: Could not solve')

	return [v, a, P]

def LimbDarkening(u = float('nan'), v = float('nan'), gamma = float('nan'), model = 'lin'):
	"""
	Calculates the ratio between the emergent intensity and the emitted intensity using linear, logarithmis, quadratic and cubic models.

	IN:
	u: first limb darkening coefficient.
	v: second limb darkneing coefficient.
	gamma: , in radians
	model: either
		'lin': linear
		'log': logarithmic
		'qua': quadratic
		'cub': cubic

	OUT:
	I(mu)/I(1): the ratio between the emergent intensity and the emitted intensity.
	"""

	if(math.isnan(u)):
		print ('LimbDarkening: ERROR: limb darkening coefficient u not specified')
	else:
		mu = math.cos(gamma)

		ratio = -1.0
		if(model[0:3] == 'lin'):
			ratio = 1 - u*(1-mu)
		else:
			if(math.isnan(v)):
				print ('LimbDarkening: ERROR: limb darkening coefficient v not specified')
			else:
				if(model[0:3] == 'log'):
					ratio = 1 - u*(1-mu)-v*mu*math.log(mu)
				elif(model[0:3] == 'qua'):
					ratio = 1 - u*(1-mu) - v*(1-mu)**2
				elif(model[0:3] == 'cub'):
					ratio = 1 - u*(1-mu) - v*(1-mu)**3
				else:
					print ('LimbDarkening: ERROR: unknown model specified: ' + model)

		return ratio

def OrbitalPhase(phi = float('nan'), omega = float('nan'), t = float('nan')):
	"""
	Calculates the orbital phase

	IN:
	phi: orbital phase, in radians
	omega: orbital speed, in radians s^-1
	t: time, in s

	OUT:
	[phi, omega]: list of the above
	"""

	if(math.isnan(phi)):
		phi = omega*t / (2*math.pi)
	elif(math.isnan(omega)):
		omega = phi*2*math.pi/t
	elif(math.isnan(t)):
		t = phi*2*math.pi/omega
	else:
		print ('OrbitalPhase: WARNING: Nothing to solve')

	if(math.isnan(phi) or math.isnan(omega) or math.isnan(t)):
		print ('OrbitalPhase: ERROR: Could not solve')

	return [phi, omega, t]

def OrbitalAngularSpeed(omega = float('nan'), P = float('nan')):
	"""
	Calculats the orbital angular speed.

	IN:
	omega: the orbital angular speed, in radians s^-1
	P: the orbital period, in s

	OUT:
	[omega, P]: list of the above
	"""

	if(math.isnan(omega)):
		omega = 2*math.pi/P
	elif(math.isnan(P)):
		P = 2*math.pi/omega
	else:
		print ('OrbitalAngularSpeed: WARNING: Nothing to solve')

	if(math.isnan(omega) or math.isnan(P)):
		print ('OrbitalAngularSpeed: ERROR: Could not solve')

	return [omega, P]

def RadialTransitSeparation(a = float('nan'), omega=float('nan'), t = float('nan'), i = float('nan')):
	"""
	Calculates the the radial separation of a planet transiting a host star.

	IN:
	a: semi-major axis, in m
	omega: orbital angular speed, in rad s^-1
	t: time, in s
	i: inclination, in rad

	OUT:
	s: radial separation, in m
	"""

	s = a * math.sqrt((math.sin(omega*t))**2 + ((math.cos(i))**2 * (math.cos(omega*t))**2))
	return s

def EclipsedArea(RS = float('nan'), RP = float('nan'), s = float('nan')):
	"""
	Calculates the eclipsed are of a star orbited by a planet

	IN:
	RS: Radius of the host star, in m
	RP: Radius of the orbiting planet, in m
	s: RadialTransitSeparation, in ,

	OUT:
	A: Eclipsed are, in m^2
	"""

	try:
		p = RP / RS
		xi = s / RS
	except ValueError:
		print ('EclipsedArea: ERROR: Value error')
		print ('RS = {0:.3G}'.format(RS))
		raise ValueError

	if(1+p < xi):
		A = 0.0
	elif((1-p < xi) and (xi <= 1+p)):
		try:
			alpha1 = math.acos((p**2 + xi**2 - 1)/(2*xi*p))
			alpha2 = math.acos((1 + xi**2 - p**2)/(2*xi))
		except ValueError:
			print ('EclipsedArea: ERROR: Value error')
			print ('s = {0:.3G}'.format(s))
			print ('p = {0:.3G}'.format(p))
			print ('xi = {0:.3G}'.format(xi))
			print ('2*xi*p = {0:.3G}'.format(2*xi*p))
			print ('(p**2 + xi**2 - 1)/(2*xi*p) = {0:.3G}'.format((p**2 + xi**2 - 1)/(2*xi*p)))
			print ('2*xi = {0:.3G}'.format(2*xi))
			print ('(1 + xi**2 - p**2)/(2*xi) = {0:.3G}'.format((1 + xi**2 - p**2)/(2*xi)))
			raise ValueError

		A = RS**2 * (p**2 * alpha1 + alpha2 - 0.5*math.sqrt(4*xi**2 - (1 + xi**2 - p**2)**2))
	elif(1-p >= xi):
		A = math.pi * p**2 * RS**2
	else:
		print ('EclipsedArea: ERROR: Could not solve')

	return A

def TotalRadiationFlux(RS = float('nan'), u = float('nan'), v = float('nan'), model = 'lin'):
	""" Calculate the total radiation flux of a star """

	# Constants
	dr = 0.1*RS # Delta radius, dimensionless

	# Calculate total star radiation flux
	radiusArray = np.arange(dr, RS+dr, dr)
	F = 0.0
	for r in radiusArray:
		# Calculate the limb darkening
		gamma = math.atan(r / RS)
		I = LimbDarkening(u=u, v=v, gamma=gamma, model=model)

		# Add too Transit Flux Depth
		F += I*2*math.pi*r*dr

	return F

def CreateLightCurve(RS = float('nan'), RP = float('nan'), P = float('nan'), a = float('nan'), i = float('nan'), u = float('nan'), v = float('nan'), model = 'lin', timeArray = None):
	"""
	Creates a model light curve

	IN:
	RS: Radius of host star, in m
	RP: Radius of orbiting planet, in m
	P: Period of orbit, in s
	a: semi-major axis, in m
	i: inclination, in radians
	u: First parameter of limb-darkening law
	v: Second parameter of limb-darkening law
	model: limb-darkening law to be used, either:
		'lin': linear
		'log': logarithmic
		'qua': quadratic
		'cub': cubic
	timeArray: array holding time intervals, optional

	OUT:
	[timeArray, areaArray, fluxArray, relativeFluxArray]
	"""

	# Constants
	dt = 1.0 #* 60# Delta time, in minutes
	dr = 0.01 * RS# Delta radius, dimensionless

	# Calculate the orbital angular speed
	w = OrbitalAngularSpeed(P=P)[0]

	# Calculate the total radiation flux
	F = TotalRadiationFlux(RS=RS, u=u, v=v, model=model)
	# print 'Total star radiation flux: {0:.3G}'.format(F)

	# Initialize timeArray
	if(timeArray is None):
		timeArray = np.arange(-P/2, P/2, dt)

	# Initialize radiusArray
	radiusArray = np.arange(dr, RS+dr, dr)

	# Calculate time based transit depth
	areaArray = []
	fluxArray = []
	relativeFluxArray = []

	for t in timeArray:
		# Calculate the orbital phase angle
		phi = math.degrees(w*t)

		# Calculate the radial transit separation
		s = RadialTransitSeparation(a=a, omega=w, t=t, i=i)

		# Calculate the total eclipsed area
		Ae = EclipsedArea(RS=RS, RP=RP, s=s)

		prev_Ae = 0.0
		cur_Ae = 0.0
		dF = 0.0 # Transit Flux Depth

		# Loop over radii, but only if there is a (partial) eclipse
		if((phi+90)%360 < 180) and (Ae > 0):
			for r in radiusArray:
				# Calculate the eclipsed area at r=r
				cur_Ae = EclipsedArea(RS=r, RP=RP, s=s)

				# Calculate the limb darkening
#				gamma = math.atan(r / RS)
				gamma = math.asin(r / RS)
				I = LimbDarkening(u=u, v=v, gamma=gamma, model=model)

				# Add to Transit Flux Depth
				dF += I*(cur_Ae-prev_Ae)

				# Debug information
				# print 'r = {0:.3E}; Ae = {1:.3E}; I = {2:.3E}; dF = {3:.3E}; F = {4:.3E}; (F-dF)/F = {5:.3E}'.format(r/RS, (cur_Ae-prev_Ae), I, dF, F, (F-dF)/F)

				# Store cur_Ae in prev_Ae
				prev_Ae = cur_Ae

		# Add calculated values to arrays
		areaArray.append(cur_Ae)
		fluxArray.append(dF)
		relativeFluxArray.append((F-dF)/F)

	return [timeArray, areaArray, fluxArray, relativeFluxArray]

def ChiSquared(modelArray, measurementArray, uncertaintyArray):
	"""
	Calculates the goodness of the fit of a model to the measurements.

	IN:
	modelArray: array holding the modeled values at time t
	measurementArray: array holding the measured values at time t
	uncertaintyArray: the uncertainty in the measurements

	OUT:
	chi^2: goodness of the fit of a model
	"""


	if (len(modelArray) != len(measurementArray) or (len(measurementArray) != len(uncertaintyArray))):
		chiSquared = -1.0
		print ('ERROR in ChiSquared: arrays have different lengths')
	else:
		chiSquared = 0.0
		for model, measurement, uncertainty in zip(modelArray, measurementArray, uncertaintyArray):
			chiSquared += ((measurement - model)/uncertainty)**2

	return chiSquared

def FermiTemperature(TF = float('nan'), n = float('nan'), m = float('nan')):
	"""
	Calculates the Fermi temperature at which electrons become degenerate.

	IN:
	TF: Fermi temperature, in K
	n: number density of particle
	m: mass of particle, in kg

	OUT:
	[TF, n, m]: list of the above
	"""

	if(math.isnan(TF)):
		TF = n**(2/3.0)*hpc.h**2/(2*math.pi*m*hpc.k)
	elif(math.isnan(n)):
		n = ((2*math.pi*m*hpc.k*TF)/hpc.h**2)**(3/2.0)
	elif(math.isnan(m)):
		m = n**(2/3.0)*hpc.h**2/(2*math.pi*TF*hpc.k)
	else:
		print ('FermiTemperature: WARNING: Nothing to solve')

	if(math.isnan(TF) or math.isnan(n) or math.isnan(m)):
		print ('FermiTemperature: ERROR: Could not solve')

	return [TF, n, m]

def KelvinHelmholtzContractionTime(tau = float('nan'), M = float('nan'), L = float('nan'), R = float('nan')):
	"""
	Calculates the Kelvin-Helmholtz contraction time

	IN:
	tau: contartion time, in s
	M: Mass of object
	R: Radius o object, in m
	L: Luminosity, in Js^-1

	OUT:
	[tau, M, R, L]: list of the above
	"""

	if(math.isnan(tau)):
		tau = (hpc.G*M**2)/(L*R)
	elif(math.isnan(M)):
		M = math.sqrt((tau*L*R)/(hpc.G))
	elif(math.isnan(L)):
		L = (hpc.G*M**2)/(tau*R)
	elif(math.isnan(R)):
		R = (hpc.G*M**2)/(tau*L)
	else:
		print ('KelvinHelmholtzContractionTime: WARNING: Nothing to solve')

	if(math.isnan(tau) or math.isnan(M) or math.isnan(R)):
		print ('KelvinHelmholtzContractionTime: ERROR: Could not solve')

	return [tau, M, L, R]

def EquilibriumTemperature(T = float('nan'), A = float('nan'), FS = float('nan')):
	"""
	Calculate the equilibrium temperature of a planet orbiting its host star.

	IN:
	T: Equilibrium temperature, in K
	A: Albedo of the planet
	FS: Radiation flux of the host star.

	OUT:
	[T, A, FS]: list of the above.
	"""

	if(math.isnan(T)):
		T = ((1-A)*FS/(4*hpc.sigma))**(0.25)
	elif(math.isnan(A)):
		A = -1*(T**4 * 4*hpc.sigma/FS - 1)
	elif(math.isnan(FS)):
		FS = T**4 * 4*hpc.sigma/(1-A)
	else:
		print ('EquilibriumTemperature: WARNING: Nothing to solve')

	if(math.isnan(T) or math.isnan(A) or math.isnan(FS)):
		print ('EquilibriumTemperature: ERROR: Could not solve')

	return [T, A, FS]

def RadiationFlux(F = float('nan'), L = float('nan'), d = float('nan')):
	"""
	Calculate the radiation flux of a star.

	IN:
	F: Radiation flux, in J m^-2 s^-1
	L: Luminosity, in J s^-1
	d: distance, in m

	OUT:
	[F, L, d]: list of the above.
	"""

	if(math.isnan(F)):
		F = L / (4*math.pi*d**2)
	elif(math.isnan(L)):
		L = F*4*math.pi*d**2
	elif(math.isnan(d)):
		d = math.sqrt(L/(4*math.pi*F))
	else:
		print ('RadiationFlux: WARNING: Nothing to solve')

	if(math.isnan(F) or math.isnan(L) or math.isnan(d)):
		print ('RadiationFlux: ERROR: Could not solve')

	return [F, L, d]

def WiensApproximation(B = float('nan'), T = float('nan'), l = float('nan')):
	"""
	Calculates the spectral radiance using the Wien's approximation.

	IN:
	B: Spectral radiance, in W m^-2 m^-1 sr^-1
	T: Temperature, in K
	l: Wavelength, in m

	OUT:
	[B, T, Lambda]: list of the above.
	"""

	if(math.isnan(B)):
		B = (2*hpc.h*hpc.c**2)/(l**5) * math.exp(-hpc.h*hpc.c/(l*hpc.k*T))
	elif(math.isnan(T)):
		print ('WiensApproximation: WARNING: Not implemented')
	elif(math.isnan(l)):
		print ('WiensApproximation: WARNING: Not implemented')
	else:
		print ('WiensApproximation: WARNING: Nothing to solve')

	if(math.isnan(B) or math.isnan(T) or math.isnan(l)):
		print ('WiensApproximation: ERROR: Could not solve')

	return [B, T, l]

def RayleighJeansLaw(B = float('nan'), T = float('nan'), l = float('nan')):
	"""
	Calculates the spectral radiance using the Rayleigh-Jeans law.

	IN:
	B: Spectral radiance, in W m^-2 m^-1 sr^-1
	T: Temperature, in K
	l: Wavelength, in m

	OUT:
	[B, T, Lambda]: list of the above.
	"""

	if(math.isnan(B)):
		B = 2*hpc.c*hpc.k*T/l**4
	elif(math.isnan(T)):
		T = B*l**4 / (2*hpc.c*hpc.k)
	elif(math.isnan(l)):
		l = (2*hpc.c*hpc.k*T/B)**0.25
	else:
		print ('RayleighJeansLaw: WARNING: Nothing to solve')

	if(math.isnan(B) or math.isnan(T) or math.isnan(l)):
		print ('RayleighJeansLaw: ERROR: Could not solve')

	return [B, T, l]

def SpectralRadiance(B = float('nan'), T = float('nan'), l = float('nan')):
	"""
	Calculates the spectral radiance using Plank's law.

	IN:
	B: Spectral radiance, in W m^-2 m^-1 sr^-1
	T: Temperature, in K
	l: Wavelength, in m

	OUT:
	[B, T, Lambda]: list of the above.
	"""

	if(math.isnan(B)):
		try:
			# Next line is equivalent to 'B = (2*hpc.h*hpc.c**2)/(l**5) * 1/math.expm1(((hpc.h*hpc*c)/(l*hpc.k*T))-1)'
			B = (2*hpc.h*hpc.c**2)/(l**5) * 1/math.expm1((hpc.h*hpc.c)/(l*hpc.k*T))
		except OverflowError:
			x = (hpc.h*hpc.c)/(l*hpc.k*T)
			if x < 0.01:
				# Use Rayleigh-Jeans Law: use Taylor expansion: if x<<1 then exp(x) = 1 + x
				B = RayleighJeansLaw(T=T, l=l)[0]
			elif x > 100:
				# Wien's approximation
				B = WiensApproximation(T=T, l=l)[0]
			else:
				print ('x = {0:.3G}'.format(x))
				raise OverflowError

	elif(math.isnan(T)):
		print ('PlanckFunction: ERROR: Not implemented')
	elif(math.isnan(l)):
		print ('PlanckFunction: ERROR: Not implemented')
	else:
		print ('PlanckFunction: WARNING: Nothing to solve')

	if(math.isnan(B) or math.isnan(T) or math.isnan(l)):
		print ('PlanckFunction: ERROR: Could not solve')

	return [B, T, l]

def DaySideTemperature(Td = float('nan'), P = float('nan'), A = float('nan'), RS = float('nan'), a = float('nan'), Teff = float('nan')):
	"""
	Calculates the day-side temperature of an orbiting planet.

	IN:
	Td: day side temperature, in K
	P: fraction of absorbed energy that is transported to the night side of the planet.
	A: Albedo
	RS: Radius of the host star, in m
	a: semi-major axis of the orbit, in m
	Teff: Effective temperature of the host star, in K

	OUT:
	[Td, P, A, RS, a, Teff]: all of the above
	"""

	if (math.isnan(Td)):
		Td = ((1-P)*(1-A)*RS**2/(2*a**2)*Teff**4)**(0.25)
	elif (math.isnan(P)):
		P = 1 - Td**4 * (2*a**2)/((1-A)*RS**2*Teff**4)

	return [Td, P, A, RS, a, Teff]

def NightSideTemperature(Tn = float('nan'), P = float('nan'), A = float('nan'), RS = float('nan'), a = float('nan'), Teff = float('nan')):
	"""
	Calculates the night-side temperature of an orbiting planet.

	IN:
	Tn: day side temperature, in K
	P: fraction of absorbed energy that is transported to the night side of the planet.
	A: Albedo
	RS: Radius of the host star, in m
	a: semi-major axis of the orbit, in m
	Teff: Effective temperature of the host star, in K

	OUT:
	[Tn, P, A, RS, a, Teff]: all of the above
	"""

	Tn = (P*(1-A)*RS**2/(2*a**2)*Teff**4)**(0.25)

	return [Tn, P, A, RS, a, Teff]

def CircularizationTimescale(t = float('nan'), Q = float('nan'), kd = float('nan'), a = float('nan'), MS = float('nan'), MP = float('nan'), RP = float('nan')):
	"""
	Calculates the circularization timescale for a planet orbiting its host star.

	IN:
	tc: circularization timescale
	Q: Tidal quality factor
	kd: dynamical Love number
	a: semi-major axis of the orbit, in m
	MS: mass of the host star, in kg
	MP: mass of the planet, in kg
	RP: Radius of the planet, in m

	OUT:
	[tc, QP, kdP, a, MS, MP, RP]: List of the above
	"""

	tc = 2.0/21.0 * Q/Kd * (a**3/(hpc.G*MS))**(0.5) * MP/MS * (a/RP)**5

	return [tc, Q, Kd, a, MS, MP, RP]

def HillRadius(R = float('nan'), a = float('nan'), MP = float('nan'), MS = float('nan')):
	"""
	Calculates the Hill radius:
	the maximum distance at which a moon can orbit a aplanet and still remain in a stable orbit.

	IN:
	R: Hill radius, in m
	a: semi-major axis, in m
	MP: mass orbiting planet, in kg
	MS: mass host star, in kg

	OUT:
	[R, a, MP, MS]: List of the above
	"""

	if(math.isnan(R)):
		R = a*(MP/(3*MS))**(1.0/3.0)
	elif(math.isnan(a)):
		a = R*(MP/(3*MS))**(-1.0/3.0)
	elif(math.isnan(MP)):
		MP = (R/a)**3 * 3*MS
	elif(math.isnan(MS)):
		MS = 3*MP/(R/a)**3
	else:
		print ('HillRadius: WARNING: Nothing to solve')

	if(math.isnan(R) or math.isnan(a) or math.isnan(MP) or math.isnan(MS)):
		print ('HillRadius: ERROR: Could not solve')

	return [R, a, MP, MS]

def RocheLimit(d = float('nan'), RM = float('nan'), MP = float('nan'), MM = float('nan')):
	"""
	Calculates the Roche limit:
	the minimum distance at which a moon can orbit a planet.

	IN:
	d: Roche limit, in m
	RM: Radius of moon, in m
	MP: Mass of hosting planet, in kg
	MM: Mass of moon, in kg

	OUT:
	[d, RM, MP, MM]: List of the above
	"""

	if(math.isnan(d)):
		d = RM*(2*MP/MM)**(1.0/3.0)
	elif(math.isnan(RM)):
		MM = d*(2*MP/MM)**(-1.0/3.0)
	elif(math.isnan(MP)):
		MP = (dr/RM)**3*MM/2
	elif(math.isnan(MM)):
		MM = 2*MP / (dr/RM)**3
	else:
		print ('RocheLimit: WARNING: Nothing to solve')

	if(math.isnan(d) or math.isnan(RM) or math.isnan(MP) or math.isnan(MM)):
		print ('RocheLimit: ERROR: Could not solve')

	return [d, RM, MP, MM]

def HubbleLaw(z = float('nan'), d = float('nan')):
	"""
	Calculates a galaxy's redshift based on its distance.

	IN:
	z: redshift
	d: distance, in m

	OUT:
	[z, d]: list of the above
	"""

	H0 =  1E3 * hac.H0/(1E6*hac.parsec) # m s^-1 m^-1

	if(math.isnan(z)):
		z = H0/hpc.c * d
	elif(math.isnan(d)):
		d = hpc.c/H0 * z
	else:
		print ('HubbleLaw: WARNING: Nothing to solve')

	if(math.isnan(z) or math.isnan(d)):
		print ('HubbleLaw: ERROR: Could not solve')

	return [z, d]

def RedShift(z = float('nan'), l0 = float('nan'), l = float('nan')):
	"""
	Calculates the redshift based on the measured wavelength.

	IN:
	z: redshift
	l0: rest wavelength
	l: observed wavelength

	OUT:
	[z, l0, l]: list of the above
	"""

	if(math.isnan(z)):
		z = (l-l0)/l
	elif(math.isnan(l0)):
		l0 = l *(1-z)
	elif(math.isnan(l)):
		l = l0/(1-z)
	else:
		print ('RedShift: WARNING: Nothing to solve')

	if(math.isnan(z) or math.isnan(l0) or math.isnan(l)):
		print ('RedShift: ERROR: Could not solve')

	return [z, l0, l]
