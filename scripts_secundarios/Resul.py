# tabla_outflow_ngc4750 para mi TFG
#
# Galaxy | Lobs,out | Aout | Rout | Δv | vout | sigma |
# log(Mout) | log(Mdot) | log(eta) | log(Eout) | log(Edot) | ne
#
# Filas:
# Halpha broad
# [OIII] broad

import numpy as np
from astropy.io import fits

# Extraemos los mapas


HALPHA_FITS = "/home/rauladeva/PyCharmMiscProject/maps_halpha.fits"
HBETA_FITS  = "/home/rauladeva/PyCharmMiscProject/maps_hbeta.fits"
OIII_FITS   = "/home/rauladeva/PyCharmMiscProject/maps_oiii.fits"



# CONSTANTES ADOPTADAS


distance_mpc = 24.71          # z=0.005404
scale_kpc_arcsec = 0.129     # NGC 4750: 129 pc/arcsec = 0.129 kpc/arcsec
pixscale_arcsec = 0.4
ne = 686.0                   # cm^-3, valor de Hermosa-Muñoz

snr_min = 3.0

MPC_TO_CM = 3.085677581e24
MSUN_TO_G = 1.98847e33
KM_TO_CM = 1e5
YR_TO_S = 365.25 * 24 * 3600
KM_S_KPC_TO_YR = 1.022712e-9

SFR_FACTOR = 7.9e-42  # Kennicutt IMF Salpeter

# ERRORES ADOPTADOS


# SFR Cano-Diaz
SFR = 0.12

SFR_ERR = 0.03

ne_err = 0.30 * ne

# Error del radio: 1 píxel
R_ERR_KPC = pixscale_arcsec * scale_kpc_arcsec


# mapas

with fits.open(HALPHA_FITS) as hdul:
    flux_ha_n = hdul["FLUX_HA_N"].data.copy()
    flux_ha_b = hdul["FLUX_HA_B"].data.copy()
    v_ha_b = hdul["VEL_HA_B"].data.copy()
    sigma_ha_b = hdul["SIGMA_HA_B"].data.copy()
    snr_ha_b = hdul["SNR_HA_B"].data.copy()

with fits.open(HBETA_FITS) as hdul:
    flux_hb_n = hdul["FLUX_HB_N"].data.copy()
    flux_hb_b = hdul["FLUX_HB_B"].data.copy()
    snr_hb_b = hdul["SNR_HB_B"].data.copy()

with fits.open(OIII_FITS) as hdul:
    flux_oiii_b = hdul["FLUX_OIII_B"].data.copy()
    v_oiii_b = hdul["VEL_OIII_B"].data.copy()
    sigma_oiii_b = hdul["SIGMA_OIII_B"].data.copy()
    snr_oiii_b = hdul["SNR_OIII_B"].data.copy()


# funciones

def luminosity(flux):
    d_cm = distance_mpc * MPC_TO_CM
    return 4.0 * np.pi * d_cm**2 * flux


def calzetti_k(wave_micron):
    Rv = 4.05
    lam = wave_micron

    if 0.12 <= lam < 0.63:
        return 2.659 * (-2.156 + 1.509/lam - 0.198/lam**2 + 0.011/lam**3) + Rv
    elif 0.63 <= lam <= 2.20:
        return 2.659 * (-1.857 + 1.040/lam) + Rv
    else:
        raise ValueError("Longitud de onda fuera del rango de Calzetti.")


def ebv_from_balmer(flux_ha, flux_hb):
    intrinsic_ratio = 2.86

    k_ha = calzetti_k(0.6563)
    k_hb = calzetti_k(0.4861)

    balmer = flux_ha / flux_hb

    ebv_raw = (
        2.5 / (k_ha - k_hb)
        * np.log10(intrinsic_ratio / balmer)
    )

    # No aplicamos extinción negativa.
    ebv = max(ebv_raw, 0.0)

    return ebv, ebv_raw, balmer


def correct_luminosity(L_obs, A_lambda):
    return L_obs * 10**(0.4 * A_lambda)

def radius_from_mask(mask, center_y, center_x):

    mask_clean = mask.copy()

    mask_clean[20:31, 0:10] = False

    yy, xx = np.indices(mask_clean.shape)
    r_pix = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    r_arcsec_map = r_pix * pixscale_arcsec

    R_arcsec = np.nanmax(r_arcsec_map[mask_clean])
    R_kpc = R_arcsec * scale_kpc_arcsec

    return R_arcsec, R_kpc


def log10(x):
    return np.log10(x)

def flux_integrated_error(flux_map, snr_map, mask):

    valid = (
        mask
        & np.isfinite(flux_map)
        & np.isfinite(snr_map)
        & (flux_map > 0)
        & (snr_map > 0)
    )

    sigma_pix = flux_map[valid] / snr_map[valid]

    return np.sqrt(np.nansum(sigma_pix**2))


def ebv_error_from_balmer(flux_ha, flux_hb, err_ha, err_hb):
    '''del decremento de Balmer visto en clase'''

    k_ha = calzetti_k(0.6563)
    k_hb = calzetti_k(0.4861)

    coeff = abs(2.5 / (k_ha - k_hb)) / np.log(10)

    rel_err = np.sqrt(
        (err_ha / flux_ha)**2
        + (err_hb / flux_hb)**2
    )

    return coeff * rel_err


def corrected_luminosity_error(L_corr, flux, flux_err, A_err):


    rel_flux = flux_err / flux
    rel_ext = 0.4 * np.log(10) * A_err

    rel_total = np.sqrt(rel_flux**2 + rel_ext**2)

    return L_corr * rel_total


def log_error(value, value_err):

    return (value_err / value) / np.log(10)


def mean_error_from_map(values):

    values = values[np.isfinite(values)]
    N = len(values)

    if N <= 1:
        return np.nan

    return np.nanstd(values) / np.sqrt(N)


def vout_error(v_values, sigma_values):

    err_vmean = mean_error_from_map(v_values)
    err_sigmean = mean_error_from_map(sigma_values)

    return np.sqrt(err_vmean**2 + (2.0 * err_sigmean)**2)


def mdot_error(Mdot, Mout, Mout_err, vout, vout_err, Rout, Rout_err):

    rel = np.sqrt(
        (Mout_err / Mout)**2
        + (vout_err / vout)**2
        + (Rout_err / Rout)**2
    )

    return Mdot * rel


def eout_error(Eout, Mout, Mout_err, sigma, sigma_err):

    rel = np.sqrt(
        (Mout_err / Mout)**2
        + (2.0 * sigma_err / sigma)**2
    )

    return Eout * rel


def edot_error(Edot, Mdot, Mdot_err, vout, vout_err, sigma, sigma_err):

    term = vout**2 + 3.0 * sigma**2

    term_err = np.sqrt(
        (2.0 * vout * vout_err)**2
        + (6.0 * sigma * sigma_err)**2
    )

    rel = np.sqrt(
        (Mdot_err / Mdot)**2
        + (term_err / term)**2
    )

    return Edot * rel

def luminosity_error(L_obs, flux, flux_err, distance_mpc=None, distance_err_mpc=None):

    rel_flux = flux_err / flux

    if distance_err_mpc is None:
        rel_total = rel_flux
    else:
        rel_dist = 2.0 * distance_err_mpc / distance_mpc
        rel_total = np.sqrt(rel_flux**2 + rel_dist**2)

    return L_obs * rel_total

# Se adopta como centro el máximo de flujo de Halpha estrecha.
center_y, center_x = np.unravel_index(np.nanargmax(flux_ha_n), flux_ha_n.shape)


# SFR CON HALPHA ESTRECHA al principio calculabamos la sfr asi pero da baja y Cristina me ha dado un valor integrado

#
# mask_sfr = (
#     np.isfinite(flux_ha_n)
#     & (flux_ha_n > 0)
# )
#
# F_ha_n = np.nansum(flux_ha_n[mask_sfr])
# L_ha_n_obs = luminosity(F_ha_n)
#
# mask_sfr_ext = (
#     np.isfinite(flux_ha_n)
#     & np.isfinite(flux_hb_n)
#     & (flux_ha_n > 0)
#     & (flux_hb_n > 0)
# )
#
# F_ha_n_ext = np.nansum(flux_ha_n[mask_sfr_ext])
# F_hb_n_ext = np.nansum(flux_hb_n[mask_sfr_ext])
#
# EBV_sfr, EBV_sfr_raw, balmer_sfr = ebv_from_balmer(F_ha_n_ext, F_hb_n_ext)
#
# A_ha_sfr = EBV_sfr * calzetti_k(0.6563)
# L_ha_n_corr = correct_luminosity(L_ha_n_obs, A_ha_sfr)
#
# SFR_HA = SFR_FACTOR * L_ha_n_corr
# SFR = 0.12

#Extincion

mask_ha = (
    np.isfinite(flux_ha_b)
    & np.isfinite(v_ha_b)
    & np.isfinite(sigma_ha_b)
    & np.isfinite(snr_ha_b)
    & (flux_ha_b > 0)
    & (snr_ha_b >= snr_min)
)

mask_ha[20:31, 0:10] = False

mask_out_ext = (
    mask_ha
    & np.isfinite(flux_hb_b)
    & (flux_hb_b > 0)
)

F_ha_b_ext = np.nansum(flux_ha_b[mask_out_ext])
F_hb_b_ext = np.nansum(flux_hb_b[mask_out_ext])

EBV_out, EBV_out_raw, balmer_out = ebv_from_balmer(F_ha_b_ext, F_hb_b_ext)


# 8. FILA HALPHA BROAD


F_ha_b = np.nansum(flux_ha_b[mask_ha])
L_ha_b_obs = luminosity(F_ha_b)

A_ha_out = EBV_out * calzetti_k(0.6563)
L_ha_b_corr = correct_luminosity(L_ha_b_obs, A_ha_out)

vmean_ha = np.nanmean(v_ha_b[mask_ha])
sigma_ha = np.nanmean(sigma_ha_b[mask_ha])

delta_v_ha = abs(vmean_ha)
vout_ha = abs(vmean_ha) + 2.0 * sigma_ha

Mout_ha = 3.2e5 * (L_ha_b_corr / 1e40) * (100.0 / ne)

R_HA_arcsec, R_HA_KPC = radius_from_mask(mask_ha, center_y, center_x)

Mdot_ha = 3.0 * Mout_ha * vout_ha / R_HA_KPC * KM_S_KPC_TO_YR

Eout_ha = 0.5 * Mout_ha * MSUN_TO_G * (sigma_ha * KM_TO_CM)**2

Mdot_ha_g_s = Mdot_ha * MSUN_TO_G / YR_TO_S
Edot_ha = 0.5 * Mdot_ha_g_s * (
    (vout_ha * KM_TO_CM)**2
    + 3.0 * (sigma_ha * KM_TO_CM)**2
)

eta_ha = Mdot_ha / SFR

#Errores


# Error flujo Halpha broad
F_ha_b_err = flux_integrated_error(flux_ha_b, snr_ha_b, mask_ha)

# Error flujo Hbeta broad en la extinción
F_hb_b_err = flux_integrated_error(flux_hb_b, snr_hb_b, mask_out_ext)

L_ha_b_obs_err = luminosity_error(
    L_ha_b_obs,
    F_ha_b,
    F_ha_b_err
)

# Error de E(B-V) del outflow
EBV_out_err = ebv_error_from_balmer(
    F_ha_b_ext,
    F_hb_b_ext,
    F_ha_b_err,
    F_hb_b_err
)

# Error de A_Halpha
A_ha_out_err = calzetti_k(0.6563) * EBV_out_err

# Error de luminosidad corregida
L_ha_b_corr_err = corrected_luminosity_error(
    L_ha_b_corr,
    F_ha_b,
    F_ha_b_err,
    A_ha_out_err
)

# Error de masa
Mout_ha_err = Mout_ha * np.sqrt(
    (L_ha_b_corr_err / L_ha_b_corr)**2
    + (ne_err / ne)**2
)

# Error cinemático
vmean_ha_err = mean_error_from_map(v_ha_b[mask_ha])
sigma_ha_err = mean_error_from_map(sigma_ha_b[mask_ha])

vout_ha_err = np.sqrt(
    vmean_ha_err**2
    + (2.0 * sigma_ha_err)**2
)

# Error radio
R_ha_err = R_ERR_KPC

# Error Mdot
Mdot_ha_err = mdot_error(
    Mdot_ha,
    Mout_ha,
    Mout_ha_err,
    vout_ha,
    vout_ha_err,
    R_HA_KPC,
    R_ha_err
)

# Error eta
eta_ha_err = eta_ha * np.sqrt(
    (Mdot_ha_err / Mdot_ha)**2
    + (SFR_ERR / SFR)**2
)

# Error energía
Eout_ha_err = eout_error(
    Eout_ha,
    Mout_ha,
    Mout_ha_err,
    sigma_ha,
    sigma_ha_err
)

# Error potencia
Edot_ha_err = edot_error(
    Edot_ha,
    Mdot_ha,
    Mdot_ha_err,
    vout_ha,
    vout_ha_err,
    sigma_ha,
    sigma_ha_err
)

# Errores en log
logMout_ha_err = log_error(Mout_ha, Mout_ha_err)
logMdot_ha_err = log_error(Mdot_ha, Mdot_ha_err)
logeta_ha_err = log_error(eta_ha, eta_ha_err)
logEout_ha_err = log_error(Eout_ha, Eout_ha_err)
logEdot_ha_err = log_error(Edot_ha, Edot_ha_err)
logv_ha_err = log_error(vout_ha, vout_ha_err)



# FILA [OIII]


mask_oiii = (
    np.isfinite(flux_oiii_b)
    & np.isfinite(v_oiii_b)
    & np.isfinite(sigma_oiii_b)
    & np.isfinite(snr_oiii_b)
    & (flux_oiii_b > 0)
    & (snr_oiii_b >= snr_min)
)

F_oiii_b = np.nansum(flux_oiii_b[mask_oiii])
L_oiii_b_obs = luminosity(F_oiii_b)

A_oiii_out = EBV_out * calzetti_k(0.5007)
L_oiii_b_corr = correct_luminosity(L_oiii_b_obs, A_oiii_out)

vmean_oiii = np.nanmean(v_oiii_b[mask_oiii])
sigma_oiii = np.nanmean(sigma_oiii_b[mask_oiii])

delta_v_oiii = abs(vmean_oiii)
vout_oiii = abs(vmean_oiii) + 2.0 * sigma_oiii

# Venturi
Mout_oiii = 0.8e8 * (L_oiii_b_corr / 1e44) * (ne / 500.0)**(-1)

R_OIII_arcsec, R_OIII_KPC = radius_from_mask(mask_oiii, center_y, center_x)

Mdot_oiii = 3.0 * Mout_oiii * vout_oiii / R_OIII_KPC * KM_S_KPC_TO_YR

Eout_oiii = 0.5 * Mout_oiii * MSUN_TO_G * (sigma_oiii * KM_TO_CM)**2

Mdot_oiii_g_s = Mdot_oiii * MSUN_TO_G / YR_TO_S
Edot_oiii = 0.5 * Mdot_oiii_g_s * (
    (vout_oiii * KM_TO_CM)**2
    + 3.0 * (sigma_oiii * KM_TO_CM)**2
)

eta_oiii = Mdot_oiii / SFR


# ERRORES [OIII]


# Error flujo [OIII]
F_oiii_b_err = flux_integrated_error(flux_oiii_b, snr_oiii_b, mask_oiii)

L_oiii_b_obs_err = luminosity_error(
    L_oiii_b_obs,
    F_oiii_b,
    F_oiii_b_err
)

# Error de A_[OIII]
A_oiii_out_err = calzetti_k(0.5007) * EBV_out_err

# Error luminosidad corregida
L_oiii_b_corr_err = corrected_luminosity_error(
    L_oiii_b_corr,
    F_oiii_b,
    F_oiii_b_err,
    A_oiii_out_err
)

# Error masa [OIII]
Mout_oiii_err = Mout_oiii * np.sqrt(
    (L_oiii_b_corr_err / L_oiii_b_corr)**2
    + (ne_err / ne)**2
)

# Error cinemático
vmean_oiii_err = mean_error_from_map(v_oiii_b[mask_oiii])
sigma_oiii_err = mean_error_from_map(sigma_oiii_b[mask_oiii])

vout_oiii_err = np.sqrt(
    vmean_oiii_err**2
    + (2.0 * sigma_oiii_err)**2
)

# Error radio
R_oiii_err = R_ERR_KPC

# Error Mdot
Mdot_oiii_err = mdot_error(
    Mdot_oiii,
    Mout_oiii,
    Mout_oiii_err,
    vout_oiii,
    vout_oiii_err,
    R_OIII_KPC,
    R_oiii_err
)

# Error eta
eta_oiii_err = eta_oiii * np.sqrt(
    (Mdot_oiii_err / Mdot_oiii)**2
    + (SFR_ERR / SFR)**2
)

# Error energía
Eout_oiii_err = eout_error(
    Eout_oiii,
    Mout_oiii,
    Mout_oiii_err,
    sigma_oiii,
    sigma_oiii_err
)

# Error potencia
Edot_oiii_err = edot_error(
    Edot_oiii,
    Mdot_oiii,
    Mdot_oiii_err,
    vout_oiii,
    vout_oiii_err,
    sigma_oiii,
    sigma_oiii_err
)

# Errores en log
logMout_oiii_err = log_error(Mout_oiii, Mout_oiii_err)
logMdot_oiii_err = log_error(Mdot_oiii, Mdot_oiii_err)
logeta_oiii_err = log_error(eta_oiii, eta_oiii_err)
logEout_oiii_err = log_error(Eout_oiii, Eout_oiii_err)
logEdot_oiii_err = log_error(Edot_oiii, Edot_oiii_err)
logv_oiii_err = log_error(vout_oiii, vout_oiii_err)


# IMPRIMIR TABLA FINAL


print("\n")
print("Galaxy | Lobs,out | Aout | Rout | Dv | vout | sigma | logMout | logMdot | logeta | logEout | logEdot | ne")
print("-" * 190)

print(
    f"NGC 4750 Halpha | "
    f"{L_ha_b_obs/1e38:.2f}±{L_ha_b_obs_err/1e38:.2f} |"
    f"{A_ha_out:.2f}±{A_ha_out_err:.2f} | "
    f"{R_HA_KPC:.3f}±{R_ha_err:.3f} | "
    f"{delta_v_ha:.1f}±{vmean_ha_err:.1f} | "
    f"{vout_ha:.1f}±{vout_ha_err:.1f} | "
    f"{sigma_ha:.1f}±{sigma_ha_err:.1f} | "
    f"{np.log10(Mout_ha):.2f}±{logMout_ha_err:.2f} | "
    f"{np.log10(Mdot_ha):.2f}±{logMdot_ha_err:.2f} | "
    f"{np.log10(eta_ha):.2f}±{logeta_ha_err:.2f} | "
    f"{np.log10(Eout_ha):.2f}±{logEout_ha_err:.2f} | "
    f"{np.log10(Edot_ha):.2f}±{logEdot_ha_err:.2f} | "
    f"{ne:.0f}±{ne_err:.0f}"
)

print(
    f"NGC 4750 [OIII] | "
    f"{L_oiii_b_obs/1e38:.2f}±{L_oiii_b_obs_err/1e38:.2f} |"
    f"{A_oiii_out:.2f}±{A_oiii_out_err:.2f} | "
    f"{R_OIII_KPC:.3f}±{R_oiii_err:.3f} | "
    f"{delta_v_oiii:.1f}±{vmean_oiii_err:.1f} | "
    f"{vout_oiii:.1f}±{vout_oiii_err:.1f} | "
    f"{sigma_oiii:.1f}±{sigma_oiii_err:.1f} | "
    f"{np.log10(Mout_oiii):.2f}±{logMout_oiii_err:.2f} | "
    f"{np.log10(Mdot_oiii):.2f}±{logMdot_oiii_err:.2f} | "
    f"{np.log10(eta_oiii):.2f}±{logeta_oiii_err:.2f} | "
    f"{np.log10(Eout_oiii):.2f}±{logEout_oiii_err:.2f} | "
    f"{np.log10(Edot_oiii):.2f}±{logEdot_oiii_err:.2f} | "
    f"{ne:.0f}±{ne_err:.0f}"
)


# IMPRIMIR TABLA FINAL SIN LOGARITMOS


print("\n")
print("TABLA 1. PROPIEDADES OBSERVACIONALES Y CINEMÁTICAS")
print("Galaxy | Lobs,out(1e38 erg/s) | Aout | Rout(kpc) | Dv(km/s) | vout(km/s) | sigma(km/s) | ne")
print("-" * 130)

print(
    f"NGC 4750 Halpha | "
    f"{L_ha_b_obs/1e38:.2f}±{L_ha_b_obs_err/1e38:.2f} | "
    f"{A_ha_out:.2f}±{A_ha_out_err:.2f} | "
    f"{R_HA_KPC:.3f}±{R_ha_err:.3f} | "
    f"{delta_v_ha:.1f}±{vmean_ha_err:.1f} | "
    f"{vout_ha:.1f}±{vout_ha_err:.1f} | "
    f"{sigma_ha:.1f}±{sigma_ha_err:.1f} | "
    f"{ne:.0f}±{ne_err:.0f}"
)

print(
    f"NGC 4750 [OIII] | "
    f"{L_oiii_b_obs/1e38:.2f}±{L_oiii_b_obs_err/1e38:.2f} | "
    f"{A_oiii_out:.2f}±{A_oiii_out_err:.2f} | "
    f"{R_OIII_KPC:.3f}±{R_oiii_err:.3f} | "
    f"{delta_v_oiii:.1f}±{vmean_oiii_err:.1f} | "
    f"{vout_oiii:.1f}±{vout_oiii_err:.1f} | "
    f"{sigma_oiii:.1f}±{sigma_oiii_err:.1f} | "
    f"{ne:.0f}±{ne_err:.0f}"
)


print("\n")
print("TABLA 2. PROPIEDADES FÍSICAS DERIVADAS SIN LOGARITMOS")
print("Galaxy | Mout(Msun) | Mdot(Msun/yr) | eta | Eout(erg) | Edot(erg/s)")
print("-" * 110)

print(
    f"NGC 4750 Halpha | "
    f"({Mout_ha/10**np.floor(np.log10(Mout_ha)):.2f}±{Mout_ha_err/10**np.floor(np.log10(Mout_ha)):.2f})e{int(np.floor(np.log10(Mout_ha)))} | "
    f"({Mdot_ha/10**np.floor(np.log10(Mdot_ha)):.2f}±{Mdot_ha_err/10**np.floor(np.log10(Mdot_ha)):.2f})e{int(np.floor(np.log10(Mdot_ha)))} | "
    f"{eta_ha:.2f}±{eta_ha_err:.2f} | "
    f"({Eout_ha/10**np.floor(np.log10(Eout_ha)):.2f}±{Eout_ha_err/10**np.floor(np.log10(Eout_ha)):.2f})e{int(np.floor(np.log10(Eout_ha)))} | "
    f"({Edot_ha/10**np.floor(np.log10(Edot_ha)):.2f}±{Edot_ha_err/10**np.floor(np.log10(Edot_ha)):.2f})e{int(np.floor(np.log10(Edot_ha)))}"
)

print(
    f"NGC 4750 [OIII] | "
    f"({Mout_oiii/10**np.floor(np.log10(Mout_oiii)):.2f}±{Mout_oiii_err/10**np.floor(np.log10(Mout_oiii)):.2f})e{int(np.floor(np.log10(Mout_oiii)))} | "
    f"({Mdot_oiii/10**np.floor(np.log10(Mdot_oiii)):.2f}±{Mdot_oiii_err/10**np.floor(np.log10(Mdot_oiii)):.2f})e{int(np.floor(np.log10(Mdot_oiii)))} | "
    f"{eta_oiii:.3f}±{eta_oiii_err:.3f} | "
    f"({Eout_oiii/10**np.floor(np.log10(Eout_oiii)):.2f}±{Eout_oiii_err/10**np.floor(np.log10(Eout_oiii)):.2f})e{int(np.floor(np.log10(Eout_oiii)))} | "
    f"({Edot_oiii/10**np.floor(np.log10(Edot_oiii)):.2f}±{Edot_oiii_err/10**np.floor(np.log10(Edot_oiii)):.2f})e{int(np.floor(np.log10(Edot_oiii)))}"
)


print("\n")
print("VALORES EN LOG SOLO PARA LAS GRÁFICAS")
print("-" * 60)
print(f"logSFR = {np.log10(SFR):.3f} ± {log_error(SFR, SFR_ERR):.3f}")
print(f"Halpha: logMdot = {np.log10(Mdot_ha):.3f} ± {logMdot_ha_err:.3f}")
print(f"Halpha: logvout = {np.log10(vout_ha):.3f} ± {logv_ha_err:.3f}")
print(f"[OIII]: logMdot = {np.log10(Mdot_oiii):.3f} ± {logMdot_oiii_err:.3f}")
print(f"[OIII]: logvout = {np.log10(vout_oiii):.3f} ± {logv_oiii_err:.3f}")

print("\nFIN DEL SCRIPT\n")
