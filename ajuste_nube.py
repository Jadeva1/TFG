from scipy.optimize import curve_fit
from functools import partial
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np

# LECTURA DEL FITS

hdul   = fits.open("/home/rauladeva/residuals_rss_NGC4750_LR_R.fits")
data   = hdul[0].data
header = hdul[0].header
hdul.close()

nl    = header['NAXIS3']   # número de canales espectrales
crval = header['CRVAL3']   # longitud de onda del píxel de referencia [Å]
crpix = header['CRPIX3']   # índice del píxel de referencia
cdelt = header['CDELT3']   # paso espectral [Å/canal]
z_sys = 0.005404
scale_kpc_per_arcsec = 24.71 * 1000 / 206265

print(scale_kpc_per_arcsec)


wave = crval + (np.arange(nl) + 1 - crpix) * cdelt

#comprobaciones para saber si se ha extraido bien
print(f"Rango espectral: {wave[0]:.2f} – {wave[-1]:.2f} Å")
print(f"Paso espectral:  {cdelt:.4f} Å/canal")
print(f"Dimensiones:     {data.shape}  →  (λ, y, x)")


# CONSTANTES FÍSICAS

C_LIGHT = 2.998e5  # km/s
R = 6100 # res de MEGARA
sigma_ins = C_LIGHT / (2.355 * R)
lam_cont_width  = 30.0
lam_line_n = 60.0
lam_line_b  = 100.0
SIGMA_N_DETECT = 60.0
SIGMA_N_MAX    = 200.0

#Lineas
HA_REST = 6562.80  # Å
NII_6584 = 6583.45  # Å
NII_6548 = 6548.05  # Å
NII_RATIO = 3.0  # Intensidad del pico mayor es el triple que el del menor

# AJUSTE GAUSSIANO INDIVIDUAL

def gauss_vel(wave, A, v_kms, sigma_kms, lam_rest, z_sys):

    sigma_kms = max(sigma_kms, 1e-3)  # evita cero
    lam0 = lam_rest * (1.0 + z_sys) * (1.0 + v_kms / C_LIGHT)
    sig_lam = lam0 * sigma_kms / C_LIGHT

    return A * np.exp(-0.5 * ((wave - lam0) / sig_lam) ** 2)

def compute_epsilon_c(wave_full, flux_full, continuum, scale, z_sys):

    lam_cont_center = 6680 * (1 + z_sys)
    mask = (wave_full > lam_cont_center - lam_cont_width / 2) & \
           (wave_full < lam_cont_center + lam_cont_width / 2)
    if mask.sum() < 5:
        # fallback: usar la ventana de continuo azul
        lam_cont_center = 6510 * (1 + z_sys)
        mask = (wave_full > lam_cont_center - 15) & \
               (wave_full < lam_cont_center + 15)
    if mask.sum() < 5:
        return 1.0   # fallback: en espacio normalizado σ≈1
    f_norm_cont = (flux_full[mask] - continuum) / scale
    return np.std(f_norm_cont)

def compute_epsilon_line(wave, residuals_norm, z_sys, include_broad=False):

    wave_obs = wave * (1 + z_sys)
    lam_ha = HA_REST * (1 + z_sys)
    half   = (lam_line_b if include_broad else lam_line_n) / 2
    mask   = (wave_obs > lam_ha - half) & (wave_obs < lam_ha + half)
    if mask.sum() < 5:
        return 0.0
    return np.std(residuals_norm[mask])

# MODELO COMPLETO: continuo + estrecho + ancho

def ha_nii_model(wave, C,
                 A_ha_n, v_n, sigma_n, A_nii_n,
                 A_ha_b, v_b, sigma_b, A_nii_b, z_sys):

    m = np.full_like(wave, C, dtype=float)

    # Componente estrecha
    m += gauss_vel(wave, A_ha_n, v_n, sigma_n, HA_REST, z_sys)
    m += gauss_vel(wave, A_nii_n, v_n, sigma_n, NII_6584, z_sys)
    m += gauss_vel(wave, A_nii_n / NII_RATIO, v_n, sigma_n, NII_6548, z_sys)

    # Componente ancha
    m += gauss_vel(wave, A_ha_b, v_b, sigma_b, HA_REST, z_sys)
    m += gauss_vel(wave, A_nii_b, v_b, sigma_b, NII_6584, z_sys)
    m += gauss_vel(wave, A_nii_b / NII_RATIO, v_b, sigma_b, NII_6548, z_sys)

    return m


# MODELO SOLO ESTRECHA (si tenemos pocos residuos ajustando con la estrecha no hace falta la ancha) lo que buscamos es que el SNR sea < 3

def ha_nii_narrow_model(wave,
                       C,
                       A_ha_n, v_n, sigma_n, A_nii_n,
                       z_sys):

    m = np.full_like(wave, C, dtype=float)

    m += gauss_vel(wave, A_ha_n, v_n, sigma_n, HA_REST, z_sys)
    m += gauss_vel(wave, A_nii_n, v_n, sigma_n, NII_6584, z_sys)
    m += gauss_vel(wave, A_nii_n / NII_RATIO, v_n, sigma_n, NII_6548, z_sys)

    return m


def make_narrow_model(z_sys):
    def model(wave, C, A_ha_n, v_n, sigma_n, A_nii_n):
        return ha_nii_narrow_model(
            wave, C, A_ha_n, v_n, sigma_n, A_nii_n, z_sys
        )
    return model

# Probamos con unos datos iniciales medio razonables

def estimate_initial_params(wave, flux, z_sys, continuum):
    lam_ha_sys  = HA_REST  * (1.0 + z_sys)
    lam_nii_sys = NII_6584 * (1.0 + z_sys)

    # Ventana estrecha solo alrededor de Hα (±15 Å)
    window_ha = 10
    mask_ha   = (wave > lam_ha_sys - window_ha) & (wave < lam_ha_sys + window_ha)

    if mask_ha.sum() > 3 and np.max(flux[mask_ha]) > 0:
        lam_peak_ha = wave[mask_ha][np.argmax(flux[mask_ha])]
    else:
        lam_peak_ha = lam_ha_sys

    v_init = (lam_peak_ha / lam_ha_sys - 1.0) * C_LIGHT

    mask_nii  = (wave > lam_nii_sys - 10) & (wave < lam_nii_sys + 10)

    A_ha_init  = max(np.max(flux[mask_ha])  - continuum, 1e-3) if mask_ha.sum()  > 0 else 1e-3
    A_nii_init = max(np.max(flux[mask_nii]) - continuum, 1e-3) if mask_nii.sum() > 0 else 1e-3

    sigma_n = sigma_ins * 2

    # p0: [C, A_ha_n, v_n, sigma_n, A_nii_n, A_ha_b, v_b, sigma_b, A_nii_b]
    p0 = [continuum,A_ha_init,v_init, sigma_n, A_nii_init,A_ha_init * 0.2,  v_init, 400.0, A_nii_init * 0.2,]

    #  Bounds para que de resultados físicos con sentido y eliminar datos feura de lo esperado
    eps = 1e-20
    lo = [
        continuum - abs(continuum) * 0.5 - eps,
        0,  -300,  sigma_ins,   0,
        0,  -500, 130,   0,
    ]
    hi = [
        continuum + abs(continuum) * 0.5 + eps,
        A_ha_init  * 15, 300, SIGMA_N_MAX,  A_nii_init * 15,
        A_ha_init  * 10,  500, 1500,  A_nii_init * 5,
    ]

    return p0, (lo, hi)

# AJUSTE ESPAXEL A ESPAXEL

def fit_spaxel(wave, flux, z_sys, continuum, scale, wave_full, flux_full):

    f_norm = (flux - continuum) / scale
    epsilon_c = compute_epsilon_c(wave_full, flux_full, continuum, scale, z_sys)
    narrow_model = make_narrow_model(z_sys)

    p0_n, bounds_n = estimate_initial_params(wave, f_norm, z_sys, continuum=0.0)
    # solo primeros parámetros (estrecha)
    p0_n     = p0_n[:5]
    bounds_n = (list(bounds_n[0][:5]), list(bounds_n[1][:5]))
    bounds_n[1][3] = SIGMA_N_DETECT   # sigma_n máx = 60 km/s solo en este paso

    try:
        popt_n_detect, _ = curve_fit(
            narrow_model, wave, f_norm,
            p0=p0_n, bounds=bounds_n,
            maxfev=10000, method='trf'
        )
    except:
        return {'success': False}

    residuals_detect = f_norm - narrow_model(wave, *popt_n_detect)
    epsilon_line = compute_epsilon_line(wave, residuals_detect, z_sys, include_broad=False)
    use_broad    = epsilon_line > 3.0 * epsilon_c


    # AJUSTE COMPLETO SI NECESARIO

    if use_broad:
        p0, bounds = estimate_initial_params(wave, f_norm, z_sys, continuum=0.0)
        # sigma_n ahora libre hasta SIGMA_N_MAX
        bounds[1][3] = SIGMA_N_MAX
        full_model = partial(ha_nii_model, z_sys=z_sys)
        try:
            popt, pcov = curve_fit(
                full_model, wave, f_norm,
                p0=p0, bounds=bounds,
                maxfev=10000, method="trf"
            )
        except:
            return {'success': False}

        # Verificar que la componente ancha mejora los residuos
        residuals_full = f_norm - full_model(wave, *popt)
        epsilon_line_b = compute_epsilon_line(
            wave, residuals_full, z_sys, include_broad=True
        )

        # Si con la ancha los residuos no mejoran, descartarla
        if epsilon_line_b >= epsilon_line:
            use_broad = False


    if not use_broad:
        p0_n2, bounds_n2 = estimate_initial_params(wave, f_norm, z_sys, continuum=0.0)
        p0_n2 = p0_n2[:5]
        bounds_n2 = (list(bounds_n2[0][:5]), list(bounds_n2[1][:5]))
        bounds_n2[1][3] = SIGMA_N_MAX  # ahora sigma_n libre hasta 120 km/s
        try:
            popt_n_final, _ = curve_fit(
                narrow_model, wave, f_norm,
                p0=p0_n2, bounds=bounds_n2,
                maxfev=10000, method='trf')
        except (RuntimeError, ValueError):
            popt_n_final = popt_n_detect

        popt = np.zeros(9)
        popt[:5] = popt_n_final
        popt[7] = 300.0

    full_model = partial(ha_nii_model, z_sys=z_sys)
    model_norm = full_model(wave, *popt)
    model_phys = model_norm * scale + continuum

    C, A_ha_n, v_n, sigma_n, A_nii_n, \
       A_ha_b, v_b, sigma_b, A_nii_b = popt

    sigma_n_int = np.sqrt(max(sigma_n**2 - sigma_ins**2, 0.0))
    lam_ha_n = HA_REST * (1 + z_sys) * (1 + v_n / C_LIGHT)
    sig_n_lam = lam_ha_n * sigma_n / C_LIGHT
    flux_ha_n = A_ha_n * sig_n_lam * np.sqrt(2*np.pi) * scale
    lam_nii_n = NII_6584 * (1 + z_sys) * (1 + v_n / C_LIGHT)
    sig_nii_lam = lam_nii_n * sigma_n / C_LIGHT
    flux_nii_n = A_nii_n * sig_nii_lam * np.sqrt(2 * np.pi) * scale

    lam_ha_b = HA_REST * (1 + z_sys) * (1 + v_b / C_LIGHT)
    sig_b_lam = lam_ha_b * sigma_b / C_LIGHT
    flux_ha_b = A_ha_b * sig_b_lam * np.sqrt(2*np.pi) * scale
    lam_nii_b = NII_6584 * (1 + z_sys) * (1 + v_b / C_LIGHT)
    sig_nii_b_lam = lam_nii_b * sigma_b / C_LIGHT
    flux_nii_b = A_nii_b * sig_nii_b_lam * np.sqrt(2 * np.pi) * scale

    snr_n = A_ha_n / epsilon_c
    snr_b = A_ha_b / epsilon_c if use_broad else 0.0
    snr_nii_n = A_nii_n / epsilon_c
    snr_nii_b = A_nii_b / epsilon_c if use_broad else 0.0

    flux_total = flux_ha_n + flux_ha_b
    ratio_broad = flux_ha_b / flux_total if flux_total > 0 else 0.0

    return {
        'success': True,
        'popt': popt,
        'use_broad': use_broad,
        'epsilon_c': epsilon_c,
        'epsilon_line': epsilon_line,
        'v_n': v_n,
        'sigma_n': sigma_n,
        'sigma_n_int': sigma_n_int,
        'v_b': v_b,
        'sigma_b': sigma_b,
        'flux_ha_n': flux_ha_n,
        'flux_ha_b': flux_ha_b,
        'snr_n': snr_n,
        'snr_b': snr_b,
        'ratio_broad': ratio_broad,
        'model_phys': model_phys,
        'flux_nii_n': flux_nii_n,
        'flux_nii_b': flux_nii_b,
        'snr_nii_n':  snr_nii_n,
        'snr_nii_b':  snr_nii_b,
    }

def plot_spaxel_fit_with_residuals(wave, flux, result, z_sys,
                                   continuum, scale, i, j,
                                   wave_full, flux_full):

    '''Esta funcion sirve para el duiagnostico de si se aplican bien las componentes anchas y esstrechas correspondientes'''

    popt  = result['popt']
    model = result['model_phys']

    C, A_ha_n, v_n, sigma_n, A_nii_n, \
       A_ha_b, v_b, sigma_b, A_nii_b = popt

    narrow = (
        gauss_vel(wave, A_ha_n,            v_n, sigma_n, HA_REST,  z_sys) +
        gauss_vel(wave, A_nii_n,           v_n, sigma_n, NII_6584, z_sys) +
        gauss_vel(wave, A_nii_n/NII_RATIO, v_n, sigma_n, NII_6548, z_sys)
    )

    broad = (
        gauss_vel(wave, A_ha_b,            v_b, sigma_b, HA_REST,  z_sys) +
        gauss_vel(wave, A_nii_b,           v_b, sigma_b, NII_6584, z_sys) +
        gauss_vel(wave, A_nii_b/NII_RATIO, v_b, sigma_b, NII_6548, z_sys)
    )
    #Cambio de unidades

    narrow_phys = narrow * scale + continuum
    broad_phys  = broad  * scale + continuum

    residuals = flux - model

    lam_ha     = HA_REST * (1 + z_sys)
    lam_nii6548 = NII_6548 * (1 + z_sys)
    lam_nii6584 = NII_6584 * (1 + z_sys)

    lam_cont_c = 6680 * (1 + z_sys)
    half_line  = (50 if result['use_broad'] else 30)

    # Para que quede bien en el TFG

    fontsize_axis = 15
    fontsize_ticks = 13
    fontsize_legend = 12
    fontsize_title = 16
    fontsize_lines = 11


    fig, (ax_full, ax_fit, ax_res) = plt.subplots(
        3, 1,
        figsize=(12, 10),
        gridspec_kw={'height_ratios': [1.4, 3.0, 1.1]},
        sharex=False
    )


    ax_full.plot(
        wave_full,
        flux_full,
        color='black',
        lw=0.9
    )

    ax_full.axvspan(
        lam_cont_c - 15,
        lam_cont_c + 15,
        alpha=0.30,
        color='orange',
        label=fr'$\epsilon_c={result["epsilon_c"]:.3f}$'
    )

    ax_full.axvspan(
        wave[0],
        wave[-1],
        alpha=0.12,
        color='blue',
        label='ventana ajuste'
    )

    ax_full.set_xlim(np.nanmin(wave_full), np.nanmax(wave_full))

    mask_y_scale = (
        np.isfinite(wave_full)
        & np.isfinite(flux_full)
        & (wave_full >= wave[0])
        & (wave_full <= wave[-1])
    )

    flux_y_scale = flux_full[mask_y_scale]

    if len(flux_y_scale) < 5:
        flux_y_scale = flux[np.isfinite(flux)]

    y_low, y_high = np.nanpercentile(flux_y_scale, [0.2, 99.5])
    yrange = y_high - y_low

    if (not np.isfinite(yrange)) or (yrange <= 0):
        y_low = np.nanmin(flux_y_scale)
        y_high = np.nanmax(flux_y_scale)
        yrange = y_high - y_low

    if (not np.isfinite(yrange)) or (yrange <= 0):
        y_low, y_high = -1, 1
        yrange = 2

    ax_full.set_ylim(
        y_low - 0.25 * yrange,
        y_high + 0.20 * yrange
    )

    ax_full.set_ylabel('Flujo', fontsize=fontsize_axis)
    ax_full.set_title(f'Spaxel ({i},{j})', fontsize=fontsize_title)
    ax_full.legend(fontsize=fontsize_legend, loc='upper right')
    ax_full.tick_params(axis='both', labelsize=fontsize_ticks)


    # Panel 2: zoom en la zona de ajuste


    ax_fit.plot(wave, flux,        'k',   lw=1.2, label='datos')
    ax_fit.plot(wave, model,       'r',   lw=2.2, label='modelo total')
    ax_fit.plot(wave, narrow_phys, 'b--', lw=1.8,
                label=fr'estrecha $\sigma={sigma_n:.0f}$ km s$^{{-1}}$')

    if result["use_broad"]:
        ax_fit.plot(wave, broad_phys, 'g--', lw=1.8,
                    label=fr'ancha $\sigma={sigma_b:.0f}$ km s$^{{-1}}$')

    ax_fit.axvspan(
        lam_ha - half_line,
        lam_ha + half_line,
        alpha=0.10,
        color='blue',
        label=fr'$\epsilon_{{line}}={result["epsilon_line"]:.3f}$'
    )

    # Marcar posiciones de Halpha y [NII]
    for lam_mark, lbl in [
        (lam_nii6548, r'[N II] 6548'),
        (lam_ha, r'H$\alpha$'),
        (lam_nii6584, r'[N II] 6584')
    ]:
        ax_fit.axvline(
            lam_mark,
            color='gray',
            lw=1.0,
            ls=':',
            alpha=0.8
        )

        ylim = ax_fit.get_ylim()
        y_text = ylim[1] - 0.06 * (ylim[1] - ylim[0])

        ax_fit.text(
            lam_mark,
            y_text,
            lbl,
            fontsize=fontsize_lines,
            color='gray',
            ha='center',
            va='top',
            rotation=90
        )

    ax_fit.set_xlim(wave[0], wave[-1])
    ax_fit.set_ylabel('Flujo', fontsize=fontsize_axis)
    ax_fit.legend(fontsize=fontsize_legend, loc='upper right')
    ax_fit.tick_params(axis='both', labelsize=fontsize_ticks)

    # Panel 3: residuos


    ax_res.plot(wave, residuals, 'k', lw=1.2)
    ax_res.axhline(0, color='r', ls='--', lw=1.2)

    ax_res.axvspan(
        lam_ha - half_line,
        lam_ha + half_line,
        alpha=0.10,
        color='blue'
    )

    ax_res.set_xlim(wave[0], wave[-1])
    ax_res.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=fontsize_axis)
    ax_res.set_ylabel('Residuos', fontsize=fontsize_axis)
    ax_res.tick_params(axis='both', labelsize=fontsize_ticks)

    # Guardar


    plt.tight_layout()
    plt.savefig(f"fit_residuals_{i}_{j}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_map(data_map, title, header=None, cmap='RdBu_r',
             vmin=None, vmax=None, cbar_label=None, filename=None):
    """
    Genera mapas 2D.
    """

    fig = plt.figure(figsize=(6, 5))
    ax = None

    if header is not None:
        try:
            h2 = header.copy()

            for key in [
                'SPECSYS', 'SSYSOBS', 'SSYSSRC',
                'RESTFRQ', 'RESTWAV', 'VELOSYS', 'ZSOURCE',
                'CTYPE3', 'CUNIT3', 'CRVAL3', 'CRPIX3', 'CDELT3',
                'CNAME3', 'CRDER3', 'CSYER3', 'NAXIS3'
            ]:
                if key in h2:
                    del h2[key]

            wcs_2d = WCS(h2, naxis=2)

            ax = fig.add_subplot(111, projection=wcs_2d)
            im = ax.imshow(data_map, origin='lower', cmap=cmap,
                           vmin=vmin, vmax=vmax)

            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')

        except Exception as e:
            print(f"[WCS] No se pudo aplicar proyección WCS: {e}. Usando ejes píxel.")

    if ax is None:
        ax = fig.add_subplot(111)
        im = ax.imshow(data_map, origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel('x [píxel]')
        ax.set_ylabel('y [píxel]')

    if cbar_label is None:
        cbar_label = title

    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)

    plt.tight_layout()

    if filename is None:
        filename = (
            title.replace(' ', '_')
                 .replace('/', '-')
                 .replace('[', '')
                 .replace(']', '')
                 .replace('α', 'alpha')
                 .replace('β', 'beta')
            + '.png'
        )

    plt.savefig(filename, dpi=150)
    plt.close()



#!!!!!!!!!!!!!!!!!!!!!!!!!!!
# SCRIPT PARA LANZAR EL CUBO
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Regiones de continuo (sin líneas de emisión)

lam_cont_lo_blue = 6480 * (1 + z_sys)   # ~6515 Å — azul, antes de [NII]6548
lam_cont_hi_blue = 6530 * (1 + z_sys)   # ~6565 Å
lam_cont_lo_red  = 6650 * (1 + z_sys)   # ~6686 Å — rojo, tras [NII]6584
lam_cont_hi_red  = 6700 * (1 + z_sys)   # ~6736 Å

mask_cont = (
    ((wave > lam_cont_lo_blue) & (wave < lam_cont_hi_blue)) |
    ((wave > lam_cont_lo_red)  & (wave < lam_cont_hi_red))
)

# Ventana de ajuste: cubre el triplete con margen
lam_fit_lo = NII_6548 * (1 + z_sys) - 30   # 30 Å antes de [NII]6548
lam_fit_hi = NII_6584 * (1 + z_sys) + 30   # 30 Å detras de [NII]6584
mask_fit   = (wave > lam_fit_lo) & (wave < lam_fit_hi)
wave_fit   = wave[mask_fit]

print(f"Ventana de ajuste: {lam_fit_lo:.1f} – {lam_fit_hi:.1f} Å  ({mask_fit.sum()} canales)")
print(f"Ventana continuo:  {mask_cont.sum()} canales")
i_c, j_c = data.shape[1] // 2, data.shape[2] // 2
spec = data[:, i_c, j_c]

# Test en espaxel central ANTES de lanzar el cubo estos datos nos dan una idea de como va la cosa
flux_spaxel = data[:, i_c, j_c]
continuum   = np.nanmedian(flux_spaxel[mask_cont])
scale       = np.nanstd(flux_spaxel[mask_cont])
flux_fit    = flux_spaxel[mask_fit]

print(f"continuum = {continuum:.4e}")
print(f"scale     = {scale:.4e}")
print(f"SNR pico  = {(np.max(flux_fit)-continuum)/scale:.1f}")

result = fit_spaxel(wave_fit, flux_fit, z_sys, continuum, scale, wave, flux_spaxel)
print(f"v_n spaxel central   = {result['v_n']:.1f} km/s  (esperado: |v| < 300)")
print(f"sigma_n spaxel central= {result['sigma_n']:.1f} km/s")
print(f"v_b   spaxel central = {result['v_b']:.1f} km/s")
print(f"sigma_b spaxel central= {result['sigma_b']:.1f} km/s")

# Mapas de salida
nl, ny, nx = data.shape
v_n_map = np.full((ny, nx), np.nan)
sigma_n_map = np.full((ny, nx), np.nan)
v_b_map = np.full((ny, nx), np.nan)
sigma_b_map = np.full((ny, nx), np.nan)
flux_n_map = np.full((ny, nx), np.nan)
flux_b_map = np.full((ny, nx), np.nan)
snr_n_map = np.full((ny, nx), np.nan)
snr_b_map = np.full((ny, nx), np.nan)
ratio_map = np.full((ny, nx), np.nan)
flux_nii_n_map  = np.full((ny, nx), np.nan)
flux_nii_b_map  = np.full((ny, nx), np.nan)
snr_nii_n_map   = np.full((ny, nx), np.nan)
snr_nii_b_map   = np.full((ny, nx), np.nan)

for i in range(ny):
    for j in range(nx):
        flux_spaxel = data[:, i, j]

        # Continuo local: mediana en ventana libre de líneas
        if mask_cont.sum() > 3:
            continuum = np.nanmedian(flux_spaxel[mask_cont])
        else:
            continuum = np.nanmedian(flux_spaxel)

        # Escala: desviación estándar del continuo
        scale = np.nanstd(flux_spaxel[mask_cont]) if mask_cont.sum() > 3 else 1e-17
        if scale < 1e-30:
            continue

        flux_fit = flux_spaxel[mask_fit]

        # Comprobar que hay señal
        lam_ha_obs   = HA_REST * (1 + z_sys)
        mask_ha_peak = (wave_fit > lam_ha_obs - 5) & (wave_fit < lam_ha_obs + 5)
        if mask_ha_peak.sum() == 0:
            continue
        snr_ha_obs = (np.nanmax(flux_fit[mask_ha_peak]) - continuum) / scale
        if snr_ha_obs < 3:
            continue

        result = fit_spaxel(wave_fit, flux_fit, z_sys, continuum, scale, wave, flux_spaxel)

        # ── Corte posterior: S/N del ajuste ──────────────────
        if not result['success']:
            continue
        if result['snr_n'] < 5:
            continue

        if result['success']:
            v_n_map[i, j] = result['v_n']
            sigma_n_map[i, j] = result['sigma_n_int']
            flux_n_map[i, j] = result['flux_ha_n']
            snr_n_map[i, j] = result['snr_n']
            flux_nii_n_map[i, j] = result['flux_nii_n']
            snr_nii_n_map[i, j] = result['snr_nii_n']

            if result['use_broad']:
                v_b_map[i, j] = result['v_b']
                sigma_b_map[i, j] = result['sigma_b']
                flux_b_map[i, j] = result['flux_ha_b']
                snr_b_map[i, j] = result['snr_b']
                ratio_map[i, j] = result['ratio_broad']
                flux_nii_b_map[i, j] = result['flux_nii_b']
                snr_nii_b_map[i, j] = result['snr_nii_b']
            #graficamos algunos spaxeles para ver el ajuste
            if (i % 10 == 0) and (j % 10 == 0):
                plot_spaxel_fit_with_residuals(
                    wave_fit, flux_fit, result, z_sys,
                    continuum, scale, i, j,
                    wave, flux_spaxel  # espectro completo para mostrar ventana εc
                )
            if (i == 20) and (j == 25):
                plot_spaxel_fit_with_residuals(
                    wave_fit,
                    flux_fit,
                    result,
                    z_sys,
                    continuum,
                    scale,
                    i, j, wave, flux_spaxel)
mask_broad = ratio_map > 0.05

#estas son máscaras de resultados que quedan mal

mask_positive_velocity_broad = (
    np.isfinite(v_b_map)
    & (v_b_map > 0)
)

flux_b_map[mask_positive_velocity_broad] = np.nan
v_b_map[mask_positive_velocity_broad] = np.nan
sigma_b_map[mask_positive_velocity_broad] = np.nan
snr_b_map[mask_positive_velocity_broad] = np.nan

print("Spaxels con velocidad positiva en Halpha broad enmascarados:",
      np.sum(mask_positive_velocity_broad))

mask_region_aislada = np.zeros_like(flux_b_map, dtype=bool)


mask_region_aislada[20:31, 0:10] = True

flux_b_map[mask_region_aislada] = np.nan
v_b_map[mask_region_aislada] = np.nan
sigma_b_map[mask_region_aislada] = np.nan
snr_b_map[mask_region_aislada] = np.nan

print("Spaxels enmascarados en Hα broad por región aislada:",
      np.sum(mask_region_aislada))


plot_map(v_n_map, 'Velocidad Hα componente estrecha [km/s]', header=header, vmin=-200, vmax=200)
plot_map(sigma_n_map, 'Dispersión Hα componente estrecha [km/s]', header=header, cmap='viridis')
plot_map(flux_n_map, 'Flujo Hα componente estrecha', header=header, cmap='inferno')
plot_map(flux_b_map, 'Flujo Hα componente ancha', header=header, cmap='inferno')
plot_map(v_b_map, 'Velocidad Hα componente ancha [km/s]', header=header, vmin=-300, vmax=300)
plot_map(sigma_b_map, 'Dispersión Hα componente ancha [km/s]', header=header, cmap='viridis')
plot_map(snr_n_map, 'SNR Hα estrecha', header=header, cmap='cividis', vmin=3, vmax=50)
plot_map(snr_b_map, 'SNR Hα ancha', header=header, cmap='cividis', vmin=5, vmax=30)
plot_map(snr_nii_n_map, 'SNR [N II] estrecha', header=header, cmap='cividis', vmin=3, vmax=50)
plot_map(flux_nii_n_map, 'Flujo [N II] estrecho', header=header, cmap='inferno')
plot_map(flux_nii_b_map, 'Flujo [N II] ancho', header=header, cmap='inferno')
plot_map(snr_nii_b_map, 'SNR [N II] ancha', header=header, cmap='cividis', vmin=5, vmax=30)



hdul = fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(flux_b_map, name="FLUX_HA_B"),
    fits.ImageHDU(v_b_map, name="VEL_HA_B"),
    fits.ImageHDU(sigma_b_map, name="SIGMA_HA_B"),
    fits.ImageHDU(snr_b_map, name="SNR_HA_B"),
    fits.ImageHDU(flux_n_map, name="FLUX_HA_N"),
    fits.ImageHDU(v_n_map, name="VEL_HA_N"),
    fits.ImageHDU(sigma_n_map, name="SIGMA_HA_N"),
    fits.ImageHDU(snr_n_map, name="SNR_HA_N"),
    fits.ImageHDU(flux_nii_n_map, name="FLUX_NII_N"),
    fits.ImageHDU(flux_nii_b_map, name="FLUX_NII_B"),
])

hdul.writeto("maps_halpha.fits", overwrite=True)