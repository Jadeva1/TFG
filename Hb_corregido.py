from scipy.optimize import curve_fit
from functools import partial
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np

#fUNCION MUY PARECIDA A AJUSTE_NUBE PERO PARA EL COMPLEJO HBETA [O III]

hdul   = fits.open("/home/rauladeva/residuals_rss_NGC4750_LR_B.fits")
data   = hdul[0].data
header = hdul[0].header
hdul.close()

nl    = header['NAXIS3']
crval = header['CRVAL3']
crpix = header['CRPIX3']
cdelt = header['CDELT3']
z_sys = 0.005404

wave = crval + (np.arange(nl) + 1 - crpix) * cdelt

# comprobaciones para saber si se ha extraido bien
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

# Líneas
HB_REST    = 4861.33
OIII_5007  = 5006.84
OIII_4959  = 4958.91
OIII_RATIO = 3.0


# AJUSTE GAUSSIANO INDIVIDUAL

def gauss_vel(wave, A, v_kms, sigma_kms, lam_rest, z_sys):

    sigma_kms = max(sigma_kms, 1e-3)
    lam0 = lam_rest * (1.0 + z_sys) * (1.0 + v_kms / C_LIGHT)
    sig_lam = lam0 * sigma_kms / C_LIGHT
    return A * np.exp(-0.5 * ((wave - lam0) / sig_lam) ** 2)

def compute_epsilon_c(wave_full, flux_full, continuum, scale, z_sys):

    lam_cont_center = 5050 * (1 + z_sys)
    mask = (wave_full > lam_cont_center - lam_cont_width / 2) & \
           (wave_full < lam_cont_center + lam_cont_width / 2)
    if mask.sum() < 5:
        lam_cont_center = 4820 * (1 + z_sys)
        mask = (wave_full > lam_cont_center - 15) & \
               (wave_full < lam_cont_center + 15)
    if mask.sum() < 5:
        return 1.0
    f_norm_cont = (flux_full[mask] - continuum) / scale
    return np.std(f_norm_cont)

def compute_epsilon_line(wave, residuals_norm, z_sys, include_broad=False):

    lam_hb = HB_REST * (1 + z_sys)
    lam_o5 = OIII_5007 * (1 + z_sys)

    if include_broad:
        margin = 40.0
    else:
        margin = 30.0

    mask = (wave > lam_hb - margin) & (wave < lam_o5 + margin)

    vals = residuals_norm[mask]
    vals = vals[np.isfinite(vals)]

    if vals.size < 5:
        return 0.0

    return np.std(vals)

# MODELO COMPLETO: continuo + estrecho + ancho


def h_o_model(wave, C,
              A_h_n, v_n, sigma_n, A_o_n,
              A_h_b, v_b, sigma_b, A_o_b, z_sys):
    """
    Modelo con 2 componentes: estrecha (n) y ancha (b)..
    """
    m = np.full_like(wave, C, dtype=float)

    # Componente estrecha
    m += gauss_vel(wave, A_h_n,             v_n, sigma_n, HB_REST,   z_sys)
    m += gauss_vel(wave, A_o_n,             v_n, sigma_n, OIII_5007, z_sys)
    m += gauss_vel(wave, A_o_n / OIII_RATIO, v_n, sigma_n, OIII_4959, z_sys)

    # Componente ancha
    m += gauss_vel(wave, A_h_b,             v_b, sigma_b, HB_REST,   z_sys)
    m += gauss_vel(wave, A_o_b,             v_b, sigma_b, OIII_5007, z_sys)
    m += gauss_vel(wave, A_o_b / OIII_RATIO, v_b, sigma_b, OIII_4959, z_sys)

    return m


# MODELO SOLO ESTRECHA (si tenemos pocos residuos ajustando con la estrecha no hace falta la ancha) lo que buscamos es que el SNR sea < 3

def hb_o_narrow_model(wave,
                      C,
                      A_h_n, v_n, sigma_n, A_o_n,
                      z_sys):

    m = np.full_like(wave, C, dtype=float)

    m += gauss_vel(wave, A_h_n,             v_n, sigma_n, HB_REST,   z_sys)
    m += gauss_vel(wave, A_o_n,             v_n, sigma_n, OIII_5007, z_sys)
    m += gauss_vel(wave, A_o_n / OIII_RATIO, v_n, sigma_n, OIII_4959, z_sys)  # FIX: era OIII_5007

    return m


def make_narrow_model(z_sys):
    def model(wave, C, A_h_n, v_n, sigma_n, A_o_n):
        return hb_o_narrow_model(
            wave, C, A_h_n, v_n, sigma_n, A_o_n, z_sys
        )
    return model


# Probamos con unos datos iniciales medio razonables

def estimate_initial_params(wave, flux, z_sys, continuum):
    lam_h_sys = HB_REST  * (1.0 + z_sys)
    lam_o_sys = OIII_5007 * (1.0 + z_sys)

    window_h = 10
    window_o = 10

    mask_h = (wave > lam_h_sys - window_h) & (wave < lam_h_sys + window_h)
    mask_o = (wave > lam_o_sys - window_o) & (wave < lam_o_sys + window_o)

    peak_h = np.nanmax(flux[mask_h]) if mask_h.sum() > 0 else -np.inf
    peak_o = np.nanmax(flux[mask_o]) if mask_o.sum() > 0 else -np.inf

    # Usar la línea más fuerte para inicializar la velocidad
    if peak_o > peak_h and np.isfinite(peak_o):
        lam_peak = wave[mask_o][np.nanargmax(flux[mask_o])]
        v_init = (lam_peak / lam_o_sys - 1.0) * C_LIGHT
    elif np.isfinite(peak_h):
        lam_peak = wave[mask_h][np.nanargmax(flux[mask_h])]
        v_init = (lam_peak / lam_h_sys - 1.0) * C_LIGHT
    else:
        v_init = 0.0

    A_h_init = max(np.max(flux[mask_h])  - continuum, 1e-3) if mask_h.sum() > 0 else 1e-3
    A_o_init = max(np.max(flux[mask_o]) - continuum, 1e-3) if mask_o.sum() > 0 else 1e-3

    sigma_n = sigma_ins * 2

    p0 = [continuum, A_h_init, v_init, sigma_n, A_o_init,
          A_h_init * 0.2, v_init, 400.0, A_o_init * 0.2]

    eps = 1e-20
    lo = [
        continuum - abs(continuum) * 0.5 - eps,
        0,  -300,  sigma_ins, 0,
        0,  -500,  130,       0,
    ]
    hi = [
        continuum + abs(continuum) * 0.5 + eps,
        A_h_init * 15, 300, SIGMA_N_MAX,  A_o_init * 15,
        A_h_init * 10, 500, 1500,          A_o_init * 5,
    ]

    return p0, (lo, hi)


# AJUSTE SPAXEL A SPAXEL

def fit_spaxel(wave, flux, z_sys, continuum, scale, wave_full, flux_full):

    f_norm = (flux - continuum) / scale
    epsilon_c = compute_epsilon_c(wave_full, flux_full, continuum, scale, z_sys)
    narrow_model = make_narrow_model(z_sys)

    p0_n, bounds_n = estimate_initial_params(wave, f_norm, z_sys, continuum=0.0)
    p0_n     = p0_n[:5]
    bounds_n = (list(bounds_n[0][:5]), list(bounds_n[1][:5]))
    bounds_n[1][3] = SIGMA_N_DETECT

    try:
        popt_n_detect, _ = curve_fit(
            narrow_model, wave, f_norm,
            p0=p0_n, bounds=bounds_n,
            maxfev=10000, method='trf'
        )
    except Exception:
        return {'success': False}

    residuals_detect = f_norm - narrow_model(wave, *popt_n_detect)

    epsilon_line = compute_epsilon_line(wave, residuals_detect, z_sys, include_broad=False)
    use_broad    = epsilon_line > 2.5 * epsilon_c

    if use_broad:
        p0, bounds = estimate_initial_params(wave, f_norm, z_sys, continuum=0.0)
        bounds[1][3] = SIGMA_N_MAX
        full_model = partial(h_o_model, z_sys=z_sys)
        try:
            popt, pcov = curve_fit(
                full_model, wave, f_norm,
                p0=p0, bounds=bounds,
                maxfev=10000, method="trf"
            )
        except Exception:
            # Si falla el ajuste ancho, NO descartamos el espaxel.
            # Volvemos al ajuste solo estrecho.
            use_broad = False

        residuals_full = f_norm - full_model(wave, *popt)

        epsilon_line_n_compare = compute_epsilon_line(
            wave, residuals_detect, z_sys, include_broad=True
        )
        epsilon_line_b = compute_epsilon_line(
            wave, residuals_full, z_sys, include_broad=True
        )

        if epsilon_line_b >= 0.95 * epsilon_line_n_compare:
            use_broad = False

    # AJUSTE ESTRECHO FINAL: se calcula SIEMPRE
    # Este será el que usaremos para los mapas estrechos.


    p0_n2, bounds_n2 = estimate_initial_params(wave, f_norm, z_sys, continuum=0.0)
    p0_n2 = p0_n2[:5]
    bounds_n2 = (list(bounds_n2[0][:5]), list(bounds_n2[1][:5]))
    bounds_n2[1][3] = SIGMA_N_MAX

    try:
        popt_n_final, _ = curve_fit(
            narrow_model, wave, f_norm,
            p0=p0_n2,
            bounds=bounds_n2,
            maxfev=10000,
            method='trf'
        )
    except (RuntimeError, ValueError):
        popt_n_final = popt_n_detect

    # Si no hay componente ancha aceptada, el modelo completo será
    # el estrecho rellenado con ceros.
    if not use_broad:
        popt = np.zeros(9)
        popt[:5] = popt_n_final
        popt[7] = 300.0

    full_model = partial(h_o_model, z_sys=z_sys)
    model_norm = full_model(wave, *popt)
    model_phys = model_norm * scale + continuum

    C_n, A_h_n, v_n, sigma_n, A_o_n = popt_n_final

    C, A_h_full_n, v_full_n, sigma_full_n, A_o_full_n, \
        A_h_b, v_b, sigma_b, A_o_b = popt

    sigma_n_int = np.sqrt(max(sigma_n**2 - sigma_ins**2, 0.0))

    lam_h_n = HB_REST * (1 + z_sys) * (1 + v_n / C_LIGHT)
    sig_n_lam = lam_h_n * sigma_n / C_LIGHT
    flux_h_n = A_h_n * sig_n_lam * np.sqrt(2 * np.pi) * scale

    lam_o_n = OIII_5007 * (1 + z_sys) * (1 + v_n / C_LIGHT)
    sig_o_lam = lam_o_n * sigma_n / C_LIGHT
    flux_o_n = A_o_n * sig_o_lam * np.sqrt(2 * np.pi) * scale

    lam_h_b = HB_REST * (1 + z_sys) * (1 + v_b / C_LIGHT)
    sig_b_lam = lam_h_b * sigma_b / C_LIGHT
    flux_h_b = A_h_b * sig_b_lam * np.sqrt(2 * np.pi) * scale

    lam_o_b = OIII_5007 * (1 + z_sys) * (1 + v_b / C_LIGHT)
    sig_o_b_lam = lam_o_b * sigma_b / C_LIGHT
    flux_o_b = A_o_b * sig_o_b_lam * np.sqrt(2 * np.pi) * scale

    snr_n   = A_h_n / epsilon_c
    snr_b   = A_h_b / epsilon_c if use_broad else 0.0
    snr_o_n = A_o_n / epsilon_c
    snr_o_b = A_o_b / epsilon_c if use_broad else 0.0

    flux_total  = flux_h_n + flux_h_b
    ratio_broad = flux_h_b / flux_total if flux_total > 0 else 0.0

    return {
        'success':    True,
        'popt':       popt,
        'popt_n_final': popt_n_final,
        'popt_full': popt,
        'use_broad':  use_broad,
        'epsilon_c':  epsilon_c,
        'epsilon_line': epsilon_line,
        'v_n':        v_n,
        'sigma_n':    sigma_n,
        'sigma_n_int': sigma_n_int,
        'v_b':        v_b,
        'sigma_b':    sigma_b,
        'flux_hb_n':  flux_h_n,
        'flux_hb_b':  flux_h_b,
        'snr_n':      snr_n,
        'snr_b':      snr_b,
        'ratio_broad': ratio_broad,
        'model_phys': model_phys,
        'flux_o_n':   flux_o_n,
        'flux_o_b':   flux_o_b,
        'snr_o_n':    snr_o_n,
        'snr_o_b':    snr_o_b,
    }


def plot_spaxel_fit_with_residuals(wave, flux, result, z_sys,
                                   continuum, scale, i, j,
                                   wave_full, flux_full):

    popt  = result['popt']
    model = result['model_phys']

    C, A_h_n, v_n, sigma_n, A_o_n, \
       A_h_b, v_b, sigma_b, A_o_b = popt


    narrow = (
        gauss_vel(wave, A_h_n, v_n, sigma_n, HB_REST,   z_sys) +
        gauss_vel(wave, A_o_n, v_n, sigma_n, OIII_5007, z_sys) +
        gauss_vel(wave, A_o_n / OIII_RATIO, v_n, sigma_n, OIII_4959, z_sys)
    )

    broad = (
        gauss_vel(wave, A_h_b, v_b, sigma_b, HB_REST,   z_sys) +
        gauss_vel(wave, A_o_b, v_b, sigma_b, OIII_5007, z_sys) +
        gauss_vel(wave, A_o_b / OIII_RATIO, v_b, sigma_b, OIII_4959, z_sys)
    )


    # Pasar a unidades físicas


    narrow_phys = narrow * scale + continuum
    broad_phys  = broad  * scale + continuum

    residuals = flux - model

    lam_h      = HB_REST * (1 + z_sys)
    lam_o4959  = OIII_4959 * (1 + z_sys)
    lam_o5007  = OIII_5007 * (1 + z_sys)
    lam_cont_c = 5050 * (1 + z_sys)

    blue_lo = wave[0]
    blue_hi = wave[-1]

    # Parámetros visuales


    fontsize_axis = 15
    fontsize_ticks = 13
    fontsize_legend = 12
    fontsize_title = 16
    fontsize_lines = 11

    # Panel superior: mantener una vista amplia del espectro


    x_full_min = np.nanmin(wave_full)
    x_full_max = np.nanmax(wave_full)


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

    y_low, y_high = np.nanpercentile(flux_y_scale, [1, 99.5])
    yrange = y_high - y_low

    if (not np.isfinite(yrange)) or (yrange <= 0):
        y_low = np.nanmin(flux_y_scale)
        y_high = np.nanmax(flux_y_scale)
        yrange = y_high - y_low

    ax_full.set_ylim(
        y_low - 0.10 * yrange,
        y_high + 0.20 * yrange
    )

    ax_full.set_ylabel('Flujo', fontsize=fontsize_axis)
    ax_full.set_title(f'Espaxel ({i},{j})', fontsize=fontsize_title)
    ax_full.legend(fontsize=fontsize_legend, loc='upper right')
    ax_full.tick_params(axis='both', labelsize=fontsize_ticks)

    mask_y_scale = (
            np.isfinite(wave_full)
            & np.isfinite(flux_full)
            & (wave_full >= wave[0])
            & (wave_full <= wave[-1])
    )

    flux_y_scale = flux_full[mask_y_scale]


    if len(flux_y_scale) < 5:
        flux_y_scale = flux[np.isfinite(flux)]

    y_low, y_high = np.nanpercentile(flux_y_scale, [1, 99.5])
    yrange = y_high - y_low

    if (not np.isfinite(yrange)) or (yrange <= 0):
        y_low = np.nanmin(flux_y_scale)
        y_high = np.nanmax(flux_y_scale)
        yrange = y_high - y_low

    ax_full.set_ylim(
        y_low - 0.10 * yrange,
        y_high + 0.20 * yrange
    )

    ax_full.set_ylabel('Flujo', fontsize=fontsize_axis)
    ax_full.set_title(f'Spaxel ({i},{j})', fontsize=fontsize_title)
    ax_full.legend(fontsize=fontsize_legend, loc='upper right')
    ax_full.tick_params(axis='both', labelsize=fontsize_ticks)

    # Panel 2: zoom del ajuste


    ax_fit.plot(
        wave,
        flux,
        color='black',
        lw=1.2,
        label='datos'
    )

    ax_fit.plot(
        wave,
        model,
        color='red',
        lw=2.2,
        label='modelo total'
    )

    ax_fit.plot(
        wave,
        narrow_phys,
        color='blue',
        ls='--',
        lw=1.8,
        label=fr'estrecha $\sigma={sigma_n:.0f}$ km s$^{{-1}}$'
    )

    if result["use_broad"]:
        ax_fit.plot(
            wave,
            broad_phys,
            color='green',
            ls='--',
            lw=1.8,
            label=fr'ancha $\sigma={sigma_b:.0f}$ km s$^{{-1}}$'
        )

    ax_fit.axvspan(
        blue_lo,
        blue_hi,
        alpha=0.10,
        color='blue',
        label=fr'$\epsilon_{{line}}={result["epsilon_line"]:.3f}$'
    )

    for lam_mark, lbl in [
        (lam_h, r'H$\beta$'),
        (lam_o4959, r'[O III] 4959'),
        (lam_o5007, r'[O III] 5007')
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


    ax_res.plot(
        wave,
        residuals,
        color='black',
        lw=1.2
    )

    ax_res.axhline(
        0,
        color='red',
        ls='--',
        lw=1.2
    )

    ax_res.axvspan(
        blue_lo,
        blue_hi,
        alpha=0.10,
        color='blue'
    )

    ax_res.set_xlim(wave[0], wave[-1])
    ax_res.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=fontsize_axis)
    ax_res.set_ylabel('Residuos', fontsize=fontsize_axis)
    ax_res.tick_params(axis='both', labelsize=fontsize_ticks)


    plt.tight_layout()
    plt.savefig(f"fit_residuals_{i}_{j}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_map(data_map, title, header=None, cmap='RdBu_r', vmin=None, vmax=None):
    """
    Genera mapa 2D.
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

    plt.colorbar(im, ax=ax, label=title)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').replace('/', '-')}.png", dpi=150)
    plt.close()

# SCRIPT PRINCIPAL

# Regiones de continuo (sin líneas de emisión)
lam_cont_lo_blue = 4740 * (1 + z_sys)
lam_cont_hi_blue = 4800 * (1 + z_sys)
lam_cont_lo_red  = 5060 * (1 + z_sys)
lam_cont_hi_red  = 5100 * (1 + z_sys)

mask_cont = (
    ((wave > lam_cont_lo_blue) & (wave < lam_cont_hi_blue)) |
    ((wave > lam_cont_lo_red)  & (wave < lam_cont_hi_red))
)

lam_fit_lo = HB_REST  * (1 + z_sys) - 30
lam_fit_hi = OIII_5007 * (1 + z_sys) + 30
mask_fit   = (wave > lam_fit_lo) & (wave < lam_fit_hi)
wave_fit   = wave[mask_fit]

print(f"Ventana de ajuste: {lam_fit_lo:.1f} – {lam_fit_hi:.1f} Å  ({mask_fit.sum()} canales)")
print(f"Ventana continuo:  {mask_cont.sum()} canales")

i_c, j_c = data.shape[1] // 2, data.shape[2] // 2

# Test en espaxel central
flux_spaxel = data[:, i_c, j_c]
continuum   = np.nanmedian(flux_spaxel[mask_cont])
scale       = np.nanstd(flux_spaxel[mask_cont])
flux_fit    = flux_spaxel[mask_fit]

print(f"continuum = {continuum:.4e}")
print(f"scale     = {scale:.4e}")
print(f"SNR pico  = {(np.max(flux_fit)-continuum)/scale:.1f}")

result = fit_spaxel(wave_fit, flux_fit, z_sys, continuum, scale, wave, flux_spaxel)
print(f"v_n spaxel central    = {result['v_n']:.1f} km/s")
print(f"sigma_n spaxel central= {result['sigma_n']:.1f} km/s")
print(f"v_b spaxel central    = {result['v_b']:.1f} km/s")
print(f"sigma_b spaxel central= {result['sigma_b']:.1f} km/s")

# Mapas de salida
nl, ny, nx = data.shape
v_n_map     = np.full((ny, nx), np.nan)
sigma_n_map = np.full((ny, nx), np.nan)
v_b_map     = np.full((ny, nx), np.nan)
sigma_b_map = np.full((ny, nx), np.nan)
flux_n_map  = np.full((ny, nx), np.nan)
flux_b_map  = np.full((ny, nx), np.nan)
snr_n_map   = np.full((ny, nx), np.nan)
snr_b_map   = np.full((ny, nx), np.nan)
ratio_map   = np.full((ny, nx), np.nan)
flux_o_n_map = np.full((ny, nx), np.nan)
flux_o_b_map = np.full((ny, nx), np.nan)
snr_o_n_map  = np.full((ny, nx), np.nan)
snr_o_b_map  = np.full((ny, nx), np.nan)

for i in range(ny):
    for j in range(nx):
        flux_spaxel = data[:, i, j]

        if mask_cont.sum() > 3:
            continuum = np.nanmedian(flux_spaxel[mask_cont])
        else:
            continuum = np.nanmedian(flux_spaxel)

        scale = np.nanstd(flux_spaxel[mask_cont]) if mask_cont.sum() > 3 else 1e-17
        if scale < 1e-30:
            continue

        flux_fit = flux_spaxel[mask_fit]

        lam_h_obs = HB_REST * (1 + z_sys)
        lam_o_obs = OIII_5007 * (1 + z_sys)

        mask_h_peak = (wave_fit > lam_h_obs - 5) & (wave_fit < lam_h_obs + 5)
        mask_o_peak = (wave_fit > lam_o_obs - 5) & (wave_fit < lam_o_obs + 5)

        snr_candidates = []

        if mask_h_peak.sum() > 0:
            snr_candidates.append((np.nanmax(flux_fit[mask_h_peak]) - continuum) / scale)

        if mask_o_peak.sum() > 0:
            snr_candidates.append((np.nanmax(flux_fit[mask_o_peak]) - continuum) / scale)

        if len(snr_candidates) == 0:
            continue

        snr_pre = np.nanmax(snr_candidates)

        if (not np.isfinite(snr_pre)) or (snr_pre < 3):
            continue

        result = fit_spaxel(wave_fit, flux_fit, z_sys, continuum, scale, wave, flux_spaxel)

        if not result['success']:
            continue

        snr_hb_n = result.get('snr_n', -np.inf)
        snr_o_n = result.get('snr_o_n', -np.inf)

        if max(snr_hb_n, snr_o_n) < 3:
            continue

        v_n_map[i, j]      = result['v_n']
        sigma_n_map[i, j]  = result['sigma_n_int']
        flux_n_map[i, j]   = result['flux_hb_n']
        snr_n_map[i, j]    = result['snr_n']
        flux_o_n_map[i, j] = result['flux_o_n']
        snr_o_n_map[i, j]  = result['snr_o_n']

        if result['use_broad']:
            v_b_map[i, j]      = result['v_b']
            sigma_b_map[i, j]  = result['sigma_b']
            flux_b_map[i, j]   = result['flux_hb_b']
            snr_b_map[i, j]    = result['snr_b']
            ratio_map[i, j]    = result['ratio_broad']
            flux_o_b_map[i, j] = result['flux_o_b']
            snr_o_b_map[i, j]  = result['snr_o_b']

        if (i % 10 == 0) and (j % 5 == 0):
            plot_spaxel_fit_with_residuals(
                wave_fit, flux_fit, result, z_sys,
                continuum, scale, i, j,
                wave, flux_spaxel
            )
        if (i == 22) and (j == 24):
            plot_spaxel_fit_with_residuals(
                wave_fit, flux_fit, result, z_sys,
                continuum, scale, i, j,
                wave, flux_spaxel
            )

mask_broad = ratio_map > 0.05

# Mapas con coordenadas WCS
plot_map(v_n_map,     'Velocidad [O III] componente estrecha (km/s)',  header=header, vmin=-200, vmax=200)
plot_map(sigma_n_map, 'Dispersión [O III] componente estrecha (km/s)', header=header, cmap='viridis')
plot_map(flux_n_map,  'Flujo Hβ estrecho',          header=header, cmap='inferno')
plot_map(flux_b_map,  'Flujo Hβ ancho',             header=header, cmap='inferno')
plot_map(v_b_map,     'Velocidad [O III] componente ancha (km/s)',      header=header, vmin=-300, vmax=300)
plot_map(sigma_b_map, 'Dispersión [O III] componente ancha (km/s)',     header=header, cmap='viridis')
plot_map(snr_n_map,   'SNR_Hβ_estrecha',            header=header, cmap='cividis', vmin=3, vmax=50)
plot_map(snr_b_map,   'SNR_Hβ_ancha',               header=header, cmap='cividis', vmin=5, vmax=30)
plot_map(snr_o_n_map, 'SNR_OIII_estrecha',          header=header, cmap='cividis', vmin=3, vmax=50)
plot_map(flux_o_n_map,'Flujo OIII componente estrecha',         header=header, cmap='inferno')
plot_map(flux_o_b_map,'Flujo OIII componente ancha',            header=header, cmap='inferno')
plot_map(snr_o_b_map, 'SNR_OIII_ancha',             header=header, cmap='cividis', vmin=5, vmax=30)

hdul = fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(np.asarray(flux_b_map, dtype=float), name="FLUX_HB_B"),
    fits.ImageHDU(np.asarray(v_b_map, dtype=float), name="VEL_HB_B"),
    fits.ImageHDU(np.asarray(sigma_b_map, dtype=float), name="SIGMA_HB_B"),
    fits.ImageHDU(np.asarray(snr_b_map, dtype=float), name="SNR_HB_B"),
    fits.ImageHDU(np.asarray(flux_n_map, dtype=float), name="FLUX_HB_N"),
    fits.ImageHDU(np.asarray(v_n_map, dtype=float), name="VEL_HB_N"),
    fits.ImageHDU(np.asarray(sigma_n_map, dtype=float), name="SIGMA_HB_N"),
    fits.ImageHDU(np.asarray(snr_n_map, dtype=float), name="SNR_HB_N"),
])

hdul.writeto("maps_hbeta.fits", overwrite=True)

hdul = fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(np.asarray(flux_o_n_map, dtype=float), name="FLUX_OIII_N"),
    fits.ImageHDU(np.asarray(v_n_map, dtype=float), name="VEL_OIII_N"),
    fits.ImageHDU(np.asarray(sigma_n_map, dtype=float), name="SIGMA_OIII_N"),
    fits.ImageHDU(np.asarray(snr_o_n_map, dtype=float), name="SNR_OIII_N"),
    fits.ImageHDU(np.asarray(flux_o_b_map, dtype=float), name="FLUX_OIII_B"),
    fits.ImageHDU(np.asarray(v_b_map, dtype=float), name="VEL_OIII_B"),
    fits.ImageHDU(np.asarray(sigma_b_map, dtype=float), name="SIGMA_OIII_B"),
    fits.ImageHDU(np.asarray(snr_o_b_map, dtype=float), name="SNR_OIII_B"),])

hdul.writeto("maps_oiii.fits", overwrite=True)

