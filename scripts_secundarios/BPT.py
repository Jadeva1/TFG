import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# Treamos los mapas para sacar la info


HALPHA_FITS = "/home/rauladeva/PyCharmMiscProject/maps_halpha.fits"
HBETA_FITS  = "/home/rauladeva/PyCharmMiscProject/maps_hbeta.fits"
OIII_FITS   = "/home/rauladeva/PyCharmMiscProject/maps_oiii.fits"

# Escala espacial
pixscale_arcsec = 0.4
scale_kpc_arcsec = 0.129

# Corte mínimo de SNR
snr_min = 3.0


def get_ext(hdul, possible_names):

    names = [h.name for h in hdul]

    for name in possible_names:
        if name in names:
            return hdul[name].data.copy()



def bpt_kewley01(x):
    """
    Línea de separación máxima de starburst: Kewley
    """
    return 0.61 / (x - 0.47) + 1.19


def bpt_kauffmann03(x):
    """
    Línea empírica SF/composite: Kauffmann
    """
    return 0.61 / (x - 0.05) + 1.30

def bpt_kewley06_sy_liner(x):
    """
    Línea de separación Seyfert/LINER de Kewley
    """
    return 1.05 * x + 0.45


def compute_radius_map(shape, center_y, center_x):
    """
    Radio proyectado respecto al centro, en kpc.
    """
    yy, xx = np.indices(shape)
    r_pix = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    r_arcsec = r_pix * pixscale_arcsec
    r_kpc = r_arcsec * scale_kpc_arcsec
    return r_kpc


def compute_bpt_points(flux_ha, flux_nii, flux_hb, flux_oiii,
                       r_kpc, snr_ha=None, snr_hb=None, snr_oiii=None):
    """
    Calcula los puntos del diagrama BPT para una componente.
    """

    mask = (
        np.isfinite(flux_ha)
        & np.isfinite(flux_nii)
        & np.isfinite(flux_hb)
        & np.isfinite(flux_oiii)
        & (flux_ha > 0)
        & (flux_nii > 0)
        & (flux_hb > 0)
        & (flux_oiii > 0)
    )

    if snr_ha is not None:
        mask &= np.isfinite(snr_ha) & (snr_ha >= snr_min)

    if snr_hb is not None:
        mask &= np.isfinite(snr_hb) & (snr_hb >= snr_min)

    if snr_oiii is not None:
        mask &= np.isfinite(snr_oiii) & (snr_oiii >= snr_min)

    x = np.log10(flux_nii[mask] / flux_ha[mask])
    y = np.log10(flux_oiii[mask] / flux_hb[mask])
    c = r_kpc[mask]

    return x, y, c, mask



# CARGAR MAPAS


with fits.open(HALPHA_FITS) as hdul:
    print("Extensiones Halpha:", [h.name for h in hdul])

    flux_ha_n = get_ext(hdul, ["FLUX_HA_N"])
    flux_ha_b = get_ext(hdul, ["FLUX_HA_B"])

    snr_ha_n = get_ext(hdul, ["SNR_HA_N"])
    snr_ha_b = get_ext(hdul, ["SNR_HA_B"])

    # Revisa estos nombres según tu FITS
    flux_nii_n = get_ext(hdul, ["FLUX_NII_N", "FLUX_NII6584_N", "FLUX_NII_6584_N"])
    flux_nii_b = get_ext(hdul, ["FLUX_NII_B", "FLUX_NII6584_B", "FLUX_NII_6584_B"])


with fits.open(HBETA_FITS) as hdul:
    print("Extensiones Hbeta:", [h.name for h in hdul])

    flux_hb_n = get_ext(hdul, ["FLUX_HB_N"])
    flux_hb_b = get_ext(hdul, ["FLUX_HB_B"])

    snr_hb_n = get_ext(hdul, ["SNR_HB_N"])
    snr_hb_b = get_ext(hdul, ["SNR_HB_B"])


with fits.open(OIII_FITS) as hdul:
    print("Extensiones OIII:", [h.name for h in hdul])

    flux_oiii_n = get_ext(hdul, ["FLUX_OIII_N", "FLUX_OIII5007_N", "FLUX_OIII_5007_N"])
    flux_oiii_b = get_ext(hdul, ["FLUX_OIII_B", "FLUX_OIII5007_B", "FLUX_OIII_5007_B"])

    snr_oiii_n = get_ext(hdul, ["SNR_OIII_N", "SNR_OIII5007_N", "SNR_OIII_5007_N"])
    snr_oiii_b = get_ext(hdul, ["SNR_OIII_B", "SNR_OIII5007_B", "SNR_OIII_5007_B"])


# Usamos el máximo de Halpha estrecha como centro.
center_y, center_x = np.unravel_index(np.nanargmax(flux_ha_n), flux_ha_n.shape)

print(f"Centro adoptado: y={center_y}, x={center_x}")

r_kpc = compute_radius_map(flux_ha_n.shape, center_y, center_x)

# CALCULAR BPT ESTRECHO Y ANCHO


x_n, y_n, r_n, mask_bpt_n = compute_bpt_points(
    flux_ha=flux_ha_n,
    flux_nii=flux_nii_n,
    flux_hb=flux_hb_n,
    flux_oiii=flux_oiii_n,
    r_kpc=r_kpc,
    snr_ha=snr_ha_n,
    snr_hb=snr_hb_n,
    snr_oiii=snr_oiii_n
)

x_b, y_b, r_b, mask_bpt_b = compute_bpt_points(
    flux_ha=flux_ha_b,
    flux_nii=flux_nii_b,
    flux_hb=flux_hb_b,
    flux_oiii=flux_oiii_b,
    r_kpc=r_kpc,
    snr_ha=snr_ha_b,
    snr_hb=snr_hb_b,
    snr_oiii=snr_oiii_b
)

print(f"N puntos BPT componente estrecha: {len(x_n)}")
print(f"N puntos BPT componente ancha: {len(x_b)}")

# PLOT BPT RESUELTO


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

x_grid_kauff = np.linspace(-1.5, 0.04, 500)
x_grid_kewley = np.linspace(-1.5, 0.46, 500)
x_grid_sy_liner = np.linspace(-0.2, 1.0, 500)

# Para que ambas figuras tengan la misma escala de color
r_all = np.concatenate([r_n, r_b]) if len(r_b) > 0 else r_n
vmin_r = np.nanmin(r_all)
vmax_r = np.nanmax(r_all)

# Componente estrecha

sc1 = axes[0].scatter(
    x_n,
    y_n,
    c=r_n,
    cmap="viridis",
    s=35,
    edgecolor="none",
    vmin=vmin_r,
    vmax=vmax_r
)

axes[0].plot(
    x_grid_kauff,
    bpt_kauffmann03(x_grid_kauff),
    "k--",
    lw=1.5,
    label="Kauffmann+2003"
)

axes[0].plot(
    x_grid_kewley,
    bpt_kewley01(x_grid_kewley),
    "k-",
    lw=1.5,
    label="Kewley+2001"
)
axes[0].plot(
    x_grid_sy_liner,
    bpt_kewley06_sy_liner(x_grid_sy_liner),
    color="gray",
    ls="-.",
    lw=1.8,
    label="Kewley+2006 Sy/LINER"
)

axes[1].plot(
    x_grid_sy_liner,
    bpt_kewley06_sy_liner(x_grid_sy_liner),
    color="gray",
    ls="-.",
    lw=1.8,
    label="Kewley+2006 Sy/LINER"
)

axes[0].text(
    -1.15, -0.65,
    "SF",
    fontsize=14,
    fontweight="bold",
    color="black"
)

axes[0].text(
    -0.20, -1.35,
    "Comp",
    fontsize=13,
    fontweight="bold",
    color="black",
    rotation=25
)

axes[0].text(
    -0.5, 0.7,
    "Seyfert",
    fontsize=13,
    fontweight="bold",
    color="black"
)

axes[0].text(
    0.4, -0.45,
    "LINER",
    fontsize=13,
    fontweight="bold",
    color="black"
)

axes[0].set_title("Componente estrecha", fontsize=15)
axes[0].set_xlabel(r"$\log([{\rm NII}]\lambda6584/{\rm H}\alpha)$", fontsize=14)
axes[0].set_ylabel(r"$\log([{\rm OIII}]\lambda5007/{\rm H}\beta)$", fontsize=14)
axes[0].legend(fontsize=11, loc="upper left")
axes[0].tick_params(axis="both", labelsize=12)

# Componente ancha

sc2 = axes[1].scatter(
    x_b,
    y_b,
    c=r_b,
    cmap="viridis",
    s=45,
    edgecolor="none",
    vmin=vmin_r,
    vmax=vmax_r
)

axes[1].plot(
    x_grid_kauff,
    bpt_kauffmann03(x_grid_kauff),
    "k--",
    lw=1.5,
    label="Kauffmann+2003"
)

axes[1].plot(
    x_grid_kewley,
    bpt_kewley01(x_grid_kewley),
    "k-",
    lw=1.5,
    label="Kewley+2001"
)

axes[1].plot(
    x_grid_sy_liner,
    bpt_kewley06_sy_liner(x_grid_sy_liner),
    color="gray",
    ls="-.",
    lw=1.8,
    label="Kewley+2006 Sy/LINER"
)

axes[1].text(
    -1.15, -0.65,
    "SF",
    fontsize=14,
    fontweight="bold",
    color="black"
)

axes[1].text(
    -0.20, -1.35,
    "Comp",
    fontsize=13,
    fontweight="bold",
    color="black",
    rotation=25
)

axes[1].text(
    0, 1.2,
    "Seyfert",
    fontsize=13,
    fontweight="bold",
    color="black"
)

axes[1].text(
    0.4, -0.45,
    "LINER",
    fontsize=13,
    fontweight="bold",
    color="black"
)


axes[1].set_title("Componente ancha", fontsize=15)
axes[1].set_xlabel(r"$\log([{\rm NII}]\lambda6584/{\rm H}\alpha)$", fontsize=14)
axes[1].tick_params(axis="both", labelsize=12)


for ax in axes:
    ax.set_xlim(-1.5, 1.0)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(alpha=0.25)

cbar = fig.colorbar(sc2, ax=axes, pad=0.02)
cbar.set_label("Distancia al centro [kpc]", fontsize=14)
cbar.ax.tick_params(labelsize=12)

plt.savefig("/home/rauladeva/PyCharmMiscProject/bpt.png",
            dpi=300, bbox_inches="tight")
plt.close()

print("Figura guardada: bpt.png")
