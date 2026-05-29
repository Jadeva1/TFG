import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Traemos el archivo .fits

fits_file = "/home/rauladeva/fit_rss_NGC4750_LR_V.fits"

with fits.open(fits_file) as hdul:
    data = hdul[0].data
    header = hdul[0].header

wcs = WCS(header).celestial

# extraemos los mapas

velocidad       = data[0]
velocidad_error = data[1]
sigma           = data[2]
sigma_error     = data[3]
h3              = data[4]
h3_error        = data[5]
h4              = data[6]
h4_error        = data[7]


vmin_vel, vmax_vel = np.nanpercentile(velocidad, [5, 95])
vmin_sig, vmax_sig = np.nanpercentile(sigma, [2, 98])
h3_lim = np.nanpercentile(np.abs(h3), 98)
h4_lim = np.nanpercentile(np.abs(h4), 98)

maps = [
    {
        "data": velocidad,
        "title": "Velocidad",
        "cbar": r"$v_\star$ [km s$^{-1}$]",
        "cmap": "jet",
        "vmin": vmin_vel,
        "vmax": vmax_vel,
    },
    {
        "data": sigma,
        "title": r"$\sigma_\star$",
        "cbar": r"$\sigma_\star$ [km s$^{-1}$]",
        "cmap": "jet",
        "vmin": vmin_sig,
        "vmax": vmax_sig,
    },
    {
        "data": h3,
        "title": r"$h_3$",
        "cbar": r"$h_3$",
        "cmap": "jet",
        "vmin": -h3_lim,
        "vmax": h3_lim,
    },
    {
        "data": h4,
        "title": r"$h_4$",
        "cbar": r"$h_4$",
        "cmap": "jet",
        "vmin": -h4_lim,
        "vmax": h4_lim,
    },
]

#figura

fig = plt.figure(figsize=(16.5, 4.6))

fig = plt.figure(figsize=(16.5, 4.2))

gs = fig.add_gridspec(
    nrows=1,
    ncols=4,
    left=0.075,
    right=0.955,
    bottom=0.180,
    top=0.960,
    wspace=0.42
)

for i, m in enumerate(maps):

    ax = fig.add_subplot(gs[0, i], projection=wcs)

    im = ax.imshow(
        m["data"],
        origin="lower",
        cmap=m["cmap"],
        vmin=m["vmin"],
        vmax=m["vmax"],
        interpolation="nearest"
    )

    ax.coords[0].set_format_unit(u.hourangle)
    ax.coords[0].set_major_formatter("hh:mm:ss")
    ax.coords[1].set_major_formatter("dd:mm:ss")

    ax.coords[0].set_ticks(number=5)
    ax.coords[1].set_ticks(number=5)

    ax.coords[0].set_axislabel("RA (J2000)", fontsize=13, minpad=0.7)

    if i == 0:
        ax.coords[1].set_axislabel("Dec (J2000)", fontsize=13, minpad=0.8)
    else:
        ax.coords[1].set_axislabel("")
        ax.coords[1].set_ticklabel_visible(False)

    ax.coords[0].set_ticklabel(size=10)
    ax.coords[1].set_ticklabel(size=10)

    ax.coords.grid(True, color="white", ls="--", alpha=0.35)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.10, axes_class=plt.Axes)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(m["cbar"], fontsize=12)
    cbar.ax.tick_params(labelsize=10)

fig.savefig("mapas_ppxf_megades_style.pdf", dpi=300)
fig.savefig("mapas_ppxf_megades_style.png", dpi=300)
plt.close(fig)


print("Guardado: mapas_ppxf_megades_style.pdf")
print("Guardado: mapas_ppxf_megades_style.png")