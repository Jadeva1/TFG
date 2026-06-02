import numpy as np
import matplotlib.pyplot as plt

# DATOS DE LITERATURA

# Fluetsch

v_fluethsch_ion = np.array([
    780., 410., 790., 740., 300., 610., 670., 900., 440., 300.,
    1090., 520., 200., 370., 350., 300., 160., 760., 800., 1500.,
    240., 100., 260., 320., 200., 370., 480., 230., 1550., 230.,
    200., 800.
])

dotM_fluethsch_ion = np.array([
    3.1, 0.6, 1.43, 1.23, 0.09, 0.13, 2.9, 130., 0.66, 0.95,
    26., 1.62, 0.49, 0.05, 8.42, 0.16, 0.004, 0.07, 32.,
    6.1, 0.07, 0.68, 0.017, 0.35, 0.71, 0.22, 1.36, 0.26,
    84., 0.07, 0.05, 0.3
])

sfr_fluethsch_ion = np.array([
    157, 8.5, 53, 122, 50.9, 200., 138., 330., 43, 20.,
    75., 112, 3., 0.23, 45., 189., 95, 20, 148, 270,
    95, 20, 234, 80, 95, 139, 2.3, 20, 930, 120, 6.5, 0.6
])


# Fiore

logSFR_fiore_3 = np.array([
    1.91, 1.56, 1.46, 2.08, 1.36, 1.26, 1.80, 0.85, 1.48, 1.51,
    2.17, 1.38, 2.03, 1.64, 1.62, 2.26, 3.18, 2.90, 2.00, 1.70,
    1.95, 2.93, 3.11, 2.63, 3.15, 2.44, 2.36, 2.57, 2.5, 1.62,
    1.98, 2.06, 1.82, 1.28, 1.64, 1.18, 2.08, 1.57, 1.57, 3.00
])

logM_dot_fiore_3 = np.array([
    1.62, 1.1, 1.16, 1.82, 1.46, 0.3, 1.60, 1.70, 2.45, 1.50,
    3.23, 1.72, 2.34, 2.06, 2.83, 3.81, 3.76, 3.52, 3.66, 3.14,
    1.76, 1.48, 1.57, 1.48, 1.44, 2.39, 1.84, 2.65, 2.29, 0.27,
    0.11, -0.50, 2.60, 2.30, 2.87, 2.59, 2.79, 2.81, 3.16, 3.54
])

v_fiore_3 = np.array([
    1511, 866, 761, 1523, 1267, 616, 1049, 999, 1300, 214,
    530, 475, 300, 300, 1240, 2160, 1890, 2380, 2300, 900,
    1450, 1234, 1124, 1200, 1054, 1500, 1950, 1600, 1900, 2817,
    535, 665, 1191, 946, 1250, 522, 939, 1046, 1821, 1300
])


# Leung

SFR_leung = np.array([
    105, 355, 7, 1, 155, 1, 219, 219, 6, 170, 51, 87,
    1, 2, 26, 35, 158, 6, 1, 21, 63, 78, 23, 25, 355, 3, 8, 129
])

M_dot_leung = np.array([
    7.1, 60.9, 8.8, 62.9, 8050.4, 55.8, 1.2, 0.1, 1973.1,
    11.0, 26.9, 78.5, 768.8, 4.4, 5.4, 3.2, 3.6, 18.4,
    1805.5, 7.9, 44.4, 361.3, 3.7, 3.4, 81.8, 16.2, 15.5, 3.4
])

v_leung = np.array([
    2605, 1706, 534, 490, 2731, 1513, 742, 446, 1807, 981,
    1056, 816, 3481, 667, 678, 498, 583, 1046, 1292, 988,
    1353, 894, 557, 388, 1432, 1060, 687, 608
])

# HECKMAN

v_heck = np.array([
    320, 410, 360, 500, 560, 510, 480, 350, 280, 240,
    450, 450, 290, 30, 400, 330, 360, 280, 350, 450,
    500, 160, 130, 34, 38, 120, 21, 38, 10, 180,
    10, 100, 34, 58, 10, 17, 65, 82, 46
])

Mdot_heck = np.array([
    33, 26, 97, 39, 9, 34, 74, 48, 37, 30,
    30, 99, 45, 3.5, 15, 47, 21, 21, 35, 28,
    46, 45, 60, 4.8, 2.3, 33, 1, 4.6, 1, 22,
    1, 12, 1, 13, 6.4, 0.6, 5.4, 30, 4.6
])

SFR_heck = np.array([
    15, 24, 37, 19, 8, 10, 29, 10, 11, 8,
    29, 7, 9, 5, 23, 14, 27, 6, 9, 36,
    41, 36, 66, 0.83, 0.32, 5.0, 0.16, 6.0, 0.016,
    2.8, 0.41, 40, 0.13, 21, 3.5, 0.33, 2.1, 4.8, 6.9
])

logSFR_heck = np.log10(SFR_heck)
logMdot_heck = np.log10(Mdot_heck)
logv_heck = np.log10(v_heck)

# SFR de nuestra galaxia, Cano-Díaz

SFR_ngc4750= 0.12

# ERRORES DE NUESTRA GAL PARA LAS GRÁFICAS

logSFR_ngc4750 = -0.921
logSFR_ngc4750_err = 0.109

# Halpha
logMdot_ha = -1.263
logMdot_ha_err = 0.166

logv_ha = 2.887
logv_ha_err = 0.008

# [OIII]
logMdot_oiii = -2.729
logMdot_oiii_err = 0.187

logv_oiii = 2.731
logv_oiii_err = 0.020



# VALORES DE HERMOSA-MUÑOZ+2024 PARA NGC 4750

Mdot_hm_ngc4750 = 0.08
vout_hm_ngc4750 = 191.0

logMdot_hm_ngc4750 = np.log10(Mdot_hm_ngc4750)
logv_hm_ngc4750 = np.log10(vout_hm_ngc4750)

logSFR_fluetsch = np.log10(sfr_fluethsch_ion)
logMdot_fluetsch = np.log10(dotM_fluethsch_ion)
logv_fluetsch = np.log10(v_fluethsch_ion)

logSFR_fiore = logSFR_fiore_3
logMdot_fiore = logM_dot_fiore_3
logv_fiore = np.log10(v_fiore_3)


logSFR_leung = np.log10(SFR_leung)
logMdot_leung = np.log10(M_dot_leung)
logv_leung = np.log10(v_leung)


SFR_heck = np.array([
    -15, 24, 37, 19, 8, -10, 29, -10, -11, -8,
    29, 7, 9, -5, 23, 14, -27, 6, -9, 36,
    41, 36, 66, -0.83, -0.32, 5, -0.16, 6, -0.016,
    2.8, -0.41, 40, -0.13, 21, 3.5, -0.33, -2.1, 4.8, 6.9
])

Mdot_heck = np.array([
    33, 26, 97, 39, 9, 34, 74, 48, 37, 30,
    30, 99, 45, 3.5, 15, 47, 21, 21, 35, 28,
    46, 45, 60, 4.8, 2.3, 33, 1, 4.6, 1, 22,
    1, 12, 1, 13, 6.4, 0.6, 5.4, 30, 4.6
])

v_heck = np.array([
    320, 410, 360, 500, 560, 510, 480, 350, 280, 240,
    450, 450, 290, 30, 400, 330, 360, 280, 350, 450,
    500, 160, 130, 34, 38, 120, 21, 38, 10, 180,
    10, 100, 34, 58, 10, 17, 65, 82, 46
])

mask_heck = SFR_heck > 0

logSFR_heck = np.log10(SFR_heck[mask_heck])
logMdot_heck = np.log10(Mdot_heck[mask_heck])
logv_heck = np.log10(v_heck[mask_heck])


gal_hm = np.array([
    "NGC 1052",
    "NGC 3226",
    "NGC 3245",
    "NGC 4278",
    "NGC 4438",
    "NGC 4750"
])

# SFR individual de cada galaxia
SFR_hm = np.array([
    0.09,
    0.038,
    0.1691,
    10**(-1.24),
    10**(-0.30),
    0.12
])

# Valores de Hermosa-Muñoz
Mdot_hm = np.array([
    0.36,
    0.26,
    0.11,
    0.004,
    0.004,
    0.08
])

vout_hm = np.array([
    655.0,
    138.0,
    580.0,
    241.0,
    213.0,
    191.0
])

logSFR_hm = np.log10(SFR_hm)
logMdot_hm = np.log10(Mdot_hm)
logv_hm = np.log10(vout_hm)

idx_hm_ngc4750 = np.where(gal_hm == "NGC 4750")[0][0]

idx_hm_ngc4750 = np.where(gal_hm == "NGC 4750")[0][0]

# GRÁFICA 1: log(Mdot_out) vs log(SFR)


plt.figure(figsize=(8.5, 6.5))

# Tamaños de fuente para que quede bien en TFG
fontsize_axis = 17
fontsize_ticks = 14
fontsize_legend = 10
fontsize_title = 17

plt.scatter(
    logSFR_fluetsch,
    logMdot_fluetsch,
    marker="o",
    s=45,
    alpha=0.65,
    label="Galaxias locales, outflows ionizados, Fluetsch-2021"
)

plt.scatter(
    logSFR_fiore,
    logMdot_fiore,
    marker="s",
    s=45,
    alpha=0.65,
    label="AGN/QSOs,outflows ionizados, Fiore-2017"
)

plt.scatter(
    logSFR_leung,
    logMdot_leung,
    marker="^",
    s=45,
    alpha=0.65,
    label="Galaxias con formación estelar, Leung-2019"
)

plt.scatter(
    logSFR_heck,
    logMdot_heck,
    marker="D",
    s=45,
    alpha=0.65,
    label="Starbursts compactas, Heckman-2015"
)

# NGC 4750 - mis resultados
plt.errorbar(
    logSFR_ngc4750,
    logMdot_ha,
    xerr=logSFR_ngc4750_err,
    yerr=logMdot_ha_err,
    fmt="*",
    markersize=20,
    markeredgecolor="black",
    markeredgewidth=1.5,
    capsize=4,
    elinewidth=1.2,
    linestyle="none",
    label="NGC 4750 Hα: LINER, este trabajo"
)

plt.errorbar(
    logSFR_ngc4750,
    logMdot_oiii,
    xerr=logSFR_ngc4750_err,
    yerr=logMdot_oiii_err,
    fmt="P",
    markersize=15,
    markeredgecolor="black",
    markeredgewidth=1.5,
    capsize=4,
    elinewidth=1.2,
    linestyle="none",
    label="NGC 4750 [O III]: LINER, este trabajo"
)


plt.scatter(
    logSFR_hm,
    logMdot_hm,
    marker="X",
    s=65,
    alpha=0.85,
    edgecolor="black",
    linewidth=0.8,
    label="LINERs, MEGARA, Hermosa-Muñoz-2024"
)

# Marcar específicamente NGC 4750 en Hermosa-Muñoz
plt.scatter(
    logSFR_hm[idx_hm_ngc4750],
    logMdot_hm[idx_hm_ngc4750],
    marker="X",
    s=100,
    edgecolor="black",
    linewidth=1.3,
    label="NGC 4750: Hermosa-Muñoz-2024"
)

plt.xlabel(r"$\log({\rm SFR})\ [M_\odot\,{\rm yr}^{-1}]$", fontsize=fontsize_axis)
plt.ylabel(r"$\log(\dot{M}_{\rm out})\ [M_\odot\,{\rm yr}^{-1}]$", fontsize=fontsize_axis)

plt.tick_params(axis="both", labelsize=fontsize_ticks)
plt.legend(fontsize=fontsize_legend, frameon=True)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("/home/rauladeva/PyCharmMiscProject/plot_logMdot_vs_logSFR.png", dpi=300, bbox_inches="tight")
plt.close()

# GRÁFICA 2: log(Vout) vs log(SFR)

plt.figure(figsize=(8.5, 6.5))

fontsize_axis = 17
fontsize_ticks = 14
fontsize_legend = 10
fontsize_title = 17

plt.scatter(
    logSFR_fluetsch,
    logv_fluetsch,
    marker="o",
    s=45,
    alpha=0.65,
    label="Galaxias locales, outflows ionizados, Fluetsch-2021"
)

plt.scatter(
    logSFR_fiore,
    logv_fiore,
    marker="s",
    s=45,
    alpha=0.65,
    label="AGN/QSOs,outflows ionizados, Fiore-2017"
)

plt.scatter(
    logSFR_leung,
    logv_leung,
    marker="^",
    s=45,
    alpha=0.65,
    label="Galaxias con formación estelar, Leung-2019"
)

plt.scatter(
    logSFR_heck,
    logv_heck,
    marker="D",
    s=45,
    alpha=0.65,
    label="Starbursts compactas, Heckman-2015"
)

# NGC 4750 mis resultados
plt.errorbar(
    logSFR_ngc4750,
    logv_ha,
    xerr=logSFR_ngc4750_err,
    yerr=logv_ha_err,
    fmt="*",
    markersize=20,
    markeredgecolor="black",
    markeredgewidth=1.5,
    capsize=4,
    elinewidth=1.2,
    linestyle="none",
    label="NGC 4750 Hα: LINER, este trabajo"
)

plt.errorbar(
    logSFR_ngc4750,
    logv_oiii,
    xerr=logSFR_ngc4750_err,
    yerr=logv_oiii_err,
    fmt="P",
    markersize=15,
    markeredgecolor="black",
    markeredgewidth=1.5,
    capsize=4,
    elinewidth=1.2,
    linestyle="none",
    label="NGC 4750 [O III]: LINER, este trabajo"
)

plt.scatter(
    logSFR_hm,
    logv_hm,
    marker="X",
    s=65,
    alpha=0.85,
    edgecolor="black",
    linewidth=0.8,
    label="LINERs, MEGARA, Hermosa-Muñoz-2024"
)


plt.scatter(
    logSFR_hm[idx_hm_ngc4750],
    logv_hm[idx_hm_ngc4750],
    marker="X",
    s=100,
    edgecolor="black",
    linewidth=0.8,
    label="NGC 4750: Hermosa-Muñoz-2024"
)

plt.xlabel(r"$\log({\rm SFR})\ [M_\odot\,{\rm yr}^{-1}]$", fontsize=fontsize_axis)
plt.ylabel(r"$\log(v_{\rm out})\ [{\rm km\,s}^{-1}]$", fontsize=fontsize_axis)

plt.tick_params(axis="both", labelsize=fontsize_ticks)
plt.legend(fontsize=fontsize_legend, frameon=True)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("/home/rauladeva/PyCharmMiscProject/plot_logVout_vs_logSFR.png", dpi=300, bbox_inches="tight")
plt.close()


print("Plots guardados correctamente.")
print(f"SFR adoptada para NGC 4750 = {SFR_ngc4750:.3f} Msun/yr")
print(f"log(SFR) NGC 4750 = {logSFR_ngc4750:.3f}")
print(f"Tu Halpha: logMdot = {logMdot_ha:.2f}, logv = {logv_ha:.2f}")
print(f"Tu [OIII]: logMdot = {logMdot_oiii:.2f}, logv = {logv_oiii:.2f}")
print(f"Hermosa-Muñoz NGC 4750: logMdot = {logMdot_hm_ngc4750:.2f}, logv = {logv_hm_ngc4750:.2f}")
