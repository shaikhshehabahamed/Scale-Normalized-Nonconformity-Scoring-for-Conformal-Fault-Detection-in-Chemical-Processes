"""Publication-quality matplotlib style for TEP paper figures.

Usage:
    import paper_style
    paper_style.apply()
    fig, ax = plt.subplots(figsize=(paper_style.W2, 4))
    plt.savefig(path)   # dpi=300 applied automatically via rcParams
"""
import matplotlib as mpl

_RC = {
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size':          9,
    'axes.labelsize':     9,
    'axes.titlesize':     10,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '0.7',
    'axes.linewidth':     0.8,
    'grid.linewidth':     0.4,
    'grid.alpha':         0.4,
    'lines.linewidth':    1.5,
    'lines.markersize':   5,
    'patch.linewidth':    0.5,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
}

W1 = 3.54   # single column — Elsevier two-column format (90 mm)
W2 = 7.48   # double column (190 mm)

def apply():
    """Apply paper-quality rcParams. Call once at the top of each script."""
    mpl.rcParams.update(_RC)
