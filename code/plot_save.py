"""
LaTeX-compatible plotting utility for generating vector graphics from matplotlib.
Creates both PDF and PGF format outputs for seamless LaTeX integration.

Enhanced version with improved structure and customization options.
"""

import matplotlib as mpl
mpl.use('pgf')

from math import sqrt

class LaTeXPlotter:
    """Enhanced plotting class for LaTeX-compatible figure generation"""
    
    def __init__(self, base_width=4.98132*0.8, aspect_ratio=None):
        # Calculate golden ratio if not provided
        self.base_width = base_width
        self.aspect_ratio = aspect_ratio or (sqrt(5.0) - 1.0) / 2.0
        
        # Configure matplotlib for LaTeX output
        self._configure_matplotlib()
    
    def _configure_matplotlib(self):
        """Configure matplotlib settings for LaTeX compatibility"""
        plot_config = {
            "text.usetex": True,
            "pgf.texsystem": "xelatex", 
            "pgf.rcfonts": False,
            "font.family": "serif",
            "font.sans-serif": [],
            "font.monospace": [],
            "figure.figsize": [self.base_width, self.base_width * self.aspect_ratio],
            "pgf.preamble": "\n".join([
                r"\usepackage[utf8]{inputenc}",
                # Additional LaTeX packages can be added here
            ])
        }
        
        mpl.rcParams.update(plot_config)
    
    def create_plot(self, width=None, ratio=None, padding=0, *args, **kwargs):
        """Create a new figure with optimized dimensions"""
        plot_width = width or self.base_width
        plot_ratio = ratio or self.aspect_ratio
        
        fig = mpl.pyplot.figure(figsize=(plot_width, plot_width * plot_ratio), *args, **kwargs)
        fig.set_tight_layout({'pad': padding})
        return fig
    
    def create_subplots(self, width=None, ratio=None, *args, **kwargs):
        """Create subplot arrangement with consistent sizing"""
        plot_width = width or self.base_width
        plot_ratio = ratio or self.aspect_ratio
        
        fig, axes = mpl.pyplot.subplots(figsize=(plot_width, plot_width * plot_ratio), *args, **kwargs)
        fig.set_tight_layout({'pad': 0})
        return fig, axes
    
    def export_figure(self, file_basename, *args, **kwargs):
        """Export current figure in both PDF and PGF formats"""
        output_files = [
            f"{file_basename}_vector.pdf",
            f"{file_basename}_latex.pgf"
        ]
        
        for output_file in output_files:
            mpl.pyplot.savefig(output_file, *args, **kwargs)
        
        return output_files

# Create global instance for backward compatibility
latex_plot_manager = LaTeXPlotter()

# Convenience functions
def new_figure(width=None, ratio=None, pad=0, *args, **kwargs):
    """Create a new figure with LaTeX-compatible settings"""
    return latex_plot_manager.create_plot(width, ratio, pad, *args, **kwargs)

def new_subplots(width=None, ratio=None, *args, **kwargs):
    """Create subplots with LaTeX-compatible settings"""
    return latex_plot_manager.create_subplots(width, ratio, *args, **kwargs)

def save_plot(basename, *args, **kwargs):
    """Save current plot in both PDF and PGF formats"""
    return latex_plot_manager.export_figure(basename, *args, **kwargs)

# Import pyplot for direct use
import matplotlib.pyplot as plt
