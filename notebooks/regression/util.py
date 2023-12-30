import numpy as np
import matplotlib.pyplot as plt

def regression_plot_with_uncertainty(
    ax,
    prediction_inputs,
    mean_prediction,
    prediction_bounds,
    scatter_points,
    quantiles=[1.0],
    labels=['Estimated Error']
):
    """
    Produce a plot with the mean prediction as a line and the bounds as shaded area.
    If multiple quantiles are given, the shaded areas are plotted sequentially starting from the largest quantile.
    The scatter points are plotted separately.

    Parameters
    ----------
    ax : matplotlib subplot axis
    prediction_inputs : NDArray of shape (n_samples, 1)
        Locations where the mean and bounds are predicted
    mean_prediction : NDArray of shape (n_samples, 1)
        Predictive mean (standard model prediction)
    prediction_bounds : NDArray of shape (n_samples, 2, n_quantiles)
        Predictive upper and lower bound for the respective quantiles
    scatter_points : Tuple[NDArray of shape (n_scatter_samples, 1), NDArray of shape (n_scatter_samples, 1)]
        Additional inputs / labels to be plotted on top
    quantiles : list
        A list of quantiles corresponding to the different prediction_bounds
    labels : list
        A list of labels corresponding to each quantile that are added to the plot's legend
    """
    
    order = np.argsort(prediction_inputs[:, 0], axis=0)

    cmap = plt.cm.Oranges
    indices = np.linspace(0.2, 0.4, num=len(quantiles))
    colors = cmap(indices)

    # We reverse the quantiles because we want to plot the largest bounds first
    quantile_indices = np.argsort(quantiles)[::-1]
    for i in quantile_indices:
        ax.fill_between(
            prediction_inputs[order].ravel(),
            prediction_bounds[order][:, 0, i].ravel(),
            prediction_bounds[order][:, 1, i].ravel(),
            alpha=1.0,
            color=colors[i],
            label=labels[i]
        )

    ax.scatter(scatter_points[0], scatter_points[1], color='blue', s=5)
    ax.plot(prediction_inputs[order], mean_prediction[order], color='red', label='Prediction')
    ax.legend()