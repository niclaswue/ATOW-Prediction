import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorlover as cl


def plot_metric_overview(model_results):
    models = list(model_results.keys())
    metrics = list(model_results[models[0]].keys())

    # Calculate the grid size
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)  # Max 3 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create color scale
    n_models = len(models)
    colors = cl.scales[str(max(3, min(n_models, 12)))]["qual"]["Set1"]

    # Create subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=metrics)

    # Add traces
    for i, metric in enumerate(metrics):
        row = i // n_cols + 1
        col = i % n_cols + 1

        for j, model in enumerate(models):
            fig.add_trace(
                go.Bar(
                    x=[model],
                    y=[model_results[model][metric]],
                    name=model,
                    marker_color=colors[j % len(colors)],
                    showlegend=i == 0,  # Show legend only for the first metric
                ),
                row=row,
                col=col,
            )

    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        width=300 * n_cols,
        title_text="Model Comparison Dashboard",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update y-axes
    for i in range(1, n_metrics + 1):
        fig.update_yaxes(
            title_text="Score", row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1
        )

    fig.show()
