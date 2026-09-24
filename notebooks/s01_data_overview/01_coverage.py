import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    from prophet_ab import paths
    from datapoints_figures import (
        set_manuscript_style,
        DATAPOINTS_COLORS,
        SINGLE_COL_WIDTH,
        FULL_WIDTH,
        WHITE_TO_PURPLE,
        FONT_SIZE_TICK,
        wrap_title,
    )

    set_manuscript_style()
    return (
        FONT_SIZE_TICK,
        FULL_WIDTH,
        WHITE_TO_PURPLE,
        mo,
        paths,
        pd,
        plt,
        wrap_title,
    )


@app.cell
def _(mo):
    mo.md("""
    # s01 — Data overview

    Sanity-check the stage-01 normalized parquets. Reports assay coverage
    across N3 / N4 (excluding controls) and per-antibody replicate counts.

    Outputs declared in `manifest.yaml`:

    - `reports/figures/s01_assay_coverage.png`
    - `reports/tables/s01_replicate_counts.csv`
    """)
    return


@app.cell
def _(paths, pd):
    long = pd.read_parquet(paths.S01 / "n3n4_long.parquet")
    long.head()
    return (long,)


@app.cell
def _(long):
    summary = (
        long.groupby("kind")
        .agg(
            n_antibodies=("antibody_name", "nunique"),
            n_measurements=("value", "size"),
            n_plates=("plateid", "nunique"),
        )
    )
    summary
    return


@app.cell
def _(long):
    coverage = (
        long.assign(study=lambda d: d["kind"] + d["is_control"].map({True: " (ctrl)", False: ""}))
        .groupby(["study", "assay"])
        .size()
        .unstack(fill_value=0)
    )
    coverage
    return (coverage,)


@app.cell
def _(long):
    replicate_counts = (
        long.groupby(["antibody_name", "kind", "is_control", "value_col"])
        .size()
        .rename("n_replicates")
        .reset_index()
    )
    replicate_counts.head()
    return (replicate_counts,)


@app.cell
def _(
    FONT_SIZE_TICK,
    FULL_WIDTH,
    WHITE_TO_PURPLE,
    coverage,
    paths,
    plt,
    wrap_title,
):
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.6), layout="constrained")
    im = ax.imshow(coverage.values, aspect="auto", cmap=WHITE_TO_PURPLE)
    ax.set_xticks(range(len(coverage.columns)))
    ax.set_xticklabels(coverage.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(coverage.index)))
    ax.set_yticklabels(coverage.index)
    ax.set_xlabel("assay")
    ax.set_ylabel("study group")
    ax.set_title(wrap_title("Measurement counts by group × assay", FULL_WIDTH))
    for i in range(coverage.shape[0]):
        for j in range(coverage.shape[1]):
            ax.text(j, i, int(coverage.values[i, j]), ha="center", va="center",
                    color="white" if coverage.values[i, j] < coverage.values.max() / 2 else "black",
                    fontsize=FONT_SIZE_TICK - 2)
    fig.colorbar(im, ax=ax, label="n measurements")

    _out = paths.FIGURES / "s01_assay_coverage.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"wrote {_out.relative_to(paths.REPO_ROOT)}")
    fig
    return


@app.cell
def _(paths, replicate_counts):
    _out = paths.TABLES / "s01_replicate_counts.csv"
    _out.parent.mkdir(parents=True, exist_ok=True)
    replicate_counts.to_csv(_out, index=False)
    print(f"wrote {_out.relative_to(paths.REPO_ROOT)}  rows={len(replicate_counts)}")
    return


if __name__ == "__main__":
    app.run()
