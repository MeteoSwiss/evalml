import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


from plotting import colormap_loader, colormap_defaults


def test_load_valid_colormap(monkeypatch, tmp_path):
    file = tmp_path / "test_colormap.ct"
    file.write_text(
        "; test colormap\n3\n0 1 2\n10 20 30\n40 50 60\n70 80 90\n100 110 120\n"
    )
    monkeypatch.setattr(colormap_loader, "BASE_DIR", tmp_path)

    result = colormap_loader.load_ncl_colormap("test_colormap.ct")
    assert isinstance(result["cmap"], ListedColormap)
    assert isinstance(result["norm"], BoundaryNorm)

    cmap = result["cmap"]
    norm = result["norm"]
    bounds = result["bounds"]

    # cmap holds the n_levs-1 inner colors, under/over excluded
    assert cmap.N == 2
    assert np.allclose(cmap(0), (*[x / 255 for x in (40, 50, 60)], 1.0))
    assert np.allclose(cmap(1), (*[x / 255 for x in (70, 80, 90)], 1.0))
    # under/over are set separately
    assert np.allclose(cmap.get_under(), (*[x / 255 for x in (10, 20, 30)], 1.0))
    assert np.allclose(cmap.get_over(), (*[x / 255 for x in (100, 110, 120)], 1.0))
    # bounds
    assert np.allclose(norm.boundaries, [0, 1, 2])
    assert np.allclose(bounds, [0, 1, 2])


def test_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(colormap_loader, "BASE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        colormap_loader.load_ncl_colormap("does_not_exist.ct")


def test_invalid_first_line(monkeypatch, tmp_path):
    file = tmp_path / "bad.ct"
    file.write_text("not_a_number\n")
    monkeypatch.setattr(colormap_loader, "BASE_DIR", tmp_path)
    with pytest.raises(ValueError):
        colormap_loader.load_ncl_colormap("bad.ct")


def test_wrong_bounds(monkeypatch, tmp_path):
    file = tmp_path / "bad_bounds.ct"
    file.write_text("2\n1 2 3\n10 20 30\n40 50 60\n")
    monkeypatch.setattr(colormap_loader, "BASE_DIR", tmp_path)
    with pytest.raises(ValueError):
        colormap_loader.load_ncl_colormap("bad_bounds.ct")


def test_wrong_rgb_count(monkeypatch, tmp_path):
    file = tmp_path / "bad_rgb.ct"
    file.write_text("2\n1 2\n10 20 30\n")
    monkeypatch.setattr(colormap_loader, "BASE_DIR", tmp_path)
    with pytest.raises(ValueError):
        colormap_loader.load_ncl_colormap("bad_rgb.ct")


@pytest.mark.parametrize("field, var", colormap_defaults.CMAP_DEFAULTS.items())
def test_cmap_defaults_smoke(field, var):
    """Smoke test: can we use every entry in CMAP_DEFAULTS to plot data?"""
    cmap = var.get("cmap", None)
    norm = var.get("norm", None)
    vmin = var.get("vmin", None)
    vmax = var.get("vmax", None)

    # make some synthetic data
    data = np.linspace(0, 1, 100).reshape(10, 10)

    # just try plotting
    fig, ax = plt.subplots()
    if norm is not None:
        ax.imshow(data, cmap=cmap, norm=norm)
    else:
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.close(fig)
