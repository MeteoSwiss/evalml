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

    # cmap has n_levs-1 colors inside
    assert cmap.N == 2
    # name is stem
    assert cmap.name == "test_colormap"
    # cmap colors
    assert np.allclose(cmap(0), (40 / 255, 50 / 255, 60 / 255, 1.0))
    assert np.allclose(cmap(1), (70 / 255, 80 / 255, 90 / 255, 1.0))
    # under/over colors
    assert np.allclose(cmap(-9999), (10 / 255, 20 / 255, 30 / 255, 1.0))
    assert np.allclose(cmap(9999), (100 / 255, 110 / 255, 120 / 255, 1.0))
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
