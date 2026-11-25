"""Exercise real astromodels and Fermipy source definitions without ScienceTools.

The recording likelihood backend isolates source construction and updates. A real
LAT integration test is still needed to validate exposure/PSF folding.
"""

import copy
import importlib
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from astromodels import (
    Constant,
    ExtendedSource,
    GalpropMap,
    Isotropic_on_sphere,
    Model,
    PointSource,
    Powerlaw,
    TabulatedSpectrum,
    clone_model,
)

FermipyModel = pytest.importorskip("fermipy.roi_model").Model
plugin = importlib.import_module("threeML.plugins.FermipyLike")


class RecordingGTAnalysis:
    """Record updates using Fermipy's real source-configuration parser."""

    def __init__(self, configuration):
        self.config = copy.deepcopy(configuration)
        self.events = []
        self.spectra = {}
        self.energy = {}
        definitions = list(configuration["model"]["sources"])
        if "galdiff" in configuration["model"]:
            definitions.append(
                dict(
                    name="galdiff",
                    SpatialModel="MapCubeFunction",
                    Spatial_Filename=configuration["model"]["galdiff"],
                )
            )
        if "isodiff" in configuration["model"]:
            definitions.append(
                dict(
                    name="isodiff",
                    SpatialModel="ConstantValue",
                    Spectrum_Filename=configuration["model"]["isodiff"],
                )
            )
        self.sources = {
            d["name"]: FermipyModel.create_from_dict(d) for d in definitions
        }
        for name, source in self.sources.items():
            if source["SpatialModel"] == "ConstantValue" and name != "isodiff":
                table = np.loadtxt(source["Spectrum_Filename"])
                self.energy[name] = np.log10(table[:, 0])
                self.spectra[name] = table[:, 1]
        self.roi = SimpleNamespace(
            sources=list(self.sources.values()),
            get_sources=lambda: list(self.sources.values()),
            get_source_by_name=lambda name: self.sources[name],
        )
        self.like = SimpleNamespace(
            logLike=SimpleNamespace(value=self.loglike), total_nobs=lambda: 0
        )

    def setup(self):
        pass

    def set_source_spectrum(self, name, spectrum_type, update_source=True):
        self.sources[name]["SpectrumType"] = spectrum_type
        self.sources[name].set_spectral_pars(
            {
                "Normalization": dict(
                    name="Normalization",
                    value=1.0,
                    scale=1.0,
                    min=1e-3,
                    max=1e3,
                    free=False,
                    error=np.nan,
                )
            }
        )
        self.energy[name] = np.array([2.0, 3.0, 4.0])
        self.spectra[name] = np.ones(3)

    def get_source_dnde(self, name):
        return self.energy[name], self.spectra[name]

    def set_source_dnde(self, name, values, update_source=True):
        assert len(values) == len(self.energy[name])
        assert np.all(np.isfinite(values))
        self.spectra[name] = np.array(values)
        self.events.append(("spectrum", name, update_source))

    def free_source(self, name, free=True):
        for p in self.sources[name].spectral_pars.values():
            p["free"] = free

    def free_norm(self, name):
        key = (
            "Prefactor"
            if self.sources[name]["SpectrumType"] == "PowerLaw"
            else "Normalization"
        )
        self.sources[name].spectral_pars[key]["free"] = True

    def get_free_source_params(self, name):
        return [k for k, v in self.sources[name].spectral_pars.items() if v["free"]]

    def _get_param(self, name, parameter):
        return self.sources[name].spectral_pars[parameter]

    def set_parameter(self, name, parameter, value, scale=1, update_source=True):
        self.sources[name].spectral_pars[parameter]["value"] = value
        self.sources[name].spectral_pars[parameter]["scale"] = scale

    def set_norm(self, name, value, update_source=True):
        key = (
            "Prefactor"
            if self.sources[name]["SpectrumType"] == "PowerLaw"
            else "Normalization"
        )
        self.set_parameter(name, key, value, update_source=update_source)
        self.events.append(("norm", name, update_source))

    def set_norm_bounds(self, name, bounds):
        key = (
            "Prefactor"
            if self.sources[name]["SpectrumType"] == "PowerLaw"
            else "Normalization"
        )
        self.sources[name].spectral_pars[key].update(min=bounds[0], max=bounds[1])

    def update_source(self, name, paramsonly=False):
        self.events.append(("metadata", name, paramsonly))

    def loglike(self):
        # A deterministic stand-in, useful for checking that changed parameters
        # reach the backend; this is not a ScienceTools likelihood calculation.
        value = sum(np.sum(s) for s in self.spectra.values())
        for source in self.sources.values():
            value += source.get_norm()
        return -float(value)


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.setattr(plugin, "_expensive_imports_hook", lambda: None)
    monkeypatch.setattr(plugin, "GTAnalysis", RecordingGTAnalysis, raising=False)
    monkeypatch.setattr(
        plugin,
        "findGalacticTemplate",
        lambda *args: str(tmp_path / "gal.fits"),
        raising=False,
    )
    monkeypatch.setattr(
        plugin,
        "findIsotropicTemplate",
        lambda *args: str(tmp_path / "iso.txt"),
        raising=False,
    )
    for name in ["events.fits", "spacecraft.fits"]:
        (tmp_path / name).touch()
    config = dict(
        data=dict(
            evfile=str(tmp_path / "events.fits"),
            scfile=str(tmp_path / "spacecraft.fits"),
        ),
        binning=dict(roiwidth=10.0, binsz=1.0, binsperdec=8),
        gtlike=dict(edisp=False),
        selection=dict(
            emin=100.0,
            emax=10000.0,
            zmax=90.0,
            evclass=128,
            evtype=3,
            filter="DATA_QUAL>0",
            ra=0.0,
            dec=0.0,
        ),
    )
    return config, tmp_path


def point_model():
    return Model(
        PointSource(
            "catalog",
            ra=0.0,
            dec=0.0,
            spectral_shape=Powerlaw(K=1e-8, piv=1e5, index=-2),
        )
    )


def map_source(path, name="ic"):
    h = fits.Header()
    for i, (ctype, crval, cdelt, crpix) in enumerate(
        [
            ("GLON-CAR", 0.0, 90.0, 1.0),
            ("GLAT-CAR", -60.0, 60.0, 1.0),
            ("Energy", 2.0, 1.0, 1.0),
        ],
        1,
    ):
        for key, value in [
            ("CTYPE", ctype),
            ("CRVAL", crval),
            ("CDELT", cdelt),
            ("CRPIX", crpix),
        ]:
            h[f"{key}{i}"] = value
    fits.HDUList(
        [
            fits.PrimaryHDU(np.ones((3, 3, 4)), h),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="Energy",
                        format="D",
                        unit="MeV",
                        array=[100.0, 1000.0, 10000.0],
                    )
                ],
                name="ENERGIES",
            ),
        ]
    ).writeto(path, overwrite=True)
    shape = GalpropMap()
    shape.load_file(path)
    constant = Constant(k=1.0)
    constant.k.fix = True
    return ExtendedSource(name, shape, constant)


def attach(environment, model, **kwargs):
    config, path = environment
    lat = plugin.FermipyLike("LAT_test", config, **kwargs)
    lat.configuration["fileio"]["outdir"] = str(path / "output")
    lat.set_model(model)
    return lat


def test_defaults_and_input_unchanged(environment):
    config, _ = environment
    original = copy.deepcopy(config)
    lat = attach(environment, point_model())
    assert config == original
    assert set(lat._gta.sources) == {"catalog", "galdiff", "isodiff"}
    assert set(lat.nuisance_parameters) == {
        "LAT_test_galdiff_Prefactor",
        "LAT_test_isodiff_Normalization",
    }
    assert lat._split_nuisance_parameter("LAT_test_isodiff_Normalization") == (
        "isodiff",
        "Normalization",
    )
    lat.get_log_like()


def test_mapcube_normalization_and_no_duplicate_nuisance(environment):
    _, path = environment
    source = map_source(path / "map.fits", "custom_galdiff")
    source.spatial_shape.K = 2.0
    model = point_model()
    model.add_source(source)
    lat = attach(environment, model, galactic_diffuse=[source.name])
    assert set(lat._gta.sources) == {"catalog", "custom_galdiff", "isodiff"}
    assert set(lat.nuisance_parameters) == {"LAT_test_isodiff_Normalization"}
    spectral = lat._gta.sources[source.name].spectral_pars
    assert spectral["Index"]["value"] == 0
    assert lat._gta.sources[source.name].get_norm() == 2
    assert not lat._gta.get_free_source_params(source.name)
    first = lat.get_log_like()
    source.spatial_shape.K = 0.0
    second = lat.get_log_like()
    assert lat._gta.sources[source.name].get_norm() == 0
    assert first != second
    source.spatial_shape.K = 25.0
    lat.get_log_like()
    assert spectral["Prefactor"]["max"] >= 25
    assert lat._gta.sources[source.name].get_norm() == 25


@pytest.mark.parametrize("tabulated", [False, True])
def test_custom_isotropic_live_spectrum(environment, tabulated):
    if tabulated:
        spectrum = TabulatedSpectrum()
        spectrum.set_table(
            np.array([100.0, 1000.0, 10000.0]) * u.MeV,
            np.array([1e-7, 1e-9, 1e-11]) / (u.MeV * u.s * u.cm**2 * u.sr),
            4 * np.pi * u.sr,
        )
    else:
        spectrum = Powerlaw(K=4 * np.pi * 1e-10, piv=1e5, index=-2.0)
    source = ExtendedSource("custom_isodiff", Isotropic_on_sphere(), spectrum)
    lat = attach(
        environment,
        Model(source),
        galactic_diffuse=None,
        isotropic_diffuse=[source.name],
    )
    assert set(lat._gta.sources) == {source.name}
    assert not lat.nuisance_parameters
    assert lat._pts_energies is None
    energy = 10 ** lat._gta.energy[source.name]
    expected = 1e-7 * (energy / 100) ** -2
    np.testing.assert_allclose(lat._gta.spectra[source.name], expected)
    spectrum.K.value *= 2
    lat.get_log_like()
    np.testing.assert_allclose(lat._gta.spectra[source.name], 2 * expected)
    if not tabulated:
        spectrum.index = -2.5
        lat.get_log_like()
        np.testing.assert_allclose(
            lat._gta.spectra[source.name], 2e-7 * (energy / 100) ** -2.5
        )


def test_mixed_sources_and_selective_diagnostics(environment):
    _, path = environment
    gal = map_source(path / "map.fits")
    iso = ExtendedSource("iso_custom", Isotropic_on_sphere(), Powerlaw(K=1e-9))
    model = point_model()
    model.add_source(gal)
    model.add_source(iso)
    lat = attach(
        environment,
        model,
        galactic_diffuse=["ic"],
        isotropic_diffuse=["iso_custom"],
        skip_source_diagnostics=["iso_custom"],
    )
    lat._gta.events.clear()
    lat._update_model_in_fermipy(update_dictionary=True, force_update=True)
    assert ("spectrum", "iso_custom", False) in lat._gta.events
    assert ("spectrum", "catalog", True) in lat._gta.events
    assert ("norm", "ic", True) in lat._gta.events
    assert ("metadata", "iso_custom", True) in lat._gta.events
    lat = attach(
        environment,
        model,
        galactic_diffuse=["ic"],
        isotropic_diffuse=None,
        exclude_sources=["iso_custom"],
        skip_source_diagnostics=True,
    )
    assert "iso_custom" not in lat._gta.sources
    assert "iso_custom" in model.sources


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(galactic_diffuse=["missing"]),
        dict(isotropic_diffuse=["catalog"]),
        dict(galactic_diffuse=["catalog"]),
        dict(skip_source_diagnostics=["missing"]),
    ],
)
def test_unknown_or_incompatible_source(environment, kwargs):
    with pytest.raises(ValueError):
        attach(environment, point_model(), **kwargs)


def test_invalid_mapcube_and_round_trip(environment):
    _, path = environment
    source = map_source(path / "map.fits")
    model = clone_model(Model(source))
    lat = attach(environment, model, galactic_diffuse=["ic"])
    assert lat._gta.sources["ic"].get_norm() == 1
    source.spectrum.main.Constant.k.free = True
    with pytest.raises(ValueError, match="fixed Constant"):
        attach(environment, Model(source), galactic_diffuse=["ic"])


def test_energy_coverage(environment):
    spectrum = TabulatedSpectrum()
    spectrum.set_table(
        np.array([200.0, 1000.0]) * u.MeV,
        np.array([1e-7, 1e-9]) / (u.MeV * u.s * u.cm**2),
    )
    source = ExtendedSource("iso", Isotropic_on_sphere(), spectrum)
    with pytest.raises(ValueError, match="must cover"):
        attach(environment, Model(source), isotropic_diffuse=["iso"])


@pytest.mark.parametrize(
    "galactic,isotropic", [("default", None), (None, "default"), (None, None)]
)
def test_independent_default_switches(environment, galactic, isotropic):
    lat = attach(
        environment,
        point_model(),
        galactic_diffuse=galactic,
        isotropic_diffuse=isotropic,
    )
    expected = {"catalog"}
    if galactic == "default":
        expected.add("galdiff")
    if isotropic == "default":
        expected.add("isodiff")
    assert set(lat._gta.sources) == expected


@pytest.mark.parametrize("selection", [True, False, []])
def test_diagnostic_boolean_compatibility(environment, selection):
    lat = attach(environment, point_model(), skip_source_diagnostics=selection)
    lat._gta.events.clear()
    lat._update_model_in_fermipy(update_dictionary=True, force_update=True)
    assert ("spectrum", "catalog", selection is not True) in lat._gta.events
    assert (("metadata", "catalog", True) in lat._gta.events) == (selection is True)


def test_conflicting_selections(environment):
    _, path = environment
    model = Model(map_source(path / "map.fits"))
    with pytest.raises(ValueError, match="cannot also be excluded"):
        attach(environment, model, galactic_diffuse=["ic"], exclude_sources=["ic"])
    with pytest.raises(ValueError, match="both Galactic and isotropic"):
        attach(environment, model, galactic_diffuse=["ic"], isotropic_diffuse=["ic"])
    model = Model(PointSource("galdiff", ra=0.0, dec=0.0, spectral_shape=Powerlaw()))
    with pytest.raises(ValueError, match="conflicts"):
        attach(environment, model)


def test_automatic_cache_identity(environment, monkeypatch):
    config, path = environment
    monkeypatch.chdir(path)
    model = Model(map_source(path / "first.fits"))
    lat = plugin.FermipyLike("LAT", config, galactic_diffuse=["ic"])
    lat.set_model(model)
    first = lat.configuration["fileio"]["outdir"]
    lat.set_model(model)
    assert lat.configuration["fileio"]["outdir"] == first
    other = plugin.FermipyLike("LAT", config, galactic_diffuse=["ic"])
    other.set_model(Model(map_source(path / "second.fits")))
    assert other.configuration["fileio"]["outdir"] != first


def test_multiple_galprop_components(environment):
    _, path = environment
    ic = map_source(path / "ic.fits", "ic")
    pi0 = map_source(path / "pi0.fits", "pi0")
    lat = attach(
        environment,
        Model(ic, pi0),
        galactic_diffuse=["ic", "pi0"],
        isotropic_diffuse=None,
    )
    assert lat._pts_energies is None
    ic.spatial_shape.K = 2
    pi0.spatial_shape.K = 3
    lat.get_log_like()
    assert lat._gta.sources["ic"].get_norm() == 2
    assert lat._gta.sources["pi0"].get_norm() == 3


def test_component_override_rejected(environment):
    config, path = environment
    config["components"] = [{"model": {"galdiff": "another.fits"}}]
    with pytest.raises(ValueError, match="component-specific"):
        attach(environment, point_model(), galactic_diffuse=None)


def test_tabulated_energy_dispersion_margin(environment):
    config, _ = environment
    config["gtlike"] = {"edisp": True, "edisp_bins": 1}
    spectrum = TabulatedSpectrum()
    spectrum.set_table(
        np.array([100.0, 10000.0]) * u.MeV,
        np.array([1e-7, 1e-11]) / (u.MeV * u.s * u.cm**2),
    )
    source = ExtendedSource("iso", Isotropic_on_sphere(), spectrum)
    with pytest.raises(ValueError, match="energy-dispersion margin"):
        attach(environment, Model(source), isotropic_diffuse=["iso"])


@pytest.mark.parametrize("as_string", [False, True])
def test_native_isotropic_file(environment, monkeypatch, as_string):
    _, path = environment
    filename = path / "custom isotropic.dat"
    table = np.array([[10.0, 1e-5], [1000.0, 1e-9], [100000.0, 1e-13]])
    np.savetxt(filename, table)

    def unexpected_lookup(*args):
        raise AssertionError("The standard isotropic template should not be loaded")

    monkeypatch.setattr(plugin, "findIsotropicTemplate", unexpected_lookup)
    model = point_model()
    model.add_source(map_source(path / "ic.fits"))
    lat = attach(
        environment,
        model,
        galactic_diffuse=["ic"],
        isotropic_spectrum=str(filename) if as_string else filename,
    )
    assert set(lat._gta.sources) == {"catalog", "ic", "isodiff"}
    staged = lat._gta.config["model"]["isodiff"]
    assert staged.endswith(".txt")
    np.testing.assert_array_equal(np.loadtxt(staged), table)
    np.testing.assert_array_equal(np.loadtxt(filename), table)
    assert set(lat.nuisance_parameters) == {"LAT_test_isodiff_Normalization"}
    initial = lat.get_log_like()
    lat.nuisance_parameters["LAT_test_isodiff_Normalization"].value = 2.0
    assert lat.get_log_like() != initial
    assert lat._gta.sources["isodiff"].get_norm() == 2.0
    assert not any(e[:2] == ("spectrum", "isodiff") for e in lat._gta.events)


@pytest.mark.parametrize("selection", [None, ["EGB"]])
def test_native_isotropic_conflicting_modes(environment, selection):
    config, _ = environment
    with pytest.raises(ValueError, match="isotropic_spectrum requires"):
        plugin.FermipyLike(
            "LAT", config, isotropic_diffuse=selection, isotropic_spectrum="unused.txt"
        )


@pytest.mark.parametrize(
    "contents",
    ["100 1e-8\n", "100 1e-8\n10 1e-7\n", "10 0\n100 1e-8\n", "10 nan\n100 1e-8\n"],
)
def test_native_isotropic_invalid_table(environment, contents):
    config, path = environment
    filename = path / "invalid.txt"
    filename.write_text(contents)
    with pytest.raises(ValueError, match="isotropic spectrum needs"):
        plugin.FermipyLike("LAT", config, isotropic_spectrum=filename)


def test_native_isotropic_missing_file(environment):
    config, path = environment
    with pytest.raises(FileNotFoundError):
        plugin.FermipyLike("LAT", config, isotropic_spectrum=path / "missing.txt")


def test_native_isotropic_cache_and_overrides(environment):
    config, path = environment
    filename = path / "isotropic.txt"
    np.savetxt(filename, [[10, 1e-5], [100000, 1e-13]])
    lat = plugin.FermipyLike("LAT", config, isotropic_spectrum=filename)
    lat.set_model(point_model())
    first = lat.configuration["fileio"]["outdir"]
    np.savetxt(filename, [[10, 2e-5], [100000, 2e-13]])
    lat.set_model(point_model())
    assert lat.configuration["fileio"]["outdir"] != first
    lat.configuration["fileio"]["outdir"] = str(path / "manual")
    lat.set_model(point_model())
    assert lat.configuration["fileio"]["outdir"] == str(path / "manual")
    lat.configuration["components"] = [{"model": {"isodiff": "other.txt"}}]
    with pytest.raises(ValueError, match="component-specific isodiff"):
        lat.set_model(point_model())
