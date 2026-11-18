"""One implementation of the screening helpers both Bethe-Heitler models use.

Pair production (:mod:`pyradmc.data.pair`) and bremsstrahlung
(:mod:`pyradmc.data.bethe_heitler`) are the same matrix element read two ways, and
they share the fine-structure constant, the Davies-Bethe-Maximon Coulomb correction
and Tsai's radiation logarithms. Those live once, in :mod:`pyradmc.data.screening`;
a medium must not be screened by two slightly different numbers.

The literal pins are the values both modules returned while each still carried its
own copy (hex floats, so the comparison is exact): unifying the helpers moved no
number of either model.
"""

from __future__ import annotations

import pytest

from pyradmc.data import bethe_heitler, pair, screening
from pyradmc.data.materials import MATERIALS


def _composition(name: str) -> tuple[tuple[int, float], ...]:
    return next(m for m in MATERIALS if m.name == name).composition


class TestOneImplementation:
    @pytest.mark.parametrize("module", [bethe_heitler, pair])
    def test_helpers_are_the_shared_objects(self, module: object) -> None:
        assert module.coulomb_correction is screening.coulomb_correction  # type: ignore[attr-defined]
        assert (
            module.FINE_STRUCTURE_CONSTANT  # type: ignore[attr-defined]
            is screening.FINE_STRUCTURE_CONSTANT
        )

    def test_radiation_logarithms_are_shared(self) -> None:
        assert pair.radiation_logarithms is screening.radiation_logarithms

    @pytest.mark.parametrize("module", [bethe_heitler, pair])
    def test_no_private_copy_survives(self, module: object) -> None:
        """Neither module keeps a table or fit constant of its own."""
        for name in (
            "_TSAI_LIGHT_ELEMENTS",
            "_LOW_Z_RADIATION_LOGARITHMS",
            "_TSAI_PHI1_COMPLETE",
            "_TSAI_PSI1_COMPLETE",
        ):
            assert not hasattr(module, name), name

    def test_fit_constants(self) -> None:
        assert screening.TSAI_PHI1_COMPLETE == 20.863
        assert screening.TSAI_PSI1_COMPLETE == 28.340
        # The bremsstrahlung shape reads the same two constants.
        assert bethe_heitler._PHI1_COMPLETE is screening.TSAI_PHI1_COMPLETE
        assert bethe_heitler._PSI1_COMPLETE is screening.TSAI_PSI1_COMPLETE


class TestNoNumberMoved:
    """Exact values from before the unification, per model and material."""

    def test_fine_structure_constant(self) -> None:
        assert screening.FINE_STRUCTURE_CONSTANT.hex() == "0x1.de3d42a1ed29dp-8"

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (
                "water",
                (
                    "0x1.364d9364d9365p-3",
                    "0x1.58b429f11e01fp-1",
                    "0x1.4c17a00004bd7p+0",
                    "0x1.056f1b5bedf76p-1",
                    "0x1.51cb453b9536dp-2",
                    "0x1.d4f14ad151a38p+1",
                ),
            ),
            (
                "air",
                (
                    "0x1.16488bf64c2d4p-3",
                    "0x1.58645b4c48e8ap-1",
                    "0x1.53418a8c49cdcp+0",
                    "0x1.0650899fa7058p-1",
                    "0x1.101f67a38245fp-2",
                    "0x1.d63ba9df214ebp+1",
                ),
            ),
            (
                "cortical_bone",
                (
                    "0x1.92430010eb5b4p-4",
                    "0x1.afa897490a8cap-1",
                    "0x1.7b585aac0dfd5p+0",
                    "0x1.be6efc7a74757p-2",
                    "0x1.fa6e2e7ad08c4p-3",
                    "0x1.53b899ff5f01cp+2",
                ),
            ),
            (
                "tungsten",
                (
                    "0x1.bacf914c1bacfp-7",
                    "0x1.b768226983128p+0",
                    "0x1.6f47bc950e427p+1",
                    "0x1.e7cff05e5a83cp-3",
                    "0x1.d0c466b6cb546p-5",
                    "0x1.dc969d4c6bd62p+4",
                ),
            ),
        ],
    )
    def test_bremsstrahlung_screening(self, name: str, expected: tuple[str, ...]) -> None:
        got = bethe_heitler.screening_parameters(_composition(name))
        assert tuple(float(v).hex() for v in got) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("water", ("0x1.1a9315a5a25d2p+6", "0x1.4f5d0e7e3147ap+2", "0x1.5157736c6ae0dp+2")),
            ("air", ("0x1.16f80c6702142p+6", "0x1.55f0472639504p+2", "0x1.57e430133c93cp+2")),
            (
                "cortical_bone",
                ("0x1.e10c79997d7c9p+5", "0x1.a1db8280e4d69p+2", "0x1.a89ae171eceb9p+2"),
            ),
            ("tungsten", ("0x1.032677b220160p+5", "0x1.6f47bc950e426p+3", "0x1.b768226983126p+3")),
        ],
    )
    def test_pair_screening(self, name: str, expected: tuple[str, ...]) -> None:
        got = pair.pair_screening_parameters(_composition(name))
        assert tuple(float(v).hex() for v in got) == expected
