from __future__ import annotations

import unittest

from mergingmap.ablation_profiles import CORE_ABLATION_PROFILES, parse_profile_keys, resolve_ablation_profile
from classes.multi_robot.dare_test_profiles import resolve_dare_test_profile
from mergingmap.run_paper_ablation import scenario_id


class PaperAblationProfileTests(unittest.TestCase):
    def test_methodology_layers_match_profile_names(self) -> None:
        map_only = resolve_ablation_profile("map_only")
        map_region = resolve_ablation_profile("map_region")
        reservation = resolve_ablation_profile("map_reservation")
        full = resolve_ablation_profile("full")
        self.assertEqual(map_only.coordination_mode, "ghost")
        self.assertFalse(map_only.enable_dynamic_regions)
        self.assertTrue(map_region.enable_dynamic_regions)
        self.assertEqual(map_region.coordination_mode, "ghost")
        self.assertEqual(reservation.coordination_mode, "collision")
        self.assertFalse(reservation.enable_dynamic_regions)
        self.assertEqual(full.coordination_mode, "collision_deadlock")
        self.assertTrue(full.enable_dynamic_regions)
        self.assertTrue(full.enable_initial_direction)

    def test_legacy_keys_resolve_to_canonical_profiles(self) -> None:
        self.assertEqual(resolve_ablation_profile("mm").key, "map_only")
        self.assertEqual(resolve_ablation_profile("mm_region").key, "map_region")
        self.assertEqual(resolve_ablation_profile("mm_collision").key, "map_reservation")
        self.assertEqual(resolve_ablation_profile("mm_collision_deadlock").key, "full")

    def test_parser_canonicalises_and_rejects_alias_duplicates(self) -> None:
        self.assertEqual(
            parse_profile_keys("mm_collision,mm"),
            ("map_reservation", "map_only"),
        )
        with self.assertRaises(ValueError):
            parse_profile_keys("full,mm_collision_deadlock")

    def test_all_canonical_profiles_enable_map_layer(self) -> None:
        self.assertEqual(set(CORE_ABLATION_PROFILES), {"map_only", "map_region", "map_reservation", "full"})
        self.assertTrue(all(p.enable_map_merging for p in CORE_ABLATION_PROFILES.values()))

    def test_scenario_identifier_separates_communication_treatments(self) -> None:
        raw = scenario_id(1, 0, 4, 123, "raw")
        compressed = scenario_id(1, 0, 4, 123, "compressed")
        self.assertNotEqual(raw, compressed)
        self.assertIn("mapcomm_raw", raw)
        self.assertIn("mapcomm_compressed", compressed)

    def test_original_dare_profile_disables_all_added_wrappers(self) -> None:
        profile = resolve_dare_test_profile("original_dare")
        self.assertEqual(profile.coordination_mode, "ghost")
        self.assertFalse(profile.enable_messages)
        self.assertFalse(profile.enable_reservations)
        self.assertFalse(profile.enable_coverage_exchange)


if __name__ == "__main__":
    unittest.main()

