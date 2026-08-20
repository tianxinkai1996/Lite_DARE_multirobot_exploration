from multi_test_driver import make_seed, parse_map_selection


def test_specified_maps():
    assert parse_map_selection("5", None) == [5]
    assert parse_map_selection("5,2,5", None) == [2, 5]


def test_all_maps_with_count():
    assert parse_map_selection("all", 3) == [0, 1, 2]


def test_seed_changes_with_map_and_sample():
    assert make_seed(0, 0, "compressed") != make_seed(0, 1, "compressed")
    assert make_seed(0, 0, "compressed") != make_seed(1, 0, "compressed")