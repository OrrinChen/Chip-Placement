from chip.openroad_bridge import OpenROADBridge


def test_openroad_bridge_reports_unavailable_when_binary_missing():
    bridge = OpenROADBridge(binary_name="definitely-missing-openroad")

    assert bridge.is_available() is False
