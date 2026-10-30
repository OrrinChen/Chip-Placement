from chip.orfs_export import ORFSDesignExporter


def test_exporter_builds_expected_output_paths(tmp_path):
    exporter = ORFSDesignExporter(work_root=tmp_path / "orfs-work", export_root=tmp_path / "orfs-export")

    bundle = exporter.bundle_paths("gcd")

    assert bundle.design_name == "gcd"
    assert bundle.odb_path == tmp_path / "orfs-work" / "results" / "nangate45" / "gcd" / "base" / "3_place.odb"
    assert bundle.output_dir == tmp_path / "orfs-export" / "nangate45" / "gcd"
    assert bundle.def_path == bundle.output_dir / "3_place.def"


def test_exporter_builds_docker_openroad_command(tmp_path):
    exporter = ORFSDesignExporter(
        work_root=tmp_path / "orfs-work",
        export_root=tmp_path / "orfs-export",
        docker_image="openroad/orfs",
    )

    bundle = exporter.bundle_paths("gcd")
    script_path = tmp_path / "orfs-export" / "export.tcl"
    command = exporter.build_export_command(bundle, script_path)

    assert command[0] == "docker"
    assert "openroad/orfs" in command
    assert command[-1] == "/workspace/orfs-export/export.tcl"


def test_exporter_reports_missing_odb_before_running(tmp_path):
    exporter = ORFSDesignExporter(work_root=tmp_path / "orfs-work", export_root=tmp_path / "orfs-export")

    bundle = exporter.bundle_paths("ibex")
    result = exporter.export_design("ibex")

    assert result["status"] == "missing_odb"
    assert result["bundle"] == bundle
