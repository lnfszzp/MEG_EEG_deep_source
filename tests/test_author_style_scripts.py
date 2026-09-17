import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    PROJECT_ROOT / "作者风格版" / "1-OASTER真实ERP.py",
    PROJECT_ROOT / "作者风格版" / "2-OASTER仿真.py",
    PROJECT_ROOT / "作者风格版" / "3-ERP四类任务验证.py",
    PROJECT_ROOT / "作者风格版" / "4-DBS频谱定位.py",
    PROJECT_ROOT / "作者风格版" / "5-左指运动双链定位.py",
    PROJECT_ROOT / "作者风格版" / "6-左指运动结果俯视图.py",
)


def test_author_style_scripts_are_linear_cell_scripts():
    forbidden = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

    for script in SCRIPTS:
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script))

        minimum_cells = 3 if script.name.startswith("6-") else 10
        assert source.count("#%%") + source.count("# %%") >= minimum_cells
        assert not any(isinstance(node, forbidden) for node in ast.walk(tree))


def test_four_paradigm_reuses_validated_ds006035_preprocessing():
    source = SCRIPTS[2].read_text(encoding="utf-8")
    events = source.index("ds_events = np.zeros")
    interpolate = source.index("real_pipeline.interpolate_stimulation_artifacts")
    filtering = source.index("ds_raw.filter")

    assert events < interpolate < filtering
    assert "tmin=-0.002" in source[interpolate:filtering]
    assert "tmax=0.008" in source[interpolate:filtering]
    assert "protected.whitening_matrix(\n        ds_eeg_noise_check" in source
    assert "protected.whitening_matrix(\n        ds_mag_noise_check" in source


def test_finger_dual_chain_has_safe_pairing_and_both_inverse_chains():
    source = SCRIPTS[4].read_text(encoding="utf-8")
    interpolate = source.index("real_pipeline.interpolate_stimulation_artifacts")
    filtering = source.index("raw.filter", interpolate)
    forward = source.index("mne.make_forward_solution")
    analysis_vertices = source.index("source_vertices = [", forward)
    oaster_inverse = source.index("oaster.reconstruct_evoked_oaster_from_whitened")
    roi_read = source.index("mne.read_labels_from_annot")
    last_inverse = source.index("mne.beamformer.apply_dics_csd")

    assert "pending_stimulus" in source
    assert "minimum_rt = 0.10" in source
    assert "maximum_rt = 0.90" in source
    assert "response_baseline = (-1.50, -1.00)" in source
    assert "filter_high = 40.0" in source
    assert "oaster_max_templates = 1" in source
    assert "tmin=-0.002" in source[interpolate:filtering]
    assert "tmax=0.008" in source[interpolate:filtering]
    assert "meg=True" in source
    assert "mag_count == 102" in source
    assert "grad_count == 204" in source
    assert "MF_response_-80_-20ms" in source
    assert "MEFI_response_20_60ms" in source
    assert "MEFII_response_120_180ms" in source
    assert "include_stimulus_m1 = False" in source
    assert "stimulus_m1_overlap_fraction" in source
    assert "OASTER-ERP" in source
    assert "require_one=False" in source
    assert 'peak_label = "not_localized"' in source
    assert "if not time_run_available[key]" in source
    assert 'oaster_window_diagnostics.csv' in source
    assert 'method="dSPM"' in source
    assert 'method="eLORETA"' in source
    assert "make_dics" in source
    assert 'response_epochs.resample(dics_sample_rate' in source
    assert "beta_pmbr_db" in source
    assert forward < analysis_vertices < oaster_inverse
    assert "mindist 裁剪后的皮层顶点必须完全一致" in source
    assert last_inverse < roi_read


def test_finger_brain_render_reads_saved_maps_without_recomputing_inverse():
    source = SCRIPTS[5].read_text(encoding="utf-8")

    assert 'views="dorsal"' in source
    assert 'surface="pial"' in source
    assert 'brain.save_image' in source
    assert 'finger_localization_pial_dorsal_montage.png' in source
    assert 'time_domain_primary_method_comparison.png' in source
    assert 'make_forward_solution' not in source
    assert 'apply_inverse' not in source
