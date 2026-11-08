from topic6_robustness.config import DatasetConfig, ExperimentConfig, ModelConfig, ProjectConfig
from topic6_robustness.experiment import build_conditions


def test_build_conditions_respects_targeted_recovery_and_repeat_counts(tmp_path) -> None:
    config = ProjectConfig(
        project_name="test",
        seed=1,
        dataset=DatasetConfig(tmp_path, tmp_path),
        model=ModelConfig(),
        experiment=ExperimentConfig(
            corruptions=("fog", "low_light"),
            severities=(1, 3, 5),
            repeats=3,
            repeats_by_corruption={"low_light": 1},
            recovery_severities={"fog": (5,), "low_light": (3, 5)},
        ),
        output_dir=tmp_path,
        source_path=tmp_path / "config.yaml",
    )

    conditions = build_conditions(config)
    names = {condition.name for condition in conditions}

    assert "clean_s0_r0" in names
    assert "fog_s1_r2" in names
    assert "fog_s1_r2_recovered" not in names
    assert "fog_s5_r2_recovered" in names
    assert "low_light_s3_r0_recovered" in names
    assert "low_light_s3_r1" not in names
