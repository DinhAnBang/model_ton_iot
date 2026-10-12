from research_pipeline.src.validation.final_split_protocol import (
    NETWORK_ENDPOINT_1,
    NETWORK_ENDPOINT_2,
    assign_calendar_split,
    assign_network_endpoint_split,
    assign_source_local_calendar_split,
    artifact_root,
    load_config,
    load_coverage,
    load_distribution,
    load_linux_schema_coverage,
    load_manifest,
    pipeline_contract_status,
    validate_distribution,
    validate_linux_schema_coverage,
    validate_manifest,
    validate_primary_known_coverage,
)


def test_calendar_boundaries_are_half_open_and_timezone_less_absolute_times_are_quarantined():
    assert assign_calendar_split("2019-04-23T23:59:59Z") == "train"
    assert assign_calendar_split("2019-04-24T00:00:00Z") == "validation"
    assert assign_calendar_split("2019-04-25T23:59:59Z") == "validation"
    assert assign_calendar_split("2019-04-26T00:00:00Z") == "test"
    assert assign_calendar_split("2019-04-24T00:00:00") == "quarantine"
    assert assign_calendar_split("") == "quarantine"
    assert assign_calendar_split("not-a-time") == "quarantine"


def test_explicit_source_local_rule_is_separate_from_absolute_rule():
    assert assign_source_local_calendar_split("2019-04-23T23:59:59") == "train"
    assert assign_source_local_calendar_split("2019-04-24T00:00:00") == "validation"
    assert assign_source_local_calendar_split("2019-04-26T00:00:00") == "test"
    assert assign_source_local_calendar_split("2019-04-24T00:00:00Z") == "quarantine"


def test_network_endpoint_rule_has_one_owner():
    first = NETWORK_ENDPOINT_1.isoformat().replace("+00:00", "Z")
    second = NETWORK_ENDPOINT_2.isoformat().replace("+00:00", "Z")
    assert assign_network_endpoint_split(first) == "validation"
    assert assign_network_endpoint_split(second) == "test"


def test_compact_distribution_reconciles_inventory():
    assert validate_distribution(load_distribution(artifact_root())) == []
    assert validate_manifest(load_manifest(artifact_root())) == []


def test_iot_is_not_promoted_and_linux_is_schema_specific():
    assert validate_primary_known_coverage(load_coverage(artifact_root())) == []
    assert validate_linux_schema_coverage(load_linux_schema_coverage(artifact_root())) == []


def test_protocol_config_keeps_diagnostics_and_unverified_contracts_explicit():
    config = load_config(artifact_root())
    assert config["decision_status"] == "PARTIALLY_FEASIBLE"
    assert config["main_scopes"] == ["Linux::process"]
    assert config["supporting_scopes"] == ["Linux::disk"]
    assert "IoT" in config["diagnostic_scopes"]
    assert config["pipeline_contract"]["network_half_open_endpoint_implemented"] is False
    assert config["preprocessing"]["full_remediated_export_executed"] is False
    assert config["protocol_b_status"].startswith("BLOCKED")


def test_pipeline_contract_matches_cutoffs_but_does_not_hide_gaps():
    status = pipeline_contract_status(load_config(artifact_root()))
    assert status["calendar_cutoffs_match"] is True
    assert status["quarantine_available_in_remediated_path"] is True
    assert status["network_half_open_endpoint_implemented"] is False
    assert status["iot_timezone_less_input_is_explicitly_source_local"] is False
    assert status["non_iot_unix_row_timestamp_path"] is True
    assert status["non_iot_iso_row_timestamp_path"] is False
    assert status["full_export_executed"] is False
