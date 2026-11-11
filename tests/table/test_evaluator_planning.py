# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

from collections.abc import Callable

import pytest

import pyiceberg.table as table_module
from pyiceberg.expressions import AlwaysTrue, BooleanExpression, EqualTo, GreaterThan
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import (
    DataFile,
    DataFileContent,
    FileFormat,
    ManifestEntry,
    ManifestEntryStatus,
    write_manifest,
)
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import FileScanTask, ManifestGroupPlanner, Table
from pyiceberg.table.metadata import TableMetadataV2
from pyiceberg.transforms import IdentityTransform
from pyiceberg.typedef import Record, StructProtocol
from pyiceberg.types import IntegerType, NestedField


def _data_file(file_number: int, partition_value: int) -> DataFile:
    return DataFile.from_args(
        file_path=f"s3://bucket/data-{file_number}.parquet",
        file_format=FileFormat.PARQUET,
        partition=Record(partition_value),
        record_count=100,
        file_size_in_bytes=1,
    )


def test_partition_evaluator_prepares_once_per_spec(table_v2: Table, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator_calls: list[list[int]] = []

    def counting_expression_evaluator(
        schema: Schema, unbound: BooleanExpression, case_sensitive: bool
    ) -> Callable[[StructProtocol], bool]:
        calls: list[int] = []
        evaluator_calls.append(calls)

        def evaluate(struct: StructProtocol) -> bool:
            value = struct[0]
            calls.append(value)
            return value > 5

        return evaluate

    monkeypatch.setattr(table_module, "expression_evaluator", counting_expression_evaluator)
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=GreaterThan("x", 5))
    partition_evaluator = planner._build_partition_evaluator(0)

    assert len(evaluator_calls) == 1
    assert not partition_evaluator(_data_file(1, 1))
    assert partition_evaluator(_data_file(2, 10))
    assert evaluator_calls == [[1, 10]]


def test_metrics_evaluator_prepares_once_per_plan(table_v2: Table, monkeypatch: pytest.MonkeyPatch) -> None:
    class CountingMetricsEvaluator:
        def __init__(
            self,
            schema: Schema,
            expr: BooleanExpression,
            case_sensitive: bool = True,
            include_empty_files: bool = False,
        ) -> None:
            self.calls: list[DataFile] = []
            instances.append(self)

        def eval(self, data_file: DataFile) -> bool:
            self.calls.append(data_file)
            return True

    instances: list[CountingMetricsEvaluator] = []
    monkeypatch.setattr(table_module, "_InclusiveMetricsEvaluator", CountingMetricsEvaluator)
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=EqualTo("x", 10))
    first_file = _data_file(1, 1)
    second_file = _data_file(2, 2)

    metrics_evaluator = planner._build_metrics_evaluator()
    assert len(instances) == 1
    assert metrics_evaluator(first_file)
    assert metrics_evaluator(second_file)
    assert instances[0].calls == [first_file, second_file]


def _manifest_entry(
    *,
    content: DataFileContent,
    file_path: str,
    sequence_number: int,
    partition: Record,
    snapshot_id: int,
    equality_ids: list[int] | None = None,
) -> ManifestEntry:
    data_file = DataFile.from_args(
        content=content,
        file_path=file_path,
        file_format=FileFormat.PARQUET,
        partition=partition,
        record_count=1,
        file_size_in_bytes=100,
        equality_ids=equality_ids,
    )
    data_file.spec_id = 0
    return ManifestEntry.from_args(
        status=ManifestEntryStatus.ADDED,
        snapshot_id=snapshot_id,
        sequence_number=sequence_number,
        file_sequence_number=sequence_number,
        data_file=data_file,
    )


def test_plan_files_with_equality_deletes(tmp_path: str) -> None:
    schema = Schema(NestedField(field_id=1, name="x", field_type=IntegerType(), required=True), schema_id=0)
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="x"), spec_id=0)
    io = PyArrowFileIO()

    partition_1 = Record(1)
    partition_2 = Record(2)
    data_file_1 = f"{tmp_path}/data-1.parquet"
    data_file_2 = f"{tmp_path}/data-2.parquet"

    manifest_path = f"{tmp_path}/manifest.avro"
    with write_manifest(2, spec, schema, io.new_output(manifest_path), snapshot_id=1, avro_compression="null") as writer:
        # Data files at sequence number 1.
        writer.add_entry(
            _manifest_entry(
                content=DataFileContent.DATA,
                file_path=data_file_1,
                sequence_number=1,
                partition=partition_1,
                snapshot_id=1,
            )
        )
        writer.add_entry(
            _manifest_entry(
                content=DataFileContent.DATA,
                file_path=data_file_2,
                sequence_number=4,
                partition=partition_2,
                snapshot_id=1,
            )
        )
        # Equality delete at sequence number 2, only for partition 1.
        writer.add_entry(
            _manifest_entry(
                content=DataFileContent.EQUALITY_DELETES,
                file_path=f"{tmp_path}/eq-delete.parquet",
                sequence_number=2,
                partition=partition_1,
                snapshot_id=1,
                equality_ids=[1],
            )
        )
        # Equality delete at sequence number 3 for partition 2; the data file there has
        # sequence number 4, so this delete must not be applied to it.
        writer.add_entry(
            _manifest_entry(
                content=DataFileContent.EQUALITY_DELETES,
                file_path=f"{tmp_path}/eq-delete-future.parquet",
                sequence_number=3,
                partition=partition_2,
                snapshot_id=1,
                equality_ids=[1],
            )
        )

    table_metadata = TableMetadataV2(
        location=str(tmp_path),
        last_column_id=1,
        format_version=2,
        current_schema_id=0,
        schemas=[schema],
        partition_specs=[spec],
    )
    planner = ManifestGroupPlanner(table_metadata=table_metadata, io=io, row_filter=AlwaysTrue())
    tasks = list(planner.plan_files([writer.to_manifest_file()]))

    tasks_by_path = {task.file.file_path: task for task in tasks}
    assert set(tasks_by_path) == {data_file_1, data_file_2}

    partition_1_deletes = tasks_by_path[data_file_1].delete_files
    assert len(partition_1_deletes) == 1
    assert next(iter(partition_1_deletes)).content == DataFileContent.EQUALITY_DELETES

    # The future delete (seq 3) does not apply to the data file with seq 4.
    assert tasks_by_path[data_file_2].delete_files == set()


def test_plan_files_keeps_position_and_equality_deletes(tmp_path: str) -> None:
    schema = Schema(NestedField(field_id=1, name="x", field_type=IntegerType(), required=True), schema_id=0)
    spec = PartitionSpec()
    io = PyArrowFileIO()

    manifest_path = f"{tmp_path}/manifest.avro"
    with write_manifest(2, spec, schema, io.new_output(manifest_path), snapshot_id=1, avro_compression="null") as writer:
        writer.add_entry(
            _manifest_entry(
                content=DataFileContent.DATA,
                file_path=f"{tmp_path}/data.parquet",
                sequence_number=1,
                partition=Record(),
                snapshot_id=1,
            )
        )
        writer.add_entry(
            _manifest_entry(
                content=DataFileContent.POSITION_DELETES,
                file_path=f"{tmp_path}/position-delete.parquet",
                sequence_number=2,
                partition=Record(),
                snapshot_id=1,
            )
        )
        writer.add_entry(
            _manifest_entry(
                content=DataFileContent.EQUALITY_DELETES,
                file_path=f"{tmp_path}/equality-delete.parquet",
                sequence_number=2,
                partition=Record(),
                snapshot_id=1,
                equality_ids=[1],
            )
        )

    table_metadata = TableMetadataV2(
        location=str(tmp_path),
        last_column_id=1,
        format_version=2,
        current_schema_id=0,
        schemas=[schema],
        partition_specs=[spec],
    )
    planner = ManifestGroupPlanner(table_metadata=table_metadata, io=io, row_filter=AlwaysTrue())
    tasks = list(planner.plan_files([writer.to_manifest_file()]))

    assert len(tasks) == 1
    task: FileScanTask = tasks[0]
    assert {delete_file.content for delete_file in task.delete_files} == {
        DataFileContent.POSITION_DELETES,
        DataFileContent.EQUALITY_DELETES,
    }
