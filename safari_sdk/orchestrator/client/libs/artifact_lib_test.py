# Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Unit tests for artifact.py."""

from unittest import mock

from absl.testing import absltest
from googleapiclient import errors

from safari_sdk.orchestrator.client.libs import artifact


class ArtifactTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_connection = mock.MagicMock()
    self.artifact_lib = artifact.OrchestratorArtifact(
        connection=self.mock_connection
    )
    self.mock_connection.orchestrator().loadArtifact().execute.return_value = {
        "success": True,
        "artifact": {
            "uri": "test_artifact_uri",
            "artifactId": "test_artifact_id",
            "name": "test_name",
            "desc": "test_description",
            "artifactObjectType": "ARTIFACT_OBJECT_TYPE_IMAGE",
            "commitTime": "2025-01-01T00:00:00Z",
            "tags": ["tag1", "tag2"],
            "version": "1",
            "isZipped": False,
        },
    }

  def test_get_artifact_success(self):

    artifact_lib = artifact.OrchestratorArtifact(
        connection=self.mock_connection
    )

    response = artifact_lib.get_artifact(artifact_id="test_artifact_id")
    self.assertTrue(response.success)
    self.assertEqual(response.artifact.uri, "test_artifact_uri")
    self.assertEqual(response.artifact.artifactId, "test_artifact_id")
    self.assertEqual(response.artifact.name, "test_name")
    self.assertEqual(response.artifact.desc, "test_description")
    self.assertEqual(
        response.artifact.artifactObjectType.value,
        "ARTIFACT_OBJECT_TYPE_IMAGE",
    )
    self.assertEqual(response.artifact.commitTime, "2025-01-01T00:00:00Z")
    self.assertEqual(response.artifact.tags, ["tag1", "tag2"])
    self.assertEqual(response.artifact.version, "1")
    self.assertFalse(response.artifact.isZipped)

  def test_get_artifact_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().loadArtifact().execute.side_effect = (
        raise_error_side_effect
    )

    artifact_lib = artifact.OrchestratorArtifact(connection=mock_connection)

    response = artifact_lib.get_artifact(artifact_id="test_artifact_id")

    self.assertFalse(response.success)
    self.assertIn(artifact._ERROR_GET_ARTIFACT, response.error_message)

  @mock.patch("time.time_ns")
  def test_get_artifact_empty_response(self, mock_time):
    mock_time.return_value = 123456789
    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().loadArtifact().execute.return_value = {}
    artifact_lib = artifact.OrchestratorArtifact(connection=mock_connection)

    response = artifact_lib.get_artifact(artifact_id="test_artifact_id")
    self.assertFalse(response.success)
    self.assertEqual(
        artifact._ERROR_EMPTY_RESPONSE + "[Error ID: 123456789]",
        response.error_message,
    )

  @mock.patch("time.time_ns")
  def test_get_artifact_none_response(self, mock_time):
    mock_time.return_value = 123456789
    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().loadArtifact().execute.return_value = None
    artifact_lib = artifact.OrchestratorArtifact(connection=mock_connection)

    response = artifact_lib.get_artifact(artifact_id="test_artifact_id")
    self.assertFalse(response.success)
    self.assertEqual(
        artifact._ERROR_EMPTY_RESPONSE + "[Error ID: 123456789]",
        response.error_message,
    )

  def test_get_artifact_bad_connection(self):
    artifact_lib = artifact.OrchestratorArtifact(connection=None)
    response = artifact_lib.get_artifact(artifact_id="test_artifact_id")
    self.assertFalse(response.success)
    self.assertEqual(
        artifact._ERROR_NO_ORCHESTRATOR_CONNECTION, response.error_message
    )

  def test_get_artifact_uri_success(self):
    artifact_lib = artifact.OrchestratorArtifact(
        connection=self.mock_connection
    )

    response = artifact_lib.get_artifact_uri(artifact_id="test_artifact_id")
    self.assertTrue(response.success)
    self.assertEqual(response.artifact_uri, "test_artifact_uri")

  def test_get_artifact_uri_bad_connection(self):
    artifact_lib = artifact.OrchestratorArtifact(connection=None)
    response = artifact_lib.get_artifact_uri(artifact_id="test_artifact_id")
    self.assertFalse(response.success)
    self.assertEqual(
        artifact._ERROR_NO_ORCHESTRATOR_CONNECTION, response.error_message
    )

  @mock.patch("time.time_ns")
  def test_get_artifact_uri_get_artifact_fails(self, mock_time):
    mock_time.return_value = 123456789
    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().loadArtifact().execute.return_value = {
        "success": False,
        "error_message": "get artifact failed",
        "artifact": None,
    }

    artifact_lib = artifact.OrchestratorArtifact(connection=mock_connection)

    response = artifact_lib.get_artifact_uri(artifact_id="test_artifact_id")

    self.assertFalse(response.success)
    self.assertEqual(
        "OrchestratorArtifact: Received empty response for get artifact"
        " request. [Error ID: 123456789]",
        response.error_message,
    )

  def test_invalid_artifact_id(self):
    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().loadArtifact().execute.return_value = {}
    artifact_lib = artifact.OrchestratorArtifact(connection=mock_connection)
    response = artifact_lib.get_artifact_uri(artifact_id="")
    self.assertFalse(response.success)
    self.assertEqual(artifact._ERROR_EMPTY_ARTIFACT_ID, response.error_message)

  def test_upload_text_log_artifact_success(self):
    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().uploadTextLogArtifact().execute.return_value = {
        "artifactId": "test_uploaded_artifact_id",
    }

    artifact_lib = artifact.OrchestratorArtifact(connection=mock_connection)

    response = artifact_lib.upload_text_log_artifact(
        robot_job_id="test_robot_job_id",
        robot_id="test_robot_id",
        source_file_name="test_log.txt",
        text_file_bytes=b"test log content",
    )
    self.assertTrue(response.success)
    self.assertEqual(response.artifact_id, "test_uploaded_artifact_id")

  def test_upload_text_log_artifact_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().uploadTextLogArtifact().execute.side_effect = (
        raise_error_side_effect
    )

    artifact_lib = artifact.OrchestratorArtifact(connection=mock_connection)

    response = artifact_lib.upload_text_log_artifact(
        robot_job_id="test_robot_job_id",
        robot_id="test_robot_id",
        source_file_name="test_log.txt",
        text_file_bytes=b"test log content",
    )

    self.assertFalse(response.success)
    self.assertIn(
        artifact._ERROR_UPLOAD_TEXT_LOG_ARTIFACT, response.error_message
    )

  @mock.patch("time.time_ns")
  def test_upload_text_log_artifact_empty_response(self, mock_time):
    mock_time.return_value = 123456789
    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().uploadTextLogArtifact().execute.return_value = (
        None
    )
    artifact_lib = artifact.OrchestratorArtifact(connection=mock_connection)

    response = artifact_lib.upload_text_log_artifact(
        robot_job_id="test_robot_job_id",
        robot_id="test_robot_id",
        source_file_name="test_log.txt",
        text_file_bytes=b"test log content",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        artifact._ERROR_EMPTY_RESPONSE + "[Error ID: 123456789]",
        response.error_message,
    )

  def test_upload_text_log_artifact_bad_connection(self):
    artifact_lib = artifact.OrchestratorArtifact(connection=None)
    response = artifact_lib.upload_text_log_artifact(
        robot_job_id="test_robot_job_id",
        robot_id="test_robot_id",
        source_file_name="test_log.txt",
        text_file_bytes=b"test log content",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        artifact._ERROR_NO_ORCHESTRATOR_CONNECTION, response.error_message
    )

  def test_upload_text_log_artifact_empty_robot_job_id(self):
    artifact_lib = artifact.OrchestratorArtifact(
        connection=self.mock_connection
    )
    response = artifact_lib.upload_text_log_artifact(
        robot_job_id="",
        robot_id="test_robot_id",
        source_file_name="test_log.txt",
        text_file_bytes=b"test log content",
    )
    self.assertFalse(response.success)
    self.assertEqual(artifact._ERROR_EMPTY_ROBOT_JOB_ID, response.error_message)

  def test_upload_text_log_artifact_empty_robot_id(self):
    artifact_lib = artifact.OrchestratorArtifact(
        connection=self.mock_connection
    )
    response = artifact_lib.upload_text_log_artifact(
        robot_job_id="test_robot_job_id",
        robot_id="",
        source_file_name="test_log.txt",
        text_file_bytes=b"test log content",
    )
    self.assertFalse(response.success)
    self.assertEqual(artifact._ERROR_EMPTY_ROBOT_ID, response.error_message)

  def test_upload_text_log_artifact_empty_source_file_name(self):
    artifact_lib = artifact.OrchestratorArtifact(
        connection=self.mock_connection
    )
    response = artifact_lib.upload_text_log_artifact(
        robot_job_id="test_robot_job_id",
        robot_id="test_robot_id",
        source_file_name="",
        text_file_bytes=b"test log content",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        artifact._ERROR_EMPTY_SOURCE_FILE_NAME, response.error_message
    )

  def test_upload_text_log_artifact_empty_text_file_bytes(self):
    artifact_lib = artifact.OrchestratorArtifact(
        connection=self.mock_connection
    )
    response = artifact_lib.upload_text_log_artifact(
        robot_job_id="test_robot_job_id",
        robot_id="test_robot_id",
        source_file_name="test_log.txt",
        text_file_bytes=b"",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        artifact._ERROR_EMPTY_TEXT_FILE_BYTES, response.error_message
    )


if __name__ == "__main__":
  absltest.main()
