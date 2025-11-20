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

from absl.testing import absltest
from safari_sdk.orchestrator.client.dataclass import artifact


class ArtifactTest(absltest.TestCase):

  def test_artifact_dataclass_parses_correctly(self):
    input_json = """
    {
      "artifact": {
        "uri": "gs://test-bucket/test-artifact.png",
        "artifactId": "test-artifact-id",
        "name": "test-artifact-name",
        "desc": "test-artifact-desc",
        "artifactObjectType": "ARTIFACT_OBJECT_TYPE_IMAGE",
        "commitTime": "2025-01-01T00:00:00Z",
        "tags": ["test-tag-1", "test-tag-2"],
        "version": "test-artifact-version",
        "isZipped": true
      }
    }
    """
    artifact_response = artifact.LoadArtifactResponse.from_json(input_json)
    self.assertEqual(
        artifact_response.artifact.uri, "gs://test-bucket/test-artifact.png"
    )
    self.assertEqual(artifact_response.artifact.artifactId, "test-artifact-id")
    self.assertEqual(artifact_response.artifact.name, "test-artifact-name")
    self.assertEqual(artifact_response.artifact.desc, "test-artifact-desc")
    self.assertEqual(
        artifact_response.artifact.artifactObjectType,
        artifact.ArtifactObjectType.ARTIFACT_OBJECT_TYPE_IMAGE,
    )
    self.assertEqual(
        artifact_response.artifact.commitTime, "2025-01-01T00:00:00Z"
    )
    self.assertEqual(
        artifact_response.artifact.tags, ["test-tag-1", "test-tag-2"]
    )
    self.assertEqual(
        artifact_response.artifact.version, "test-artifact-version"
    )
    self.assertEqual(artifact_response.artifact.isZipped, True)

  def test_artifact_dataclass_parses_correctly_with_defaults(self):
    input_json = """
    {
      "artifact": {
        "uri": "gs://test-bucket/test-artifact.png"
      }
    }
    """
    artifact_response = artifact.LoadArtifactResponse.from_json(input_json)
    self.assertEqual(
        artifact_response.artifact.uri, "gs://test-bucket/test-artifact.png"
    )
    self.assertEqual(
        artifact_response.artifact.artifactObjectType,
        artifact.ArtifactObjectType.ARTIFACT_OBJECT_TYPE_UNSPECIFIED,
    )

  def test_artifact_dataclass_parses_correctly_with_none_values(self):
    input_json = """
    {
      "artifact": {
        "uri": null,
        "artifactId": null,
        "name": null,
        "desc": null,
        "artifactObjectType": null,
        "commitTime": null,
        "tags": null,
        "version": null,
        "isZipped": null
      }
    }
    """
    artifact_response = artifact.LoadArtifactResponse.from_json(input_json)
    self.assertIsNone(artifact_response.artifact.uri)
    self.assertIsNone(artifact_response.artifact.artifactId)
    self.assertIsNone(artifact_response.artifact.name)
    self.assertIsNone(artifact_response.artifact.desc)
    self.assertIsNone(artifact_response.artifact.commitTime)
    self.assertIsNone(artifact_response.artifact.tags)
    self.assertIsNone(artifact_response.artifact.version)
    self.assertIsNone(artifact_response.artifact.isZipped)
    self.assertEqual(
        artifact_response.artifact.artifactObjectType,
        artifact.ArtifactObjectType.ARTIFACT_OBJECT_TYPE_UNSPECIFIED,
    )

  def test_artifact_dataclass_does_not_parse_with_empty_values(self):
    input_json = """
    {
      "artifact": {
        "uri": "",
        "artifactId": "",
        "name": "",
        "desc": "",
        "artifactObjectType": "",
        "commitTime": "",
        "tags": [],
        "version": "",
        "isZipped": false
      }
    }
    """
    with self.assertRaises(ValueError):
      artifact.LoadArtifactResponse.from_json(input_json)

  def test_artifact_dataclass_parses_correctly_with_no_values(self):
    input_json = """
    {
      "artifact": { }
    }
    """
    artifact_response = artifact.LoadArtifactResponse.from_json(input_json)
    self.assertIsNone(artifact_response.artifact.uri)
    self.assertIsNone(artifact_response.artifact.artifactId)
    self.assertIsNone(artifact_response.artifact.name)
    self.assertIsNone(artifact_response.artifact.desc)
    self.assertEqual(
        artifact_response.artifact.artifactObjectType,
        artifact.ArtifactObjectType.ARTIFACT_OBJECT_TYPE_UNSPECIFIED,
    )
    self.assertIsNone(artifact_response.artifact.commitTime)
    self.assertIsNone(artifact_response.artifact.tags)
    self.assertIsNone(artifact_response.artifact.version)
    self.assertIsNone(artifact_response.artifact.isZipped)


if __name__ == "__main__":
  absltest.main()
