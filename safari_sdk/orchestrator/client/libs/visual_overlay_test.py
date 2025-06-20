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

"""Unit tests for visual_overlay."""

from absl.testing import absltest
import numpy as np
from PIL import Image

from safari_sdk.orchestrator.client.libs import visual_overlay

_work_unit = visual_overlay.work_unit
_reference_image_metadata_1 = _work_unit.SceneReferenceImage(
    artifactId='test_artifact_id_1',
    renderedCanvasWidth=100,
    renderedCanvasHeight=100,
    sourceTopic='test_source_topic_1',
    rawImageWidth=200,
    rawImageHeight=200,
)


class RendererTest(absltest.TestCase):

  def test_reset_all_object_settings(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    renderer._overlay_image = Image.new(
        mode='RGB',
        size=(100, 100),
        color='red',
    )
    renderer._overlay_image_np = np.array(renderer._overlay_image)
    renderer._workunit_objects = [
        _work_unit.SceneObject(
            objectId='test_object_id_1',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[
                    _work_unit.OverlayText(
                        text='test_label_1',
                    )
                ]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE
                ),
                layerOrder=1,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=10, y=10)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_1',
        ),
    ]
    renderer._overlay_objects = [
        visual_overlay.visual_overlay_icon.DrawCircleIcon(
            object_id='1',
            overlay_text_label='label',
            rgb_hex_color_value='FF0000',
            layer_order=1,
            x=10,
            y=10,
        )
    ]
    response = renderer.reset_all_object_settings()

    self.assertTrue(response.success)
    self.assertEqual(renderer._overlay_image.size, (200, 200))
    self.assertEqual(renderer._overlay_image_np.shape, (200, 200, 3))
    self.assertEmpty(renderer._workunit_objects)
    self.assertEmpty(renderer._overlay_objects)

  def test_get_image_as_pil_image(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    response = renderer.get_image_as_pil_image()

    self.assertTrue(response.success)
    self.assertIsInstance(response.visual_overlay_image, Image.Image)
    self.assertEqual(response.visual_overlay_image.size, (200, 200))
    self.assertEqual(response.visual_overlay_image.mode, 'RGB')

  def test_get_image_as_np_array(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    response = renderer.get_image_as_np_array()

    self.assertTrue(response.success)
    self.assertIsInstance(response.visual_overlay_image, np.ndarray)
    self.assertEqual(response.visual_overlay_image.shape, (200, 200, 3))

  def test_get_image_as_bytes(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    response = renderer.get_image_as_bytes()

    self.assertTrue(response.success)
    self.assertIsInstance(response.visual_overlay_image, bytes)
    self.assertNotEmpty(response.visual_overlay_image)

  def test_load_scene_objects_from_work_unit_with_filter(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    scene_objects = [
        _work_unit.SceneObject(
            objectId='test_object_id_1',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[
                    _work_unit.OverlayText(
                        text='test_label_1',
                    )
                ]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE
                ),
                layerOrder=1,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=10, y=10)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_1',
        ),
        _work_unit.SceneObject(
            objectId='test_object_id_2',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[
                    _work_unit.OverlayText(
                        text='test_label_2',
                    )
                ]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE
                ),
                layerOrder=2,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=20, y=20)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_2',
        ),
    ]

    response = renderer.load_scene_objects_from_work_unit(
        scene_objects=scene_objects
    )

    self.assertTrue(response.success)
    self.assertLen(renderer._workunit_objects, 2)
    self.assertLen(renderer._overlay_objects, 1)

  def test_load_scene_objects_from_work_unit_with_no_valid_objects(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    scene_objects = [
        _work_unit.SceneObject(
            objectId='test_object_id_1',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[
                    _work_unit.OverlayText(
                        text='test_label_1',
                    )
                ]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE
                ),
                layerOrder=1,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=10, y=10)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_2',
        ),
        _work_unit.SceneObject(
            objectId='test_object_id_2',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[
                    _work_unit.OverlayText(
                        text='test_label_2',
                    )
                ]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE
                ),
                layerOrder=2,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=20, y=20)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_2',
        ),
    ]

    response = renderer.load_scene_objects_from_work_unit(
        scene_objects=scene_objects
    )

    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message,
        visual_overlay._ERROR_NO_SCENE_OBJECTS_FOR_REFERENCE_IMAGE,
    )
    self.assertLen(renderer._workunit_objects, 2)
    self.assertEmpty(renderer._overlay_objects)

  def test_load_scene_objects_from_work_unit_with_no_objects(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )

    response = renderer.load_scene_objects_from_work_unit(scene_objects=[])

    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, visual_overlay._ERROR_NO_SCENE_OBJECTS_FOUND
    )
    self.assertEmpty(renderer._workunit_objects)
    self.assertEmpty(renderer._overlay_objects)

    response = renderer.load_scene_objects_from_work_unit(scene_objects=None)

    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, visual_overlay._ERROR_NO_SCENE_OBJECTS_FOUND
    )
    self.assertEmpty(renderer._workunit_objects)
    self.assertEmpty(renderer._overlay_objects)

  def test_render_overlay_with_add_single_object(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    empty_image = renderer._overlay_image_np.copy()
    renderer.add_single_object(
        overlay_object=visual_overlay.visual_overlay_icon.DrawContainer(
            object_id='test_object_id_1',
            overlay_text_label='test_label_1',
            rgb_hex_color_value='FF0000',
            layer_order=1,
            x=30,
            y=30,
            w=30,
            h=30,
        )
    )
    renderer.add_single_object(
        overlay_object=visual_overlay.visual_overlay_icon.DrawTriangleIcon(
            object_id='test_object_id_2',
            overlay_text_label='test_label_2',
            rgb_hex_color_value='FF0000',
            layer_order=2,
            x=50,
            y=50,
        )
    )
    renderer.add_single_object(
        overlay_object=visual_overlay.visual_overlay_icon.DrawSquareIcon(
            object_id='test_object_id_3',
            overlay_text_label='test_label_3',
            rgb_hex_color_value='FF0000',
            layer_order=3,
            x=50,
            y=50,
        )
    )
    renderer.add_single_object(
        overlay_object=visual_overlay.visual_overlay_icon.DrawCircleIcon(
            object_id='test_object_id_4',
            overlay_text_label='test_label_4',
            rgb_hex_color_value='FF0000',
            layer_order=4,
            x=50,
            y=50,
        )
    )
    renderer.add_single_object(
        overlay_object=visual_overlay.visual_overlay_icon.DrawArrowIcon(
            object_id='test_object_id_5',
            overlay_text_label='test_label_5',
            rgb_hex_color_value='FF0000',
            layer_order=5,
            x=50,
            y=50,
            rad=0.5,
        )
    )
    renderer.add_single_object(
        overlay_object=visual_overlay.visual_overlay_icon.DrawContainer(
            object_id='test_object_id_6',
            overlay_text_label='test_label_6',
            rgb_hex_color_value='FF0000',
            layer_order=6,
            x=50,
            y=50,
            radius=5,
        )
    )
    renderer.render_overlay()
    image_with_overlay = renderer.get_image_as_np_array()

    self.assertFalse(np.array_equal(empty_image, image_with_overlay))

  def test_render_overlay_with_scene_objects(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    empty_image = renderer._overlay_image_np.copy()
    scene_objects = [
        _work_unit.SceneObject(
            objectId='test_object_id_1',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[
                    _work_unit.OverlayText(
                        text='test_label_1',
                    )
                ]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE
                ),
                layerOrder=1,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=10, y=10)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_1',
        ),
        _work_unit.SceneObject(
            objectId='test_object_id_2',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[
                    _work_unit.OverlayText(
                        text='test_label_2',
                    )
                ]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE
                ),
                layerOrder=2,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=20, y=20)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_2',
        ),
    ]

    response = renderer.load_scene_objects_from_work_unit(
        scene_objects=scene_objects
    )
    self.assertTrue(response.success)

    response = renderer.render_overlay()
    self.assertTrue(response.success)

    response = renderer.get_image_as_np_array()
    self.assertTrue(response.success)

    self.assertFalse(np.array_equal(empty_image, response.visual_overlay_image))

  def test_render_overlay_with_new_image(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    initial_image = renderer._overlay_image_np.copy()
    self.assertListEqual(initial_image[0][0].tolist(), [68, 68, 68])

    scene_objects = [
        _work_unit.SceneObject(
            objectId='test_object_id_1',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[_work_unit.OverlayText(text='test_label_1')]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE
                ),
                layerOrder=1,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=10, y=10)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_1',
        ),
        _work_unit.SceneObject(
            objectId='test_object_id_2',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[_work_unit.OverlayText(text='test_label_2')]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE
                ),
                layerOrder=2,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=20, y=20)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_2',
        ),
        _work_unit.SceneObject(
            objectId='test_object_id_3',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[_work_unit.OverlayText(text='test_label_3')]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_ARROW
                ),
                layerOrder=1,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=10, y=10),
                    direction=_work_unit.PixelDirection(rad=0.5),
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_1',
        ),
        _work_unit.SceneObject(
            objectId='test_object_id_4',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[_work_unit.OverlayText(text='test_label_4')]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_SQUARE
                ),
                layerOrder=1,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=10, y=10)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_1',
        ),
        _work_unit.SceneObject(
            objectId='test_object_id_5',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[_work_unit.OverlayText(text='test_label_5')]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_TRIANGLE
                ),
                layerOrder=1,
                rgbHexColorValue='FF0000',
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=10, y=10)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_1',
        ),
        _work_unit.SceneObject(
            objectId='test_object_id_6',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[_work_unit.OverlayText(text='test_label_6')]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CONTAINER
                ),
                layerOrder=1,
                rgbHexColorValue='FF0000',
                containerArea=_work_unit.ContainerArea(
                    circle=_work_unit.ShapeCircle(
                        radius=10,
                        center=_work_unit.PixelLocation(x=10, y=10),
                    )
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_1',
        ),
        _work_unit.SceneObject(
            objectId='test_object_id_7',
            overlayTextLabels=_work_unit.OverlayTextLabel(
                labels=[_work_unit.OverlayText(text='test_label_7')]
            ),
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=(
                    _work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CONTAINER
                ),
                layerOrder=1,
                rgbHexColorValue='FF0000',
                containerArea=_work_unit.ContainerArea(
                    box=_work_unit.ShapeBox(x=10, y=10, w=2, h=2),
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_1',
        ),
    ]

    response = renderer.load_scene_objects_from_work_unit(
        scene_objects=scene_objects
    )
    self.assertTrue(response.success)

    new_image = Image.new(mode='RGB', size=(200, 200), color='black')
    response = renderer.render_overlay(new_image=new_image)
    self.assertTrue(response.success)

    self.assertFalse(np.array_equal(initial_image, renderer._overlay_image_np))
    self.assertListEqual(renderer._overlay_image_np[0][0].tolist(), [0, 0, 0])

    new_image_np = np.zeros((200, 200, 3), dtype=np.uint8)
    response = renderer.render_overlay(new_image=new_image_np)
    self.assertTrue(response.success)

    self.assertFalse(np.array_equal(initial_image, renderer._overlay_image_np))
    self.assertListEqual(renderer._overlay_image_np[0][0].tolist(), [0, 0, 0])

    img_byte_arr = visual_overlay.io.BytesIO()
    new_image = Image.new(mode='RGB', size=(200, 200), color='black')
    new_image.save(img_byte_arr, format=visual_overlay.ImageFormat.JPEG.value)
    new_image_bytes = img_byte_arr.getvalue()
    response = renderer.render_overlay(new_image=new_image_bytes)
    self.assertTrue(response.success)

    self.assertFalse(np.array_equal(initial_image, renderer._overlay_image_np))
    self.assertListEqual(renderer._overlay_image_np[0][0].tolist(), [0, 0, 0])


if __name__ == '__main__':
  absltest.main()
