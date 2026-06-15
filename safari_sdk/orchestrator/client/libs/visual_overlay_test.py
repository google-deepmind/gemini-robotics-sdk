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
    self.assertIsNone(renderer._custom_thickness)
    self.assertIsNone(renderer._custom_font_size)

  def test_custom_thickness_and_font_size(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    self.assertIsNone(renderer._custom_thickness)
    self.assertIsNone(renderer._custom_font_size)

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
    response = renderer.get_image_as_np_array()
    self.assertTrue(response.success)
    image_with_default_overlay = response.visual_overlay_image

    response = renderer.reset_all_object_settings()
    self.assertTrue(response.success)
    self.assertEqual(renderer._overlay_image.size, (200, 200))
    self.assertEqual(renderer._overlay_image_np.shape, (200, 200, 3))
    self.assertEmpty(renderer._workunit_objects)
    self.assertEmpty(renderer._overlay_objects)
    self.assertIsNone(renderer._custom_thickness)
    self.assertIsNone(renderer._custom_font_size)

    response = renderer.set_custom_thickness(thickness=4)
    self.assertTrue(response.success)
    self.assertEqual(renderer._custom_thickness, 4)
    response = renderer.set_custom_font_size(font_size=0.5)
    self.assertTrue(response.success)
    self.assertEqual(renderer._custom_font_size, 0.5)

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
    response = renderer.get_image_as_np_array()
    self.assertTrue(response.success)
    image_with_custom_overlay = response.visual_overlay_image

    is_same_image = np.array_equal(
        image_with_default_overlay, image_with_custom_overlay
    )
    self.assertFalse(is_same_image)

  def test_generate_xy_position_with_no_overruns(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    image_width, image_height = renderer._overlay_image.size
    ideal_x, ideal_y = renderer._generate_xy_position_with_no_overruns(
        x=100,
        y=100,
        x_offset=10,
        y_offset=10,
        image_width=image_width,
        image_height=image_height,
        text_width=10,
        text_height=10,
        text_baseline=5,
    )
    self.assertEqual(ideal_x, 110)
    self.assertEqual(ideal_y, 110)

    ideal_x, ideal_y = renderer._generate_xy_position_with_no_overruns(
        x=1000,
        y=1000,
        x_offset=0,
        y_offset=0,
        image_width=image_width,
        image_height=image_height,
        text_width=10,
        text_height=10,
        text_baseline=5,
    )
    self.assertEqual(ideal_x, 185)
    self.assertEqual(ideal_y, 190)

  def test_find_ideal_text_position(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    text = 'test_label'
    font_scale = 0.75
    thickness = 2
    icon_object = visual_overlay.visual_overlay_icon.DrawCircleIcon(
        object_id='test_object_id_1',
        overlay_text_label='test_label_1',
        rgb_hex_color_value='FF0000',
        layer_order=1,
        x=50,
        y=50,
    )
    ideal_x, ideal_y = renderer._find_ideal_text_position(
        text=text,
        font_scale=font_scale,
        thickness=thickness,
        icon_object=icon_object,
    )
    self.assertEqual(ideal_x, 64)
    self.assertEqual(ideal_y, 55)

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
    response = renderer.get_image_as_np_array()
    self.assertTrue(response.success)
    image_with_overlay = response.visual_overlay_image

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

  def test_adjust_overlay_scale_to_large_reference_image(self):
    large_metadata = _work_unit.SceneReferenceImage(
        artifactId='test_large',
        renderedCanvasWidth=100,
        renderedCanvasHeight=100,
        sourceTopic='topic',
        rawImageWidth=1500,
        rawImageHeight=1000,
    )
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=large_metadata
    )
    self.assertIsNotNone(renderer._custom_font_size)
    self.assertIsNotNone(renderer._custom_thickness)

  def test_update_image_unsupported_type(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    with self.assertRaises(ValueError):
      renderer.render_overlay(new_image=12345)

  def test_process_scene_object_bad_container(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    bad_object = _work_unit.SceneObject(
        objectId='bad_container',
        evaluationLocation=_work_unit.FixedLocation(
            overlayIcon=_work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CONTAINER,
            containerArea=_work_unit.ContainerArea(),
        ),
        sceneReferenceImageArtifactId='test_artifact_id_1',
    )
    with self.assertRaises(ValueError):
      renderer._process_scene_object(bad_object)

  def test_process_scene_object_undefined_icon(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    bad_object = _work_unit.SceneObject(
        objectId='bad_icon',
        evaluationLocation=_work_unit.FixedLocation(
            overlayIcon=-999,
        ),
        sceneReferenceImageArtifactId='test_artifact_id_1',
    )
    with self.assertRaises(ValueError):
      renderer._process_scene_object(bad_object)

  def test_find_ideal_text_position_all_arrow_angles_and_negative_container(
      self,
  ):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    for rad in [-0.2, 0.2, 2.5, -2.5, 1.0, -1.0]:
      arrow = visual_overlay.visual_overlay_icon.DrawArrowIcon(
          object_id='test',
          overlay_text_label='txt',
          rgb_hex_color_value='FF0000',
          layer_order=1,
          x=50,
          y=50,
          rad=rad,
      )
      renderer._find_ideal_text_position(
          text='test',
          font_scale=0.75,
          thickness=2,
          icon_object=arrow,
          arrow_end_point=(10, 10),
      )

    container = visual_overlay.visual_overlay_icon.DrawContainer(
        object_id='test',
        overlay_text_label='txt',
        rgb_hex_color_value='FF0000',
        layer_order=1,
        x=50,
        y=50,
        w=-10,
        h=-10,
    )
    renderer._find_ideal_text_position(
        text='test', font_scale=0.75, thickness=2, icon_object=container
    )

  def test_get_image_as_bytes_png(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    response = renderer.get_image_as_bytes(
        img_format=visual_overlay.ImageFormat.PNG
    )

    self.assertTrue(response.success)
    self.assertIsInstance(response.visual_overlay_image, bytes)
    self.assertNotEmpty(response.visual_overlay_image)

  def test_render_overlay_with_empty_label(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    renderer.add_single_object(
        overlay_object=visual_overlay.visual_overlay_icon.DrawCircleIcon(
            object_id='test_object_id_1',
            overlay_text_label='',
            rgb_hex_color_value='FF0000',
            layer_order=1,
            x=50,
            y=50,
        )
    )
    response = renderer.render_overlay()
    self.assertTrue(response.success)

    scene_objects = [
        _work_unit.SceneObject(
            objectId='test_object_id_2',
            evaluationLocation=_work_unit.FixedLocation(
                overlayIcon=_work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE,
                location=_work_unit.PixelVector(
                    coordinate=_work_unit.PixelLocation(x=10, y=10)
                ),
            ),
            sceneReferenceImageArtifactId='test_artifact_id_1',
        ),
    ]
    response = renderer.load_scene_objects_from_work_unit(scene_objects)
    self.assertTrue(response.success)
    response = renderer.render_overlay()
    self.assertTrue(response.success)

  def test_generate_xy_position_with_no_overruns_left_top(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    image_width, image_height = renderer._overlay_image.size

    # Left overrun
    ideal_x, _ = renderer._generate_xy_position_with_no_overruns(
        x=-10,
        y=100,
        x_offset=0,
        y_offset=0,
        image_width=image_width,
        image_height=image_height,
        text_width=10,
        text_height=10,
        text_baseline=5,
    )
    self.assertEqual(ideal_x, 5)

    # Top overrun
    _, ideal_y = renderer._generate_xy_position_with_no_overruns(
        x=100,
        y=-10,
        x_offset=0,
        y_offset=0,
        image_width=image_width,
        image_height=image_height,
        text_width=10,
        text_height=10,
        text_baseline=5,
    )
    self.assertEqual(ideal_y, 15)  # 5 + text_height

  def test_convert_color_to_tuple_invalid(self):
    renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=_reference_image_metadata_1
    )
    with self.assertRaises(AssertionError):
      renderer._convert_color_to_tuple('FF')

    with self.assertRaises(ValueError):
      renderer._convert_color_to_tuple('ZZZZZZ')


if __name__ == '__main__':
  absltest.main()
