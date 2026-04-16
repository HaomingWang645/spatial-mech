from spatial_subspace.scene import Camera, Frame, Object3D, QAItem, Scene


def test_scene_roundtrip(tmp_path):
    scene = Scene(
        scene_id="a_test",
        tier="A",
        objects=[
            Object3D(
                object_id=0,
                shape="cube",
                color="red",
                size="small",
                centroid=(1.0, 2.0, 0.0),
                bbox_min=(0.5, 1.5, -0.5),
                bbox_max=(1.5, 2.5, 0.5),
            ),
        ],
        frames=[
            Frame(
                frame_id=0,
                image_path="frames/000.png",
                mask_path="masks/000.png",
                camera=Camera(
                    intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    extrinsics=[
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    kind="orthographic",
                ),
            ),
        ],
        qa=[QAItem(question="q?", answer="yes", kind="rel", involves=[0])],
    )
    scene.save(tmp_path)
    loaded = Scene.load(tmp_path)
    assert loaded.scene_id == scene.scene_id
    assert loaded.tier == "A"
    assert loaded.objects[0].shape == "cube"
    assert tuple(loaded.objects[0].centroid) == (1.0, 2.0, 0.0)
    assert loaded.frames[0].image_path == "frames/000.png"
    assert loaded.qa[0].kind == "rel"
