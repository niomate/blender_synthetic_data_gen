import blenderproc as bproc
import numpy as np
import random
import typing as tp
import os


label_mapping = bproc.utility.LabelIdMapping.from_dict({"ball": 1, "box": 2, "cube": 3, "hex": 4, "tube": 5, "ground": 0})
MATERIALS = ["Black", "Green", "Dark Blue", "Grey", "Orange", "Turquoise"]

bproc.init()

def random_angle() -> float:
    return random.random() * np.pi * 2

def random_rpy() -> tp.List[float]:
    return [random_angle(), random_angle(), random_angle()]

def random_scale(lo, hi):
    val = lo + random.random() * (hi - lo)
    return [val, val, val]

# Define a function that samples the pose of a given sphere
def sample_pose_boxes(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform([7, 7, 1.1], [-7, -7, 1.1]))
    obj.set_rotation_euler([np.radians(90), 0, random_angle()])

def sample_pose(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform([7, 7, 3], [-7, -7, 3]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

tube = bproc.loader.load_obj("tube.obj")[0]
hexnut = bproc.loader.load_obj("hex.obj")[0]
ball = bproc.loader.load_obj("ball.obj")[0]
cube = bproc.loader.load_obj("cube.obj")[0]

tube.set_cp("category_id", label_mapping.id_from_label("tube"))
hexnut.set_cp("category_id", label_mapping.id_from_label("hex"))
ball.set_cp("category_id", label_mapping.id_from_label("ball"))
cube.set_cp("category_id", label_mapping.id_from_label("cube"))

objects = [tube, hexnut, ball, cube]

objs = bproc.loader.load_blend("scene.blend", obj_types=["mesh", "empty", "light"], data_blocks=["objects", "lights"])
white_box = bproc.filter.one_by_attr(bproc.filter.all_with_type(objs, bproc.types.MeshObject), "name", "White Box")
black_box = bproc.filter.one_by_attr(bproc.filter.all_with_type(objs, bproc.types.MeshObject), "name", "Black Box")
boxes = [black_box, white_box]

ground = bproc.filter.one_by_attr(bproc.filter.all_with_type(objs, bproc.types.MeshObject), "name", "Ground")
ground.enable_rigidbody(active=False)
ground.set_cp("category_id", label_mapping.id_from_label("ground"))

materials = bproc.material.collect_all()
materials = [bproc.filter.one_by_attr(materials, "name", x + " Scratched") for x in MATERIALS]

N_IMAGES = 500

for i in range(N_IMAGES):
    bproc.utility.reset_keyframes()

    cam_pose = bproc.math.build_transformation_mat([0, -14.776, 21.518], [np.radians(36.6), 0, 0])
    bproc.camera.add_camera_pose(cam_pose)
    bproc.camera.set_intrinsics_from_blender_params(lens=50, lens_unit="MILLIMETERS", image_height=480, image_width=640)

    for box in boxes:       
        box.enable_rigidbody(active=True, collision_shape="MESH")
        box.set_cp("category_id", label_mapping.id_from_label("box"))

    for obj in objects:
        obj.clear_materials()
        obj.set_scale(random_scale(0.8, 1.1))
        obj.add_material(random.choice(materials))
        obj.enable_rigidbody(active=True)

    # Sample the poses of all spheres above the ground without any collisions in-between
    bproc.object.sample_poses(
        objects,
        sample_pose_func=sample_pose
    )
   
    bproc.object.sample_poses(
        boxes,
        sample_pose_func=sample_pose_boxes,
        objects_to_check_collisions=boxes + [ground]
    )


    for box in boxes:        
        box.enable_rigidbody(active=False, collision_shape="MESH")


    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20, check_object_interval=1)
    # Render the scene
    bproc.renderer.enable_segmentation_output(map_by=["instance", "class", "name"])
    data = bproc.renderer.render()

    # Write the rendering into an hdf5 file
    bproc.writer.write_coco_annotations(os.path.join("output", "coco_data"),
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="JPEG",
                                        label_mapping=label_mapping)
