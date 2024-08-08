import blenderproc as bproc
import numpy as np
import random
import typing as tp
from blenderproc.scripts.saveAsImg import save_array_as_image
import os

# random.seed(1337)
bproc.init()

def random_angle() -> float:
    return random.random() * np.pi * 2

def random_rpy() -> tp.List[float]:
    return [random_angle(), random_angle(), random_angle()]

# Define a function that samples the pose of a given sphere
def sample_pose(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform([6, 6, 2], [-6, -6, 2]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

objs = bproc.loader.load_blend("scene.blend", obj_types=["mesh", "empty", "light"], data_blocks=["objects", "lights"])

ground = bproc.filter.one_by_attr(bproc.filter.all_with_type(objs, bproc.types.MeshObject), "name", "Ground")
ground.enable_rigidbody(active=False)

materials = bproc.material.collect_all()

PRIMITIVES = ["CUBE", "SPHERE", "CYLINDER"]
objects = []


for i in range(7):
    # Create a simple object:
    obj = bproc.object.create_primitive(random.choice(PRIMITIVES))
    # obj.set_location([0, 0, 2])
    # obj.set_rotation_euler(random_rpy())
    obj.add_material(random.choice(materials))
    obj.enable_rigidbody(active=True)

    objects.append(obj)


# Sample the poses of all spheres above the ground without any collisions in-between
bproc.object.sample_poses(
    objects,
    sample_pose_func=sample_pose
)

# Set the camera to be in front of the object
cam_pose = bproc.math.build_transformation_mat([0, -14.776, 21.518], [np.radians(36.6), 0, 0])
bproc.camera.add_camera_pose(cam_pose)

bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20, check_object_interval=1)
# Render the scene
data = bproc.renderer.render()

# Write the rendering into an hdf5 file
bproc.writer.write_hdf5("output/", data)

for index, image in enumerate(data["colors"]):
    save_array_as_image(image, "colors", os.path.join("output/", f"colors_{index}.png"))
