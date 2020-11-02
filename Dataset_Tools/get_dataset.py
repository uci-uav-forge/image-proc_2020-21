import bpy
import csv
import cv2
from imutils.paths import list_images
import imutils
import math
import mathutils
import numpy as np
import os
import pathlib
from random import choice, random, randint, seed, uniform
import random
import sys
import string
import argparse

DIR="/usr/local/lib/python3.7/site-packages/"
sys.path.append(DIR)
sys.path.append('')

from hdri_operators import rotate, add_new_sun, add_rotation_driver, calculate_sun_position


'''How to use:
blender --background --use-extension 1 -E CYCLES -t 0 -P 'get_dataset.py' 
blender is your path to blender
for macos it is /Applications/Blender.app/Contents/MacOS/blender
'''




def updateCamera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=110.0):
	"""
	Focus the camera to a focus point and place the camera at a specific distance from that
	focus point. The camera stays in a direct line with the focus point.

	:param camera: the camera object
	:type camera: bpy.types.object
	:param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
	:type focus_point: mathutils.Vector
	:param distance: the distance to keep to the focus point (default=``10.0``)
	:type distance: float
	"""
	looking_direction = camera.location - focus_point
	rot_quat = looking_direction.to_track_quat('Z', 'Y')

	camera.rotation_euler = rot_quat.to_euler()
	camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))


def clamp(x, minimum, maximum):
	return max(minimum, min(x, maximum))

def camera_view_bounds_2d(scene, cam_ob, me_ob):
	"""
	Returns camera space bounding box of mesh object.

	Negative 'z' value means the point is behind the camera.

	Takes shift-x/y, lens angle and sensor size into account
	as well as perspective/ortho projections.

	:arg scene: Scene to use for frame size.
	:type scene: :class:`bpy.types.Scene`
	:arg obj: Camera object.
	:type obj: :class:`bpy.types.Object`
	:arg me: Untransformed Mesh.
	:type me: :class:`bpy.types.Mesh´
	:return: a Box object (call its to_tuple() method to get x, y, width and height)
	:rtype: :class:`Box`
	"""

	mat = cam_ob.matrix_world.normalized().inverted()
	depsgraph = bpy.context.evaluated_depsgraph_get()
	mesh_eval = me_ob.evaluated_get(depsgraph)
	me = mesh_eval.to_mesh()
	me.transform(me_ob.matrix_world)
	me.transform(mat)

	camera = cam_ob.data
	frame = [-v for v in camera.view_frame(scene=scene)[:3]]
	camera_persp = camera.type != 'ORTHO'

	lx = []
	ly = []

	for v in me.vertices:
		co_local = v.co
		z = -co_local.z

		if camera_persp:
			if z == 0.0:
				lx.append(0.5)
				ly.append(0.5)
			# Does it make any sense to drop these?
			# if z <= 0.0:
			# 	continue
			else:
				frame = [(v / (v.z / z)) for v in frame]

		min_x, max_x = frame[1].x, frame[2].x
		min_y, max_y = frame[0].y, frame[1].y

		x = (co_local.x - min_x) / (max_x - min_x)
		y = (co_local.y - min_y) / (max_y - min_y)

		lx.append(x)
		ly.append(y)

	min_x = clamp(min(lx), 0.0, 1.0)
	max_x = clamp(max(lx), 0.0, 1.0)
	min_y = clamp(min(ly), 0.0, 1.0)
	max_y = clamp(max(ly), 0.0, 1.0)

	mesh_eval.to_mesh_clear()

	r = scene.render
	fac = r.resolution_percentage * 0.01
	dim_x = r.resolution_x * fac
	dim_y = r.resolution_y * fac

	# Sanity check
	if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
		return (0, 0, 0, 0)

	return (
		round(min_x * dim_x),			 # X
		round(dim_y - max_y * dim_y),	 # Y
		round((max_x - min_x) * dim_x),  # Width
		round((max_y - min_y) * dim_y)   # Height
	)


class Colour:
	_START = '#'
	def __init__(self, value):
		value = list(value)
		if len(value) != 3:
			raise ValueError('value must have a length of three')
		self._values = value

	def __str__(self):
		return _START + ''.join('{:02X}'.format(v) for v in self)

	def __iter__(self):
		return iter(self._values)

	def __getitem__(self, index):
		return self._values[index]

	def __setitem__(self, index):
		return self._values[index]

	@staticmethod
	def from_string(string):
		colour = iter(string)
		if string[0] == _START:
			next(colour, None)
		return Colour(int(''.join(v), 16) for v in zip(colour, colour))

	@staticmethod
	def random():
		return Colour(random.randrange(256) for _ in range(3))

	def contrast(self):
		return Colour(255 - v for v in self)

	@staticmethod
	def hex_to_rgb(hex):
		return tuple(int(hex[i:i+2], 16)/255 for i in (1, 2, 4))

def get_material_name(obj):
	mesh = obj.data
	mat_name = ''
	for f in mesh.polygons:  # iterate over faces
		slot = obj.material_slots[f.material_index]
		mat = slot.material
		mat_name = mat.name
	return mat_name



def getBinaryMask(img_path, bgr_color):
	img = cv2.imread(img_path)  
	img_height, img_width, _ = img.shape
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask = cv2.compare(gray,11,cv2.CMP_LT)
	inverted_mask = cv2.bitwise_not(mask)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	eroded_mask = cv2.erode(inverted_mask, kernel)
	dilated_mask = cv2.dilate(eroded_mask, kernel)

	b, g, r = bgr_color
	dilated_mask_3d = np.repeat(dilated_mask[:, :, np.newaxis], 3, axis=2)
	dilated_mask_3d[:, :, 0] = np.where(dilated_mask_3d[:, :, 0]>0, b, dilated_mask_3d[:, :, 0])
	dilated_mask_3d[:, :, 1] = np.where(dilated_mask_3d[:, :, 1]>0, g, dilated_mask_3d[:, :, 1])
	dilated_mask_3d[:, :, 2] = np.where(dilated_mask_3d[:, :, 2]>0, r, dilated_mask_3d[:, :, 2])

	return dilated_mask_3d


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())

# GPU
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'


seed(1)

parser = ArgumentParserForBlender()
parser.add_argument("--shape", type=str, required=True, choices=['Half_Circle', 'Circle', 'Heart', 'Plus', 'Square', 'Triangle'])
parser.add_argument("--letter", type=str, required=True, help='uppercase letter A-Z')
parser.add_argument("--num_images", type=int, required=True, help='number of images per shape letter pair')
args = parser.parse_args()

shapes_list = ['Half_Circle', 'Circle', 'Heart', 'Plus', 'Square', 'Triangle']
letters_list = list(string.ascii_uppercase)
color_matrix = {'Half_Circle': (255, 0, 0), 'Circle': (0, 255, 0), 'Heart': (0, 0, 255),
				'Plus': (255, 255, 0), 'Square': (0, 255, 255), 'Triangle': (255, 255, 255)}
shape_name = args.shape
letter_name = args.letter
num_images = args.num_images



# import object
shape_path = os.path.join(os.getcwd(), 'shapes', shape_name + '.obj')
letter_path = os.path.join(os.getcwd(), 'letters', letter_name + '.obj')
bpy.ops.import_scene.obj( filepath = shape_path, filter_glob="*.obj;*.mtl" )
bpy.ops.import_scene.obj( filepath = letter_path, filter_glob="*.obj;*.mtl" )
object_name = shape_name + '_' + letter_name
dataset_img_path = os.path.join(os.getcwd(), 'Dataset', object_name, 'images')
dataset_mask_path = os.path.join(os.getcwd(), 'Dataset', object_name, 'masks')
if not os.path.exists(dataset_img_path):
	os.mkdir(dataset_img_path)
if not os.path.exists(dataset_mask_path):
	os.mkdir(dataset_mask_path)
annotations_summary_save_path = os.sep.join(dataset_img_path.split(os.sep)[:-1])

if not os.path.isdir(dataset_img_path):
	os.makedirs(dataset_img_path)


# unlink cube
target = bpy.data.objects['Cube']
objs = bpy.data.objects
objs.remove(objs["Cube"], do_unlink=True)

letter = None
shape = None
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
	if 'Camera' not in obj.name and 'Light' not in obj.name:
		if any([True for shape in shapes_list if shape in obj.name]):
			obj.name = shape_name
			shape = bpy.data.objects[obj.name]
		elif any([True for letter in letters_list if letter in obj.name]):
			obj.name = letter_name
			letter = bpy.data.objects[obj.name]
		bpy.data.objects[obj.name].select_set(True)


# target settings
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
letter.location = (0, 0, .001)
shape.location = (0, 0, 0)
dg = bpy.context.evaluated_depsgraph_get() 
letter.rotation_mode = 'XYZ' 
shape.rotation_mode = 'XYZ' 

# original dimensions
shape_x_dim = shape.dimensions.x
shape_y_dim = shape.dimensions.y
shape_z_dim = shape.dimensions.z

# prep shape/letter size
def meters_to_feet(target, feet, orig_x_dim, orig_y_dim, orig_z_dim):
	feet_to_meters = 0.3048 * feet
	largest_dimension = max((orig_x_dim, orig_y_dim, orig_z_dim))
	muliplier = feet_to_meters / largest_dimension
	target.dimensions=(muliplier*orig_x_dim, muliplier*orig_y_dim, muliplier*orig_z_dim)



# scale range is scale for letter to fit shape, lower bound is scale for letter to fit shape of size feet=1, upper bound is scale for letter to fit shape of size feet=2
scale_range_dict = {'Half_Circle': (6, 12),
					'Circle': (9, 18),
					'Heart': (8, 16),
					'Plus': (7.5, 15),
					'Square': (10, 22),
					'Triangle': (5.5, 11)}


# print names of selected objects 
sel = bpy.context.selected_objects
for obj in sel:
	print("selected obj:", obj.name)


# settings for rendered image
res_x = 448 
res_y = 448 
bpy.data.scenes["Scene"].render.resolution_x = res_x
bpy.data.scenes["Scene"].render.resolution_y = res_y
bpy.context.scene.render.image_settings.file_format='JPEG'

# set up randomized backgrounds
background_image_paths = list(list_images('Supplementary_Dataset/Backgrounds'))

# set up background image
bpy.context.scene.render.film_transparent = True
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
composite = tree.nodes[0]
render_layers = tree.nodes[1]
alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
background_img_node = tree.nodes.new(type="CompositorNodeImage")
scale_node = tree.nodes.new(type="CompositorNodeScale")
links = tree.links
link_1 = links.new(render_layers.outputs[0], alpha_over.inputs[2])
link_2 = links.new(alpha_over.outputs[0], composite.inputs[0])
link_3 = links.new(background_img_node.outputs[0], scale_node.inputs[0])
link_4 = links.new(scale_node.outputs[0], alpha_over.inputs[1])
bpy.data.scenes["Scene"].node_tree.nodes["Scale"].space = 'RENDER_SIZE'






# camera settings
cam = bpy.data.objects['Camera']
bpy.data.cameras['Camera'].type = 'PERSP'
cam.rotation_euler = [0, 0, 0]
cam.location = [0, 0, 0]
cam.data.clip_end = 1e+08
feet_to_meters = 0.3048 
camera_dist_range = (10 * feet_to_meters, 40 * feet_to_meters)
camera_location_range = (-2.5, 2.5)




# hdri
bpy.ops.preferences.addon_install(filepath=os.path.join(os.getcwd(), 'hdri-sun-aligner-1_5.zip'))
bpy.ops.preferences.addon_enable(module='hdri-sun-aligner-1_5')
hdri_dataset_path = 'Supplementary_Dataset/HDRI'
hdri_images = []
for f in os.scandir(hdri_dataset_path):
	if os.path.isfile(f.path):
		hdri_images.append(f.path)
scene = bpy.context.scene
scene.render.film_transparent = True
# set up hdri nodes
world = bpy.context.scene.world
world.use_nodes = True
node_tree = world.node_tree
links = node_tree.links
background_node = world.node_tree.nodes['Background']
environment_texture_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
mapping_node = node_tree.nodes.new(type="ShaderNodeMapping")
tex_coord_node = node_tree.nodes.new(type="ShaderNodeTexCoord")
node_tree.links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
node_tree.links.new(mapping_node.outputs["Vector"], environment_texture_node.inputs["Vector"])
node_tree.links.new(environment_texture_node.outputs["Color"], node_tree.nodes["Background"].inputs["Color"])


img_id = 0
with open(os.path.join(annotations_summary_save_path, 'annotations_summary.csv'), 'w+', newline='') as annotations_summary:
	summary_writer = csv.writer(annotations_summary, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	# don't rewrite the headers when appending new rows. open append (at end), seek to the top and read it all the way until the end, your at the end again for writing.
	annotations_summary.seek(os.SEEK_SET)
	annotations_summary = [row for row in csv.DictReader(annotations_summary)]
	if len(annotations_summary) == 0:
		summary_writer.writerow(['Object_Name', 'Image_Path', 'X', 'Y', 'Width', 'Height', 'Img_Width', 'Img_Height'])
		
	while img_id < num_images:
		# random size
		random_feet = uniform(1, 2)
		percent_inc = random_feet - 1
		new_letter_scale = (scale_range_dict[shape_name][1] - scale_range_dict[shape_name][0]) * percent_inc + scale_range_dict[shape_name][0]
		meters_to_feet(shape, random_feet, shape_x_dim, shape_y_dim, shape_z_dim)
		letter.scale = (new_letter_scale, new_letter_scale, new_letter_scale)

		# random color
		base = Colour.random()
		color_1 = str(base)
		color_2 = str(base.contrast())
		letter_rgb = Colour.hex_to_rgb(color_1)
		shape_rgb = Colour.hex_to_rgb(color_2)
		material_name = get_material_name(shape)
		bpy.data.materials[material_name].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (letter_rgb[0], letter_rgb[1], letter_rgb[2], 1)
		material_name = get_material_name(letter)
		bpy.data.materials[material_name].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (shape_rgb[0], shape_rgb[1], shape_rgb[2], 1)

		# update camera
		random_cam_dist = uniform(camera_dist_range[0], camera_dist_range[1])
		percent_inc = (random_cam_dist - camera_dist_range[0]) / camera_dist_range[0] + 1
		percent_inc = random_cam_dist / camera_dist_range[0]
		random_cam_location_x = uniform(feet_to_meters*camera_location_range[0] * percent_inc, feet_to_meters*camera_location_range[1] * percent_inc)
		random_cam_location_y = uniform(feet_to_meters*camera_location_range[0] * percent_inc, feet_to_meters*camera_location_range[1] * percent_inc)
		cam.location = (random_cam_location_x, random_cam_location_y, random_cam_dist)
		cam.rotation_euler = [0, 0, uniform(0, 2*math.pi)]

		# render img
		fn = dataset_img_path + '/' + object_name +  '_img_{}.jpg'.format(img_id)
		print(fn)
		bpy.data.scenes["Scene"].render.filepath = fn
		bpy.ops.render.render(write_still=True)

		# write annotations
		x, y, w, h = camera_view_bounds_2d(bpy.context.scene, bpy.context.scene.camera, shape)
		img_name = object_name +  '_img_{}.jpg'.format(img_id)

		summary_writer.writerow([object_name, img_name, x, y, w, h, res_x, res_y])


		# save mask
		mask = getBinaryMask(fn, color_matrix[shape_name])
		cv2.imwrite(dataset_mask_path + '/' + object_name +  '_img_{}.jpg'.format(img_id), mask)

		# resave img with random background image and hdri
		random_hdri = choice(hdri_images)
		environment_texture_node.image = bpy.data.images.load(random_hdri)
		print('HDRI__', environment_texture_node.image)
		# set up sun aligned with hdri
		for obj in bpy.data.objects:
			if 'HDRI Sun' == obj.name:
				bpy.data.objects.remove(bpy.data.objects['HDRI Sun'], do_unlink=True)
		context = bpy.context
		add_new_sun(context)
		calculate_sun_position(context)
		add_rotation_driver(context)
		rotate(context)
		sun = bpy.data.objects['HDRI Sun']
		sun.data.energy = uniform(0.5, 1.5)
		mapping_node.inputs['Rotation'].default_value = (0.0, 0.0, random.random()*360)
		# make sure to rotate sun
		context = bpy.context
		add_rotation_driver(context)
		rotate(context)

		background_img = choice(background_image_paths)
		background_img_node.image = bpy.data.images.load(background_img)
		fn = dataset_img_path + '/' + object_name +  '_img_{}.jpg'.format(img_id)
		print(fn)
		bpy.data.scenes["Scene"].render.filepath = fn
		bpy.ops.render.render(write_still=True)

		# remove hdri and background
		bpy.data.images.remove(environment_texture_node.image)
		bpy.data.images.remove(background_img_node.image)
		cam.location = [0, 0, 0]
		img_id += 1














