import math
from transformations import *
import numpy as np

PI = math.pi
type3=(
    [
        [ 1.0,     0,     0],
        [ 0,     1.0,     0],
        [ 0,       0,   1.0],
    ],
    [
        [ math.cos(PI/6), math.sin(PI/6),          0],
        [-math.sin(PI/6), math.cos(PI/6),          0],
        [              0,              0,        1.0]
    ],
    [
        [ math.cos(PI*2/6), math.sin(PI*2/6),          0],
        [-math.sin(PI*2/6), math.cos(PI*2/6),          0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*3/6), math.sin(PI*3/6),          0],
        [-math.sin(PI*3/6), math.cos(PI*3/6),          0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*4/6), math.sin(PI*4/6),          0],
        [-math.sin(PI*4/6), math.cos(PI*4/6),          0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*5/6), math.sin(PI*5/6),          0],
        [-math.sin(PI*5/6), math.cos(PI*5/6),          0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*6/6), math.sin(PI*6/6),         0],
        [-math.sin(PI*6/6), math.cos(PI*6/6),         0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*7/6), math.sin(PI*7/6),         0],
        [-math.sin(PI*7/6), math.cos(PI*7/6),         0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*8/6), math.sin(PI*8/6),         0],
        [-math.sin(PI*8/6), math.cos(PI*8/6),         0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*9/6), math.sin(PI*9/6),         0],
        [-math.sin(PI*9/6), math.cos(PI*9/6),         0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*10/6), math.sin(PI*10/6),         0],
        [-math.sin(PI*10/6), math.cos(PI*10/6),         0],
        [                 0,                 0,        1.0]
    ],
    [
        [ math.cos(PI*11/6), math.sin(PI*11/6),         0],
        [-math.sin(PI*11/6), math.cos(PI*11/6),         0],
        [                 0,                 0,        1.0]
    ],)
hinge_knob_rot=    (    
    [
        [ 1.0,     0,     0],
        [ 0,     1.0,     0],
        [ 0,       0,   1.0],
    ],
    [
        [ math.cos(PI/6), math.sin(PI/6),          0],
        [-math.sin(PI/6), math.cos(PI/6),          0],
        [              0,              0,        1.0]
    ],
    [
        [ math.cos(PI*2/6), math.sin(PI*2/6),          0],
        [-math.sin(PI*2/6), math.cos(PI*2/6),          0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*3/6), math.sin(PI*3/6),          0],
        [-math.sin(PI*3/6), math.cos(PI*3/6),          0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*4/6), math.sin(PI*4/6),          0],
        [-math.sin(PI*4/6), math.cos(PI*4/6),          0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*5/6), math.sin(PI*5/6),        0],
        [-math.sin(PI*5/6), math.cos(PI*5/6),        0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*6/6), math.sin(PI*6/6),        0],
        [-math.sin(PI*6/6), math.cos(PI*6/6),        0],
        [                0,               0,        1.0]
    ],
    [
        [ math.cos(PI*7/6), math.sin(PI*7/6),        0],
        [-math.sin(PI*7/6), math.cos(PI*7/6),        0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*8/6), math.sin(PI*8/6),        0],
        [-math.sin(PI*8/6), math.cos(PI*8/6),        0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*9/6), math.sin(PI*9/6),        0],
        [-math.sin(PI*9/6), math.cos(PI*9/6),        0],
        [                0,                0,        1.0]
    ],
    [
        [ math.cos(PI*10/6), math.sin(PI*10/6),        0],
        [-math.sin(PI*10/6), math.cos(PI*10/6),        0],
        [                 0,                 0,        1.0]
    ],
    [
        [ math.cos(PI*11/6), math.sin(PI*11/6),        0],
        [-math.sin(PI*11/6), math.cos(PI*11/6),        0],
        [                 0,                 0,        1.0]
    ],
    ######################  inverse  ######################
    [
        [ math.sin(PI/6), math.cos(PI/6),        0],
        [ math.cos(PI/6), -math.sin(PI/6),        0],
        [              0,              0,       -1.0]
    ],
    [
        [ math.sin(PI*2/6), math.cos(PI*2/6),        0],
        [ math.cos(PI*2/6), -math.sin(PI*2/6),        0],
        [                0,                0,       -1.0]
    ],
    [
        [ math.sin(PI*3/6), math.cos(PI*3/6),        0],
        [ math.cos(PI*3/6), -math.sin(PI*3/6),        0],
        [                0,                0,       -1.0]
    ],
    [
        [ math.sin(PI*4/6), math.cos(PI*4/6),        0],
        [ math.cos(PI*4/6), -math.sin(PI*4/6),        0],
        [                0,                0,       -1.0]
    ],
    [
        [ math.sin(PI*5/6), math.cos(PI*5/6),        0],
        [ math.cos(PI*5/6), -math.sin(PI*5/6),        0],
        [                0,                0,       -1.0]
    ],
    [
        [ math.sin(PI*6/6), math.cos(PI*6/6),        0],
        [ math.cos(PI*6/6), -math.sin(PI*6/6),        0],
        [                0,                0,       -1.0]
    ],
    [
        [ math.sin(PI*7/6), math.cos(PI*7/6),        0],
        [ math.cos(PI*7/6), -math.sin(PI*7/6),        0],
        [                0,                0,       -1.0]
    ],
    [
        [ math.sin(PI*8/6), math.cos(PI*8/6),        0],
        [ math.cos(PI*8/6), -math.sin(PI*8/6),        0],
        [                0,                0,       -1.0]
    ],
    [
        [ math.sin(PI*9/6), math.cos(PI*9/6),        0],
        [ math.cos(PI*9/6), -math.sin(PI*9/6),        0],
        [                0,                0,       -1.0]
    ],
    [
        [ math.sin(PI*10/6), math.cos(PI*10/6),        0],
        [ math.cos(PI*10/6), -math.sin(PI*10/6),        0],
        [                 0,                 0,       -1.0]
    ],
    [
        [ math.sin(PI*11/6), math.cos(PI*11/6),        0],
        [ math.cos(PI*11/6), -math.sin(PI*11/6),        0],
        [                 0,                 0,       -1.0]
    ],
    [
        [ math.sin(PI*12/6), math.cos(PI*12/6),        0],
        [ math.cos(PI*12/6), -math.sin(PI*12/6),        0],
        [                 0,                 0,       -1.0]
    ],
)

def get_symmetry_tfs(class_name,allow_reflection=True):
  tfs = []
  if class_name in ["mug"]:
    for yangle in np.arange(0,360,180)/180*np.pi:
        tf = euler_matrix(0,yangle,0,axes='sxyz')
        tfs.append(tf)
  elif class_name in ["hinge_knob"]:
    for rotation in hinge_knob_rot:
      T = np.eye(4)
      T[:3, :3] = rotation
      tfs.append(T)
  elif class_name in ["hinge_handle","line_fixed_handle"]:
    for zangle in np.arange(0,360,180)/180*np.pi:
      tf = euler_matrix(0,0,zangle,axes='sxyz')
      tfs.append(tf)
  elif class_name in ["round_fixed_handle"]:
    for rotation in type3:
      T = np.eye(4)
      T[:3, :3] = rotation
      tfs.append(T)
  else:
    tf = euler_matrix(0,0,0,axes='sxyz')
    tfs.append(tf)
  if not allow_reflection:
    new_tfs = []
    for i in range(len(tfs)):
      if np.linalg.det(tfs[i][:3,:3])<0:
        continue
      new_tfs.append(tfs[i])
    tfs = new_tfs
  return np.array(tfs)
