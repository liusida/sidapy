import time
import sidapy
import numpy as np
from scipy.ndimage import label as scipy_label
np.random.seed(0)

start = time.time()
map = np.random.randint(low=0, high=2, size=[5,5], dtype=np.int8) # use np.int8 to save space
print(map)
labeled_map = sidapy.label(map)
print(labeled_map)
print(labeled_map.dtype)
end = time.time()
print(end - start)

map = np.random.randint(low=0, high=2, size=[512,512], dtype=np.int8)

print("==sidapy.label==")
start = time.time()
# print(map)
labeled_map = sidapy.label(map)
print(labeled_map)
print(labeled_map.dtype)
end = time.time()
print(end - start)

print("==scipy_label==")
start = time.time()
# map = np.random.randint(low=0, high=2, size=[512,512], dtype=np.int8)
# print(map)
labeled_map,_ = scipy_label(map)
print(labeled_map)
print(labeled_map.dtype)
end = time.time()
print(end - start)

print("==sidapy.label==")
start = time.time()
# print(map)
labeled_map = sidapy.label(map)
print(labeled_map)
print(labeled_map.dtype)
end = time.time()
print(end - start)

print("==scipy_label==")
start = time.time()
# map = np.random.randint(low=0, high=2, size=[512,512], dtype=np.int8)
# print(map)
labeled_map,_ = scipy_label(map)
print(labeled_map)
print(labeled_map.dtype)
end = time.time()
print(end - start)
