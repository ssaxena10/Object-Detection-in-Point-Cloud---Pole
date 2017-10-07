from math import cos, sin, pi, sqrt
import pcl
import numpy as np

csv = open('/Users/sharul/desktop/geo/final project/final_project_data/final_project_point_cloud.fuse', 'rb')
point_details = []

def cartesian(lat,lon, elevation):
    r = 6378137.0 + elevation
    f = 1.0 / 298.257224
    cos_Lat = cos(lat * pi / 180.0)
    sin_Lat = sin(lat * pi / 180.0)
    cos_Lon = cos(lon * pi / 180.0)
    sin_Lon = sin(lon * pi / 180.0)
    C = 1.0 / sqrt(cos_Lat * cos_Lat + (1 - f) * (1 - f) * sin_Lat * sin_Lat)
    S = (1.0 - f) * (1.0 - f) * C
    xaxis = (r * C + 0.0) * cos_Lat * cos_Lon
    yaxis = (r * C + 0.0) * cos_Lat * sin_Lon
    zaxis = (r * S + 0.0) * sin_Lat
    return xaxis, yaxis, zaxis

#file in form of: [latitude] [longitude] [altitude] [intensity]
#write original point cloud data into a file
for line in csv:
    points = []
    r = line.strip().split(' ')
    x, y, z = cartesian(float(r[0]), float(r[1]), float(r[2]))
    points.append(x)
    points.append(y)
    points.append(z)
    point_details.append(points)

original = open('Original_pointcloud.obj', 'w')

for point in point_details:
    line = "v " + str(point[0]) + " " + str(point[1]) + " "+ str(point[2])
    original.write(line)
    original.write("\n")

original.close()

point_details = np.array(point_details, dtype=np.float32)
p = pcl.PointCloud()
p.from_array(point_details)

print ("Original point cloud Data")
print (p)

f1 = p.make_statistical_outlier_filter()
f1.set_mean_k(50)
f1.set_std_dev_mul_thresh(5.0)

f1.set_negative(False)
filtered_cloud = f1.filter()

print ("Filtered without outliers")
print (filtered_cloud)

m1 = open('intermediate1.obj', 'w')
for point in filtered_cloud:
    line = "v " + str(point[0]) + " " + str(point[1]) + " "+ str(point[2])
    m1.write(line)
    m1.write("\n")
m1.close()

kd = filtered_cloud.make_kdtree_flann()
ind, sqr_dist = kd.nearest_k_search_for_cloud(filtered_cloud, 1000)


distances = np.sum(sqr_dist, axis=1)
remove_indices = []


for i in xrange(np.shape(distances)[0]):
    if distances[i] < 5000.0:
        remove_indices.extend(ind[i])
remove_unique_indices = set(remove_indices)

filtered_cloud = filtered_cloud.extract(remove_unique_indices, negative=True)
print ("Filtered  without large components")
print (filtered_cloud)

m2 = open('intermediate2.obj', 'w')
for point in filtered_cloud:
    line = "v " + str(point[0]) + " " + str(point[1]) + " "+ str(point[2])
    m2.write(line)
    m2.write("\n")
m2.close()


#cylindrical segmentation which discard all indices those are not found within isolated cylindrical segments
s = filtered_cloud.make_segmenter_normals(ksearch=50)
s.set_optimize_coefficients(True)
s.set_model_type(pcl.SACMODEL_CYLINDER)
s.set_normal_distance_weight(0.1)
s.set_method_type(pcl.SAC_RANSAC)
s.set_max_iterations(1000)
s.set_distance_threshold(20)
s.set_radius_limits(0, 10) # pole radious
segment_indices, model = s.segment()


#return just cylindrical segments - FEATURE EXTRACTION
filtered_cloud = filtered_cloud.extract(segment_indices, negative=False)
print ("Cloud after cylindrical segmentation")
print (filtered_cloud)


#segment ground plane, and then discard all indices associated with plane
s = filtered_cloud.make_segmenter_normals(ksearch=50)
s.set_optimize_coefficients(True)
s.set_model_type(pcl.SACMODEL_PLANE)
s.set_method_type(pcl.SAC_RANSAC)
s.set_normal_distance_weight(0.1)
s.set_distance_threshold(85)
s.set_max_iterations(100)
indices, model = s.segment()

#return indices which are not in identified ground plane
filtered_cloud = filtered_cloud.extract(indices, negative=True)

#filter cloud data based on height
fl = filtered_cloud.make_passthrough_filter()
fl.set_filter_field_name("x")
fl.set_filter_limits(0, 4364071.0)

#apply height based filter and return which are not in bounds
filtered_cloud = fl.filter()
print ("Final point cloud Data")
print (filtered_cloud)


#write final data file
final_output = open('final.obj', 'w')
for points in filtered_cloud:
    line = "v " + str(points[0]) + " " + str(points[1]) + " "+ str(points[2])
    final_output.wrilte(line)
    final_output.write("\n")
final_output.close()
