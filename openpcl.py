from plyfile import PlyData, PlyElement
import numpy as np
import os
import kit


file_path_1 = '/home/tongtue8301/project/PoLoPCAC/data/opacity_decompressed/point_cloud.ply.bin.ply'
file_path_2 = '/home/tongtue8301/project/PoLoPCAC/data/bicycle/point_cloud.ply'


ply_data1 = PlyData.read(file_path_1)
ply_data2 = PlyData.read(file_path_2)
# print(ply_data1)
# print(ply_data2)

ply_data1['vertex']['opacity'] = kit.inverse_sigmoid(ply_data1['vertex']['opacity']/1000)

# 첫 번째 몇 개의 점 정보 출력
# first_points_1 = ply_data1['vertex']['opacity'].data[-995245:-995240]  # 첫 5개의 점 정보
# first_points_2 = ply_data2['vertex']['opacity'].data[-995245:-995240]  # 첫 5개의 점 정보
# first_points_1 = np.array(ply_data1['vertex']['opacity'].mean())  # 첫 5개의 점 정보
# first_points_2 = np.array(ply_data2['vertex']['opacity'].mean())  # 첫 5개의 점 정보
# max_first_points_1, min_first_point_1 = ply_data1['vertex']['opacity'].max(), ply_data1['vertex']['opacity'].min()  # 첫 5개의 점 정보
# max_first_points_2, min_first_point_2 = ply_data2['vertex']['opacity'].max(), ply_data2['vertex']['opacity'].min()  # 첫 5개의 점 정보

# print(max_first_points_1, min_first_point_1, max_first_points_2, min_first_point_2)


# for point1 in first_points_1:
#     print(point1)
    
# # print(first_points_1)
# # print(first_points_2)

# for point in first_points_2:
#     print(point)
    
# # 2. Vertex 데이터 접근 (필요한 속성만 선택)
# x = vertex_data['x']
# y = vertex_data['y']
# z = vertex_data['z']
# opacity = vertex_data['opacity']


ply_data2['vertex']['x'] = ply_data1['vertex']['x'].data
ply_data2['vertex']['y'] = ply_data1['vertex']['y'].data
ply_data2['vertex']['z'] = ply_data1['vertex']['z'].data
ply_data2['vertex']['opacity'] = ply_data1['vertex']['opacity'].data

# first_points_1 = ply_data1['vertex']['x'].data[:5]  # 첫 5개의 점 정보
# first_points_2 = ply_data2['vertex']['x'].data[:5]  # 첫 5개의 점 정보

# for point in first_points_1:
#     print(point)

# for point in first_points_2:
#     print(point)


ply_data2.write('/home/tongtue8301/project/PoLoPCAC/data/bicycle_compress_sum/point_cloud.ply')

# points = ply_data2['vertex']['opacity'].max()
# print(points)


# # PLY 파일 경로
# decomp_file_path = '/home/songing/PycharmProjects/pclc/PoLoPCAC/data/Sematic_decompressed/filtered_xyz_rgb.ply.bin.ply'
# bef_comp_file_path = '/home/songing/PycharmProjects/pclc/PoLoPCAC/filtered_xyz_rgb.ply'
# # PLY 파일 읽기
# bef_ply_data = PlyData.read(bef_comp_file_path)
# decomp_data = PlyData.read(decomp_file_path)
#
# # PLY 파일의 메타데이터와 첫 번째 몇 개의 점 정보 출력
# print(bef_ply_data)
# print(decomp_data)
#
#
# bef_ply_data_size = os.path.getsize(bef_comp_file_path)
# comp_ply_data_size = os.path.getsize(decomp_file_path)
#
# print(bef_ply_data_size, comp_ply_data_size)
#
#
# # 각 파일의 포인트 수 계산 (vertex 데이터 길이)
# bef_points_count = len(bef_ply_data['vertex'].data)
# decomp_points_count = len(decomp_data['vertex'].data)
#
# # 파일 크기를 비트로 변환
# bef_comp_size_bits = bef_ply_data_size * 8
# decomp_size_bits = comp_ply_data_size * 8
#
# # 각 파일의 bpp 계산 (bit-per-point)
# bef_bpp = bef_comp_size_bits / bef_points_count
# decomp_bpp = decomp_size_bits / decomp_points_count
#
# # 결과 출력
# print(f"Before Compression: {bef_ply_data_size} bytes, {bef_points_count} points, bpp: {bef_bpp:.2f} bits per point")
# print(f"After Decompression: {comp_ply_data_size} bytes, {decomp_points_count} points, bpp: {decomp_bpp:.2f} bits per point")
#

# # 첫 번째 몇 개의 점 정보 출력
# first_points = ply_data['vertex'].data[:5]  # 첫 5개의 점 정보
# for point in first_points:
#     print(point)

#
# # 2. Vertex 데이터 접근 (필요한 속성만 선택)
# vertex_data = ply_data['vertex'].data
# x = vertex_data['x']
# y = vertex_data['y']
# z = vertex_data['z']
# opacity = vertex_data['opacity']
# scale_0 = vertex_data['scale_0']
# scale_1 = vertex_data['scale_1']
# scale_2 = vertex_data['scale_2']
# f_dc_0 = vertex_data['f_dc_0']
# f_dc_1 = vertex_data['f_dc_1']
# f_dc_2 = vertex_data['f_dc_2']
# rot_0 = vertex_data['rot_0']
# rot_1 = vertex_data['rot_1']
# rot_2 = vertex_data['rot_2']
# rot_3 = vertex_data['rot_3']
#
# # 3. 새로운 구조의 Vertex 배열 만들기
# filtered_vertices = np.array(list(zip(x, y, z, opacity, scale_0, scale_1, scale_2,f_dc_0, f_dc_1,f_dc_2, rot_0, rot_1, rot_2, rot_3)),
#                              dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('opacity', 'f4'),
#                                     ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
#                                     ('f_dc_0', 'f4'), ('f_dc_1', 'f4'),('f_dc_2', 'f4'),
#                                     ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'),
#                                     ('rot_3', 'f4')])
#
# # 4. 새로운 PlyElement 생성
# vertex_element = PlyElement.describe(filtered_vertices, 'vertex')
#
# # 5. 새로운 PLY 파일로 저장
# output_file = 'filtered_gaussian_att.ply'
# PlyData([vertex_element]).write(output_file)
#
# print(f"Filtered PLY file saved as {output_file}")
