#!/usr/bin/env python
#coding=utf8

# Copyright (c) 2016 Satya Mallick <spmallick@learnopencv.com>
# All rights reserved. No warranty, explicit or implicit, provided.
# Modifications and small fixes by Steffen Kühne, 2018

import os
import math
import sys
import cv2
import numpy as np

if len(sys.argv) < 2:
    print(
        'Missing arguments. Please provide a path to the image folder.\n'
        'The folder should contain both images (.jpg) and landmarks (.txt)\n'
        'Usage example: python average.py ./images\n'
    )
    exit()

def main():
    # Default size of the output image
    w = 170
    h = 240

    # Overwrite default image size
    if len(sys.argv) == 3:
        if str.isdigit(sys.argv[2]):
            w = int(sys.argv[2])
        if str.isdigit(sys.argv[3]):
            h = int(sys.argv[3])

    path = sys.argv[1]

    # Read points for all images
    all_points = read_points(path)
    # Read all images
    images = read_images(path)

    # Eye corners
    eyecorner_dst = [
        (np.int(0.3 * w), np.int(h / 3)),
        (np.int(0.7 * w), np.int(h / 3))
    ]

    images_norm = []
    points_norm = []

    # Add boundary points for delaunay triangulation
    boundary_pts = np.array([
        (0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2),
        (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)
    ])

    # Initialize location of average points to 0s
    points_avg = np.array(
        [(0, 0)] * (len(all_points[0]) + len(boundary_pts)),
        np.float32()
    )

    num_images = len(images)

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    for i in range(0, num_images):

        points1 = all_points[i]

        # Corners of the eye in input image
        eyecorner_src = [all_points[i][36], all_points[i][45]]

        # Compute similarity transform
        tform = similarity_transform(eyecorner_src, eyecorner_dst)

        # Apply similarity transformation
        img = cv2.warpAffine(images[i], tform, (w, h))

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68, 1, 2))
        points = cv2.transform(points2, tform)
        points = np.float32(np.reshape(points, (68, 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundary_pts, axis=0)

        # Calculate location of average landmark points.
        points_avg = points_avg + points / num_images

        points_norm.append(points)
        images_norm.append(img)

    # Delaunay triangulation
    rect = (0, 0, w, h)
    tri = calculate_triangles(rect, np.array(points_avg))

    # Output image
    output = np.zeros((h, w, 3), np.float32())

    # Warp input images to average image landmarks
    for i in range(0, len(images_norm)):
        img = np.zeros((h, w, 3), np.float32())
        # Transform triangles one by one
        for j in range(0, len(tri)):
            t_in = []
            t_out = []

            for k in range(0, 3):
                p_in = points_norm[i][tri[j][k]]
                p_in = constrain_point(p_in, w, h)

                p_out = points_avg[tri[j][k]]
                p_out = constrain_point(p_out, w, h)

                t_in.append(p_in)
                t_out.append(p_out)

            warp_triangle(images_norm[i], img, t_in, t_out)

        # Add image intensities for averaging
        output = output + img

    # Divide by num_images to get average
    output = output / num_images

    # Display result
    # cv2.imshow('image', output)
    # cv2.waitKey(0)
    # Saving result
    cv2.imwrite( "average_face.jpg", 255 * output)
  
# Read points from text files in directory
def read_points(path):
    # Create an array of array of points.
    points_array = []

    # List all files in the directory and read points from text files one by one
    for file_path in sorted(os.listdir(path)):
        print(file_path)

        if file_path.endswith('.txt'):
            # Create an array of points.
            points = []

            # Read points from file_path
            with open(os.path.join(path, file_path)) as f:
                for line in f:
                    x, y = line.split()
                    points.append((int(x), int(y)))

            # Store array of points
            points_array.append(points)

    return points_array

# Read all jpg images in folder.
def read_images(path):
    #Create array of array of images.
    images_array = []

    #List all files in the directory and read points from text files one by one
    for file_path in sorted(os.listdir(path)):
        if file_path.endswith('.jpg'):
            # Read image found.
            img = cv2.imread(os.path.join(path, file_path))

            # Convert to float_ing point
            img = np.float32(img) / 255.0

            # Add to array of images
            images_array.append(img)

    return images_array

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.

def similarity_transform(in_points, out_points):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    in_pts = np.copy(in_points).tolist()
    out_pts = np.copy(out_points).tolist()

    xin = c60 * (in_pts[0][0] - in_pts[1][0]) - s60 * \
        (in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    yin = s60 * (in_pts[0][0] - in_pts[1][0]) + c60 * \
        (in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]

    in_pts.append([np.int(xin), np.int(yin)])

    xout = c60 * (out_pts[0][0] - out_pts[1][0]) - s60 * \
        (out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    yout = s60 * (out_pts[0][0] - out_pts[1][0]) + c60 * \
    (out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]

    out_pts.append([np.int(xout), np.int(yout)])

    tform = cv2.estimateAffinePartial2D(np.array([in_pts]), np.array([out_pts]));
    
    return tform[0]

# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Calculate Delanauy triangles
def calculate_triangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangle_list = subdiv.getTriangleList()
    # Find the indices of triangles in the points array
    delaunay_tri = []

    for t in triangle_list:
        pt = []

        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                delaunay_tri.append((ind[0], ind[1], ind[2]))

    return delaunay_tri

def constrain_point(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))

    return p

# Apply affine transform calculated using src_tri and dst_tri to src and
# output an image of size.
def apply_affine_transform(src, src_tri, dst_tri, size):

    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warp_triangle(img1, img2, t1, t2):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

if __name__ == '__main__':
    main()
