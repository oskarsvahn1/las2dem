import laspy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import matplotlib.image
import cv2
import rasterio
from rasterio.transform import from_origin

from shapely.geometry import LineString, mapping
import fiona

def create_dem(las_data, grid_resolution):
    xyz = np.column_stack((las_data.x, las_data.y, las_data.z))
    classification = las_data.raw_classification
    filter_ground = classification == 2

    offset = np.array([las_data.header.x_min, las_data.header.y_min, las_data.header.z_offset])
    xyz -= offset
    xyz = xyz[filter_ground]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int((xmax - xmin) / grid_resolution) + 3
    ny = int((ymax - ymin) / grid_resolution) + 3

    # Initialize grid for DEM
    dem = np.zeros((ny, nx))
    counts = np.zeros((ny, nx))
    intensity = np.zeros((ny, nx))

    x_scaled = (x - xmin) / grid_resolution
    y_scaled = (y - ymin) / grid_resolution
    
    indices_x = np.round(x_scaled).astype(int)
    indices_y = np.round(y_scaled).astype(int)

    d_x = indices_x - x_scaled
    d_y = indices_y - y_scaled

    weights = 2 - np.abs(d_x) - np.abs(d_y)

    np.add.at(dem, (indices_y, indices_x), z * weights)
    np.add.at(intensity, (indices_y, indices_x), las_data.intensity[filter_ground] * weights)
    np.add.at(counts, (indices_y, indices_x), weights)

    # Add contributions to adjacent cells
    for i, j in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        np.add.at(dem, (indices_y + i, indices_x + j), z * (2 - np.abs(d_x - j) - np.abs(d_y - i)))
        np.add.at(intensity, (indices_y + i, indices_x + j),
                  las_data.intensity[filter_ground] * (2 - np.abs(d_x - j) - np.abs(d_y - i)))
        np.add.at(counts, (indices_y + i, indices_x + j), (2 - np.abs(d_x - j) - np.abs(d_y - i)))

    # Avoid division by zero when computing the average
    dem_filtered = np.where(counts > 0, dem / counts, np.nan)
    intensity_filtered = np.where(counts > 0, intensity / counts, 0)


    mask = np.isnan(dem_filtered)

    dem0 = dem_filtered.copy()
    dem_filtered[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), dem0[~mask])
    dem0.T[mask.T] = np.interp(np.flatnonzero(mask.T), np.flatnonzero(~mask.T), dem0.T[~mask.T])
    dem_filtered = (dem_filtered + dem0)/2

    return dem_filtered, intensity_filtered, xmin, xmax, ymin, ymax, offset



def create_contours(dem, xmin, xmax, ymin, ymax, offset, interval=5):
    x_coords = np.linspace(xmin, xmax, dem.shape[1])
    y_coords = np.linspace(ymin, ymax, dem.shape[0])
    X, Y = np.meshgrid(x_coords, y_coords)
    low = int(np.nanmin(dem))
    levels = np.arange(low - low%interval, np.ceil(np.nanmax(dem)) + interval, interval)
    contour_lines = []
    contour = plt.contour(X, Y, dem, levels=levels)
    # plt.show()
    for lev in contour.allsegs[1:]:
        for seg in lev:
            new_curves = np.where(np.abs(np.diff(seg, axis=0)).sum(1)>5)[0]+1
            if new_curves.size == 0:
                # plt.plot(seg[:,0], seg[:,1])
                contour_lines.append(LineString(seg))
                continue
            # plt.plot(seg[0:new_curves[0],0], seg[0:new_curves[0],1])
            contour_lines.append(LineString(seg[:new_curves[0]]))

            for i, curve in enumerate(new_curves[:-1]):
                # plt.plot(seg[curve:new_curves[i+1], 0], seg[curve:new_curves[i+1], 1])
                contour_lines.append(LineString(seg[curve:new_curves[i+1]]))
            # plt.plot(seg[new_curves[-1]:,0], seg[new_curves[-1]:,1])
            contour_lines.append(LineString(seg[new_curves[-1]:]))
    return contour_lines

def normilize_img(img):
    max = dem.max()
    min = dem.min()
    img = 255*img/(max-min)
    img -= img.min()

    return img

def normilize_gray_img(img):
    return np.clip((img*80+128), 0, 255)


def export_geotiff(dem, xmin, xmax, ymin, ymax, output_file):
    # Define the geotransform (coordinate system and resolution) for the output GeoTIFF
    pixel_width = 1
    pixel_height = 1

    # Calculate the affine transformation matrix
    transform = from_origin(xmin, ymax, pixel_width, pixel_height)

    # Open a new GeoTIFF file for writing
    with rasterio.open(output_file, 'w', driver='GTiff', height=dem.shape[0], width=dem.shape[1],
                       count=dem.shape[2], dtype=str(dem.dtype), crs='+proj=utm +zone=33 +ellps=GRS80 +units=m +no_defs',
                       transform=transform) as dst:
        # Write the DEM data to the GeoTIFF
        dst.write(dem.transpose(2, 0, 1))


def save_shp(contour_lines, output_shapefile):
    schema = {'geometry': 'LineString', 'properties': {'id': 'int'}}
    crs = {
        'proj': 'tmerc',
        'lat_0': 0,
        'lon_0': 15,
        'k': 0.9996,
        'x_0': 500000,
        'y_0': 0,
        'ellps': 'GRS80',
        'units': 'm',
        'no_defs': True
    }
    with fiona.open(output_shapefile, 'w', 'ESRI Shapefile', schema, crs=crs) as c:
        for i, line in enumerate(contour_lines):
            c.write({'geometry': mapping(line), 'properties': {'id': i}})


def dem2phase(dem):
    sobelx = scipy.ndimage.sobel(dem, 1)
    sobely = scipy.ndimage.sobel(dem, 0)

    # k  = np.array([0.030320,  0.249724,  0.439911,  0.249724,  0.030320])
    # d  =  np.array([0.104550,  0.292315,  0.000000, -0.292315, -0.104550])
    # # d2 = [0.232905  0.002668 -0.471147  0.002668  0.232905]
    # from scipy.signal import convolve2d, sepfir2d
    # dx = sepfir2d(dem, d, k)
    # dy = sepfir2d(dem, k, d)

    matplotlib.image.imsave('sobelx.png', np.flipud(normilize_gray_img(sobelx)), cmap='gray')
    matplotlib.image.imsave('sobely.png', 255-np.flipud(normilize_gray_img(sobely)), cmap='gray')

    angles = cv2.phase(sobelx, sobely)
    mag = np.clip(cv2.magnitude(sobelx, sobely)*80, 0, 255)

    hsvimg = np.zeros(sobelx.shape + (3,), dtype=np.uint8)
    hsvimg[:,:,0] = angles*90/np.pi
    hsvimg[:,:,1] = np.ones_like(mag, dtype=np.uint8)*255
    hsvimg[:,:,2] = mag
    bgrimg = cv2.cvtColor(hsvimg, cv2.COLOR_HSV2BGR)
    return bgrimg


if __name__ == "__main__":
    las_data = laspy.read("21C011_647_53_2525.laz")

    grid_resolution = 1

    dem, intencity, xmin, xmax, ymin, ymax, offset = create_dem(las_data, grid_resolution)
    plt.imsave('intensity.png', np.clip(np.flipud(intencity / 40), 0, 255), cmap='gray')

    bgrimg = dem2phase(dem)
    matplotlib.image.imsave('phase.png', np.flipud(bgrimg))
    export_geotiff(np.flipud(bgrimg), xmin+offset[0], xmax+offset[0], ymin+offset[1], ymax+offset[1], "phase.tif")

    contour_lines = create_contours(dem, xmin, xmax, ymin, ymax, offset, 1)
    save_shp(contour_lines, "contours.shp")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    matplotlib.image.imsave('dem2.png', np.flipud(normilize_img(dem)), cmap='gray' )
    matplotlib.pyplot.imshow(np.flipud(normilize_img(dem)), cmap="gray")
   

