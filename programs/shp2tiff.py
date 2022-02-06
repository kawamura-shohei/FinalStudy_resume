from osgeo import gdal, ogr

#making the shapefile as an object.
input_shp= ogr.Open("C:/Users/kawamurashohei/B4_semi/QGIS_DEM_practice/AtamiData03_BuildingMask/AtamiDosya_Before_Building_resize.shp")
#getting layer information of shapefile.
shp_layer= input_shp.GetLayer()
#pixel_size determines the size of the new raster.
#pixel_size is proportional to size of shapefile.
pixel_size= 0.2
#get extent values to set size of output raster.
x_min, x_max, y_min, y_max= shp_layer.GetExtent()
#calculate size/resolution of the raster.
x_res= int((x_max -x_min) /pixel_size)
y_res= int((y_max -y_min) /pixel_size)
#get GeoTiff driver by
image_type= 'GTiff'
driver= gdal.GetDriverByName(image_type)
#passing the filename, x and y direction resolution, no. of bands, new raster.
new_raster= driver.Create("C:/Users/kawamurashohei/B4_semi/GeoTiff_practice/pra06_data/src06/AtamiDosya_Before_Building_Mask_nodata0.tif", x_res, y_res, 1, gdal.GDT_Byte)
#transforms between pixel raster space to projection coordinate space.
new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))
#get required raster band.
band= new_raster.GetRasterBand(1)
#assign no data value to empty cells.
no_data_value= 0
band.SetNoDataValue(no_data_value)
band.FlushCache()
#main conversion method
gdal.RasterizeLayer(new_raster, [1], shp_layer)
#adding a spatial reference
# new_rasterSRS= osr.SpatialReference()
# new_rasterSRS.ImportFromEPSG(2450)
# new_raster.SetProjection(new_rasterSRS.ExportToWkt())
# return gdal.Open(output_raster)