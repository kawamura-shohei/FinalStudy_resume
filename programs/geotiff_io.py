from osgeo import gdal, gdal_array
import numpy as np
import cv2


# tiff画像読み込み関数
# 引数は入力画像geotiffのpath
# 戻り値は入力画像geotiffのデータ(gdal.Openしたもの)
def read_geotiff(path):
    # 入力画像読み込み
    src = gdal.Open(path)

    return src


# tiffをCV2で処理しやすくする関数(1バンド)
# 引数は入力画像geotiffのデータ(gdal.Openしたもの)
# 戻り値はCV2で処理しやすいようデータをndarrayに格納したもの
def cvtTiff2CV2_1band_to_1band(src):
    # 1バンドをnumpy.ndarrayとして格納
    src_origin = src.GetRasterBand(1).ReadAsArray()

    return src_origin


# tiffをCV2で処理しやすくする関数(1バンドを3バンドに拡張)
# 引数は入力画像geotiffのデータ(gdal.Openしたもの)
# 戻り値はCV2で処理しやすいようデータをndarrayに格納したもの
def cvtTiff2CV2_1band_to_3band(src):
    # 1バンドをnumpy.ndarrayとして格納
    src_origin = src.GetRasterBand(1).ReadAsArray()
    # 3次元配列に変換
    src_R = src_origin.reshape(src_origin.shape[0], src_origin.shape[1], 1)
    src_G = src_origin.reshape(src_origin.shape[0], src_origin.shape[1], 1)
    src_B = src_origin.reshape(src_origin.shape[0], src_origin.shape[1], 1)
    src_CV2 = np.array(np.concatenate((src_B, src_G, src_R), axis=2), dtype=np.uint8)

    return src_CV2

# tiffをCV2で処理しやすくする関数(3バンド)
# 引数は入力画像geotiffのデータ(gdal.Openしたもの)
# 戻り値はCV2で処理しやすいようデータをndarrayに格納したもの
def cvtTiff2CV2_3band(src):
    # 1, 2, 3バンドをnumpy.ndarrayとして格納
    src_b1_origin = src.GetRasterBand(1).ReadAsArray()
    src_b2_origin = src.GetRasterBand(2).ReadAsArray()
    src_b3_origin = src.GetRasterBand(3).ReadAsArray()
    # 3次元配列に変換
    src_R = src_b1_origin.reshape(src_b1_origin.shape[0], src_b1_origin.shape[1], 1)
    src_G = src_b2_origin.reshape(src_b2_origin.shape[0], src_b2_origin.shape[1], 1)
    src_B = src_b3_origin.reshape(src_b3_origin.shape[0], src_b3_origin.shape[1], 1)
    # 色をBGR順に結合(3,2,1の順にバンドを結合), とりあえずdtypeはunit8を指定
    src_b123_CV2_origin = np.array(np.concatenate((src_B, src_G, src_R), axis=2), dtype=np.uint8)

    return src_b123_CV2_origin


# バンド数が1の場合のtiff画像書き込み関数
# 引数は順に 出力画像geotiffのpath, 入力画像geotiffのデータ(gdal.Openしたもの), 処理済みのndarray
# 戻り値はとりあえずなし
def write_geotiff_1(outfile_path, src, edit_b):
    # X,Yのサイズとバンド数を求める
    xsize = src.RasterXSize
    ysize = src.RasterYSize
    band = 1

    # データタイプ番号
    dtype = src.GetRasterBand(1).DataType # 型番号 (ex: 6 -> numpy.float32)
    gdal_array.GDALTypeCodeToNumericTypeCode(dtype) # 型番号 -> 型名 変換

    # 出力画像
    output = gdal.GetDriverByName('GTiff').Create(outfile_path, xsize, ysize, band, dtype)

    # 座標系指定
    output.SetGeoTransform(src.GetGeoTransform())

    # 空間情報を結合
    output.SetProjection(src.GetProjection())
    output.GetRasterBand(1).WriteArray(edit_b)
    output.FlushCache()


# バンド数が3の場合のtiff画像書き込み関数
# 引数は順に 出力画像geotiffのpath, 入力画像geotiffのデータ(gdal.Openしたもの), 処理済みのndarray
# 戻り値はとりあえずなし
def write_geotiff_3(outfile_path, src, edit_b):
    # X,Yのサイズとバンド数を求める
    xsize = src.RasterXSize
    ysize = src.RasterYSize
    band = 3

    # 第1~3バンドを格納
    b3,b2,b1 = cv2.split(edit_b)

    # データタイプ番号
    dtype = src.GetRasterBand(1).DataType # 型番号 (ex: 6 -> numpy.float32)
    gdal_array.GDALTypeCodeToNumericTypeCode(dtype) # 型番号 -> 型名 変換

    # 出力画像
    output = gdal.GetDriverByName('GTiff').Create(outfile_path, xsize, ysize, band, dtype)

    # 座標系指定
    output.SetGeoTransform(src.GetGeoTransform())

    # 空間情報を結合
    output.SetProjection(src.GetProjection())
    output.GetRasterBand(1).WriteArray(b1)
    output.GetRasterBand(2).WriteArray(b2)
    output.GetRasterBand(3).WriteArray(b3)
    output.FlushCache()


# (0~255)正規化を行う関数
# 引数は順に 正規化したい画像のndarray, 最大の画素値, 最小の画素値
# 戻り値は　正規化されたndarray
def normalized(src_b1, max, min):
    src_b1 = (src_b1 - min)*255 / (max - min)

    return src_b1